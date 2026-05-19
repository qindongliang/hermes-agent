import sys
import threading
import types
from types import SimpleNamespace

import pytest

from gateway import ambient_gate
import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _event(text: str = "今天下雨了"):
    source = SimpleNamespace(
        platform=SimpleNamespace(value="feishu"),
        chat_type="group",
        is_bot=False,
        chat_name="Hermes Agent Dev Group",
        user_name="spark",
    )
    event = SimpleNamespace(
        text=text,
        source=source,
        reply_to_text=None,
        _feishu_ambient_candidate=True,
        is_command=lambda: False,
    )
    return event


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message, conversation_history=None, task_id=None, persist_user_message=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None)
    runner.session_store = SimpleNamespace()
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_123",
        chat_type="group",
        user_id="ou_123",
    )


def _bot_source() -> SessionSource:
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_123",
        chat_type="group",
        user_id="ou_bot",
        user_name="ops",
        is_bot=True,
    )


def _human_source() -> SessionSource:
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_123",
        chat_type="group",
        user_id="ou_human",
        user_name="spark",
        is_bot=False,
    )


def test_is_feishu_ambient_candidate_requires_feishu_group_human_text():
    event = _event()

    assert ambient_gate.is_feishu_ambient_candidate(event)

    event.source.is_bot = True
    assert not ambient_gate.is_feishu_ambient_candidate(event)


def test_load_ambient_gate_config_from_feishu_section(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        """
feishu:
  ambient_listening:
    enabled: true
    confidence_threshold: 0.72
    uncertain_policy: respond
    timeout_seconds: 3
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(ambient_gate, "get_hermes_home", lambda: tmp_path)

    config = ambient_gate.load_ambient_gate_config()

    assert config.enabled is True
    assert config.confidence_threshold == 0.72
    assert config.uncertain_policy == "respond"
    assert config.timeout_seconds == 3
    assert config.retry_on_timeout is True
    assert config.retry_timeout_seconds == 12
    assert config.timeout_notice is True


def test_ambient_relevance_relevant_above_threshold(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "feishu:\n  ambient_listening:\n    enabled: true\n",
        encoding="utf-8",
    )
    (tmp_path / "SOUL.md").write_text("你是后端工程师。", encoding="utf-8")
    monkeypatch.setattr(ambient_gate, "get_hermes_home", lambda: tmp_path)

    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "call_llm",
        lambda **kwargs: _response(
            '{"decision":"relevant","confidence":0.91,"reason":"backend issue","reply_intent":"triage API failure"}'
        ),
    )

    decision = ambient_gate._evaluate_sync(_event("接口 500 了"), ambient_gate.load_ambient_gate_config())

    assert decision.should_respond
    assert decision.decision == "relevant"
    assert decision.reply_intent == "triage API failure"


def test_ambient_relevance_retries_timeout_once(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        """
feishu:
  ambient_listening:
    enabled: true
    timeout_seconds: 3
    retry_timeout_seconds: 9
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(ambient_gate, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(ambient_gate, "_retry_jitter_seconds", lambda event: 0.0)

    from agent import auxiliary_client

    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("first gate timeout")
        return _response('{"decision":"relevant","confidence":0.91,"reason":"backend issue"}')

    monkeypatch.setattr(auxiliary_client, "call_llm", fake_call_llm)

    decision = ambient_gate._evaluate_sync(_event("接口 404 了"), ambient_gate.load_ambient_gate_config())

    assert decision.decision == "relevant"
    assert [call["timeout"] for call in calls] == [3, 9]


def test_ambient_relevance_timeout_after_retry_stays_uncertain(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "feishu:\n  ambient_listening:\n    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ambient_gate, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(ambient_gate, "_retry_jitter_seconds", lambda event: 0.0)

    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "call_llm",
        lambda **kwargs: (_ for _ in ()).throw(TimeoutError("still slow")),
    )

    decision = ambient_gate._evaluate_sync(_event("接口 404 了"), ambient_gate.load_ambient_gate_config())

    assert decision.decision == "uncertain"
    assert decision.is_timeout_error
    assert "after retry" in decision.reason


def test_ambient_relevance_low_confidence_relevant_becomes_uncertain(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        """
feishu:
  ambient_listening:
    enabled: true
    confidence_threshold: 0.8
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(ambient_gate, "get_hermes_home", lambda: tmp_path)

    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "call_llm",
        lambda **kwargs: _response(
            '{"decision":"relevant","confidence":0.5,"reason":"weak signal"}'
        ),
    )

    decision = ambient_gate._evaluate_sync(_event("这个要看下"), ambient_gate.load_ambient_gate_config())

    assert decision.decision == "uncertain"
    assert not decision.should_respond


def test_ambient_relevance_invalid_output_is_uncertain(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "feishu:\n  ambient_listening:\n    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ambient_gate, "get_hermes_home", lambda: tmp_path)

    from agent import auxiliary_client

    monkeypatch.setattr(auxiliary_client, "call_llm", lambda **kwargs: _response("not json"))

    decision = ambient_gate._evaluate_sync(_event(), ambient_gate.load_ambient_gate_config())

    assert decision.decision == "uncertain"
    assert not decision.should_respond


def test_ambient_channel_prompt_keeps_response_lightweight():
    decision = ambient_gate.AmbientDecision(
        "relevant",
        0.91,
        reply_intent="triage API failure",
    )

    prompt = ambient_gate.build_ambient_channel_prompt(decision)

    assert "non-@ group message" in prompt
    assert "Tools are disabled" in prompt
    assert "do not @ another bot" in prompt
    assert "triage API failure" in prompt


def test_bot_delegation_approval_prompt_requests_human_consent():
    prompt = ambient_gate.build_bot_delegation_approval_prompt(
        "ops",
        "@后端开发backend 请继续排查 500",
    )

    assert "ops" in prompt
    assert "人类确认" in prompt
    assert "/approve" in prompt
    assert "/deny" in prompt


def test_bot_delegation_approval_prompt_hides_feishu_internal_ids():
    prompt = ambient_gate.build_bot_delegation_approval_prompt(
        "ou_4612abcdef",
        "@后端开发backend 请继续排查 500",
    )

    assert "ou_4612abcdef" not in prompt
    assert "另一个 bot" in prompt


def test_bot_delegation_approval_prompt_infers_sender_from_bot_intro():
    prompt = ambient_gate.build_bot_delegation_approval_prompt(
        "ou_4612abcdef",
        "ops 在，我来先看运行环境侧。Kanban - Doing: 接口 500 triage",
    )

    assert "ou_4612abcdef" not in prompt
    assert "ops 提议我介入" in prompt
    assert "另一个 bot" not in prompt


def test_bot_delegation_approval_prompt_uses_soul_open_id_mapping(monkeypatch, tmp_path):
    soul = tmp_path / "SOUL.md"
    soul.write_text(
        '| ops | `<at user_id="ou_90f00bc9b93673ab89cedba333fe8294">运维ops</at>` |\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(ambient_gate, "get_hermes_home", lambda: tmp_path)

    prompt = ambient_gate.build_bot_delegation_approval_prompt(
        "ou_90f00bc9b93673ab89cedba333fe8294",
        "我继续做一次实时状态确认，避免只重复口头接手。",
    )

    assert "ou_90f00bc9b93673ab89cedba333fe8294" not in prompt
    assert "ops 提议我介入" in prompt


def test_bot_delegation_approval_prompt_prefers_inline_bot_status_over_prose():
    prompt = ambient_gate.build_bot_delegation_approval_prompt(
        "ou_4612abcdef",
        (
            "我继续做一次实时状态确认，避免只重复口头接手。"
            "ops 在，我接这个 500。Kanban - Doing: 接口 500 排查 "
            "- Heartbeat: ops 在线，继续跟进。"
        ),
    )

    assert "我继续做一次实时状态确认 提议" not in prompt
    assert "ops 提议我介入" in prompt


def test_bot_delegation_approval_prompt_does_not_treat_target_mention_as_sender():
    prompt = ambient_gate.build_bot_delegation_approval_prompt(
        "ou_4612abcdef",
        "@后端开发backend 请继续排查 500",
    )

    assert "后端开发backend 提议" not in prompt
    assert "另一个 bot" in prompt


def test_bot_delegation_approval_prompt_can_omit_text_fallback_for_buttons():
    prompt = ambient_gate.build_bot_delegation_approval_prompt(
        "ops",
        "@后端开发backend 请继续排查 500",
        include_text_fallback=False,
    )

    assert "ops" in prompt
    assert "/approve" not in prompt
    assert "/deny" not in prompt


def test_bot_delegation_approval_choice_matching():
    assert gateway_run.GatewayRunner._match_feishu_bot_delegation_approval_choice(
        MessageEvent(text="/approve", source=_human_source())
    ) == "approve"
    assert gateway_run.GatewayRunner._match_feishu_bot_delegation_approval_choice(
        MessageEvent(text="同意", source=_human_source())
    ) == "approve"
    assert gateway_run.GatewayRunner._match_feishu_bot_delegation_approval_choice(
        MessageEvent(text="/deny", source=_human_source())
    ) == "cancel"
    assert gateway_run.GatewayRunner._match_feishu_bot_delegation_approval_choice(
        MessageEvent(text="随便聊聊", source=_human_source())
    ) is None


@pytest.mark.asyncio
async def test_bot_delegation_approval_replays_original_event(monkeypatch):
    runner = _make_runner()
    original = MessageEvent(
        text="@后端开发backend 请继续排查 500",
        source=_bot_source(),
    )
    approval = MessageEvent(text="/approve", source=_human_source())
    captured = {}

    async def fake_handle_message(event):
        captured["event"] = event
        return "done"

    monkeypatch.setattr(runner, "_handle_message", fake_handle_message)

    prompt = await runner._request_feishu_bot_delegation_approval(original)
    assert "/approve" in prompt

    result = await runner._resolve_feishu_bot_delegation_approval(approval, "approve")

    assert result == "done"
    assert captured["event"].text == original.text
    assert getattr(captured["event"], "_feishu_bot_delegation_approved") is True
    assert "human approved" in captured["event"].channel_prompt.lower()


@pytest.mark.asyncio
async def test_bot_delegation_approval_uses_feishu_buttons_when_available():
    runner = _make_runner()
    original = MessageEvent(
        text="ops 在，我来先看运行环境侧。@后端开发backend 这边接口 500 需要一起看。",
        source=SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_123",
            chat_type="group",
            user_id="ou_bot",
            user_name="ou_4612abcdef",
            is_bot=True,
        ),
    )
    captured = {}

    class ButtonAdapter:
        async def send_bot_delegation_approval(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(success=True)

    runner.adapters[Platform.FEISHU] = ButtonAdapter()

    result = await runner._request_feishu_bot_delegation_approval(original)

    assert result is None
    assert captured["title"] == "确认 bot 协作排查"
    assert "ou_4612abcdef" not in captured["message"]
    assert "另一个 bot" not in captured["message"]
    assert "ops 提议我介入" in captured["message"]
    assert "/approve" not in captured["message"]
    from tools import slash_confirm as _slash_confirm_mod

    _slash_confirm_mod.clear(captured["session_key"])


@pytest.mark.asyncio
async def test_bot_delegation_denial_does_not_replay(monkeypatch):
    runner = _make_runner()
    original = MessageEvent(
        text="@后端开发backend 请继续排查 500",
        source=_bot_source(),
    )
    denial = MessageEvent(text="/deny", source=_human_source())
    called = False

    async def fake_handle_message(event):
        nonlocal called
        called = True
        return "done"

    monkeypatch.setattr(runner, "_handle_message", fake_handle_message)

    await runner._request_feishu_bot_delegation_approval(original)
    result = await runner._resolve_feishu_bot_delegation_approval(denial, "cancel")

    assert "取消" in result
    assert called is False


@pytest.mark.asyncio
async def test_ambient_gate_timeout_notice_is_sent_once(monkeypatch):
    runner = _make_runner()
    sent = []

    class Adapter:
        async def send(self, chat_id, content, metadata=None):
            sent.append((chat_id, content, metadata))
            return SimpleNamespace(success=True)

    monkeypatch.setattr(runner, "_active_profile_name", lambda: "backend-eng")
    runner.adapters[Platform.FEISHU] = Adapter()
    event = MessageEvent(
        text="接口返回 404，谁来看看？",
        source=_source(),
        message_id="om_timeout_1",
    )
    decision = ambient_gate.AmbientDecision("uncertain", 0.0, "gate error: TimeoutError after retry")

    await runner._maybe_send_ambient_gate_timeout_notice(event, decision, min_interval_seconds=60)
    await runner._maybe_send_ambient_gate_timeout_notice(event, decision, min_interval_seconds=60)

    assert len(sent) == 1
    assert sent[0][0] == "oc_123"
    assert "backend-eng：相关性判断超时，未启动处理" in sent[0][1]


@pytest.mark.asyncio
async def test_ambient_response_mode_disables_agent_tools(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"provider": "openai", "api_key": "fake"},
    )

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"file", "terminal"})

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="Hermes 环境监听测试：接口返回 500，谁来看看？",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="session-1",
        session_key="agent:main:feishu:group:oc_123",
        channel_prompt=ambient_gate.build_ambient_channel_prompt(
            ambient_gate.AmbientDecision("relevant", 0.9, reply_intent="triage API failure")
        ),
        ambient_response_mode=True,
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init["enabled_toolsets"] == []
    assert "Tools are disabled" in _CapturingAgent.last_init["ephemeral_system_prompt"]
