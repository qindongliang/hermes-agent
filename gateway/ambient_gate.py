"""Ambient group-message relevance gate.

This module keeps noisy "listen to every group message" behavior out of the
main agent loop.  It is intentionally conservative: failures and uncertainty
fall back to silence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home
from utils import is_truthy_value

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.65
_DEFAULT_TIMEOUT_SECONDS = 8.0
_DEFAULT_RETRY_TIMEOUT_SECONDS = 12.0
_DEFAULT_TIMEOUT_NOTICE_MIN_INTERVAL_SECONDS = 60.0
_DEFAULT_MAX_IDENTITY_CHARS = 5000
_DEFAULT_MAX_MESSAGE_CHARS = 2000
_FEISHU_INTERNAL_ID_RE = re.compile(
    r"^(?:ou|on|u|oc|om|cli|app|bot)_[A-Za-z0-9_-]+$",
    re.IGNORECASE,
)
_BOT_SELF_INTRO_RE = re.compile(
    r"^\s*(?!@)([A-Za-z][A-Za-z0-9_-]{1,31}|[\u4e00-\u9fffA-Za-z0-9_-]{2,16})\s*(?:在|在线|已接手|接手|:|：)"
)
_BOT_HEARTBEAT_RE = re.compile(
    r"\bHeartbeat\s*[:：]\s*([A-Za-z0-9_\-\u4e00-\u9fff]{2,32})\s*(?:在|在线)\b",
    re.IGNORECASE,
)
_BOT_INLINE_STATUS_RE = re.compile(
    r"(?<![@\w])([A-Za-z][A-Za-z0-9_-]{1,31})\s*(?:在|在线|已接手|接手)(?:[，,。.\s]|$)"
)
_BOT_NAME_STOP_PREFIXES = (
    "我",
    "你",
    "他",
    "她",
    "它",
    "这",
    "那",
    "请",
    "接口",
    "避免",
    "继续",
    "需要",
    "已经",
    "仍",
    "先",
)
_SOUL_FEISHU_MENTION_ROW_RE = re.compile(
    r"\|\s*([^|`\n]+?)\s*\|\s*`?<at\s+[^>]*user_id=[\"'](ou_[A-Za-z0-9_-]+)[\"'][^>]*>(.*?)</at>`?\s*\|",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AmbientGateConfig:
    enabled: bool = False
    confidence_threshold: float = _DEFAULT_THRESHOLD
    uncertain_policy: str = "ignore"
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    retry_on_timeout: bool = True
    retry_timeout_seconds: float = _DEFAULT_RETRY_TIMEOUT_SECONDS
    timeout_notice: bool = True
    timeout_notice_min_interval_seconds: float = _DEFAULT_TIMEOUT_NOTICE_MIN_INTERVAL_SECONDS
    max_identity_chars: int = _DEFAULT_MAX_IDENTITY_CHARS
    max_message_chars: int = _DEFAULT_MAX_MESSAGE_CHARS


@dataclass(frozen=True)
class AmbientDecision:
    decision: str
    confidence: float
    reason: str = ""
    reply_intent: str = ""
    raw: str = ""

    @property
    def should_respond(self) -> bool:
        return self.decision == "relevant"

    @property
    def is_gate_error(self) -> bool:
        return self.reason.startswith("gate error:")

    @property
    def is_timeout_error(self) -> bool:
        lowered = self.reason.lower()
        return self.is_gate_error and ("timeout" in lowered or "connection" in lowered)


def build_ambient_channel_prompt(decision: AmbientDecision) -> str:
    """Build the per-turn prompt for a bot approved by the ambient gate."""
    intent = (decision.reply_intent or "Respond from your role only.").strip()
    return (
        "Ambient relevance gate approved this non-@ group message for your role.\n"
        "This is ambient listening, not an explicit task assignment. Send a brief, "
        "conversational reply from your own identity only. Tools are disabled for "
        "this ambient response, so do not pretend to inspect files, read logs, run "
        "terminal commands, search the web, or start a full investigation. If active "
        "investigation is needed, ask the user to explicitly assign or @ you. Do "
        "not create or update Kanban/status items, do not say work is in progress, "
        "and do not @ another bot. If useful, offer the first triage angle or ask "
        "for one concrete detail.\n"
        f"Gate intent: {intent}"
    )


def safe_feishu_display_name(value: str, fallback: str = "另一个 bot") -> str:
    """Return a user-facing display name without leaking Feishu internal IDs."""
    name = (value or "").strip()
    if not name or _FEISHU_INTERNAL_ID_RE.match(name):
        return fallback
    return name


def _is_likely_bot_display_name(value: str) -> bool:
    name = (value or "").strip()
    if not name:
        return False
    if any(name.startswith(prefix) for prefix in _BOT_NAME_STOP_PREFIXES):
        return False
    # Pure Chinese phrases longer than a short display name are usually prose,
    # not a bot name.
    if re.fullmatch(r"[\u4e00-\u9fff]+", name) and len(name) > 8:
        return False
    return True


def _lookup_feishu_bot_name_from_soul(open_id: str) -> str:
    """Look up a Feishu bot open_id in the active profile's SOUL.md mapping."""
    oid = (open_id or "").strip()
    if not oid.startswith("ou_"):
        return ""
    try:
        soul_path = get_hermes_home() / "SOUL.md"
        text = soul_path.read_text(encoding="utf-8")
    except Exception:
        return ""

    for match in _SOUL_FEISHU_MENTION_ROW_RE.finditer(text):
        role = (match.group(1) or "").strip()
        mapped_open_id = (match.group(2) or "").strip()
        display = (match.group(3) or "").strip()
        if mapped_open_id != oid:
            continue
        for candidate in (role, display):
            safe_candidate = safe_feishu_display_name(candidate, "")
            if safe_candidate and _is_likely_bot_display_name(safe_candidate):
                return safe_candidate
    return ""


def infer_bot_delegation_sender_name(
    sender_name: str,
    message: str,
    fallback: str = "另一个 bot",
) -> str:
    """Infer a readable bot name for a bot-to-bot delegation prompt."""
    safe_name = safe_feishu_display_name(sender_name, "")
    if safe_name and _is_likely_bot_display_name(safe_name):
        return safe_name

    mapped_name = _lookup_feishu_bot_name_from_soul(sender_name)
    if mapped_name:
        return mapped_name

    text = " ".join((message or "").split())
    for pattern in (_BOT_HEARTBEAT_RE, _BOT_INLINE_STATUS_RE, _BOT_SELF_INTRO_RE):
        match = pattern.search(text)
        if not match:
            continue
        candidate = safe_feishu_display_name(match.group(1), "")
        if candidate and _is_likely_bot_display_name(candidate):
            return candidate
    return fallback


def build_bot_delegation_approval_prompt(
    sender_name: str,
    message: str,
    *,
    include_text_fallback: bool = True,
) -> str:
    """Build the human approval prompt for a Feishu bot-to-bot delegation."""
    sender = infer_bot_delegation_sender_name(sender_name, message, "另一个 bot")
    preview = " ".join((message or "").split())
    if len(preview) > 500:
        preview = preview[:497] + "..."
    prompt = (
        f"{sender} 提议我介入处理这条 bot-to-bot 协作请求。\n\n"
        f"> {preview}\n\n"
        "需要人类确认后，我才会读取文件、查看日志、运行命令或调用其他工具开始排查。"
    )
    if include_text_fallback:
        prompt += "\n同意请回复 `/approve` 或“同意”；拒绝请回复 `/deny` 或“取消”。"
    return prompt


def build_approved_bot_delegation_prompt(approver_name: str) -> str:
    """Build the per-turn prompt after a human approved bot delegation."""
    approver = safe_feishu_display_name(approver_name, "a human in the group")
    return (
        "A human approved this Feishu bot-to-bot delegation request. "
        f"Approver: {approver}. You may now use the normal tools needed for the "
        "approved work. Keep the scope limited to the delegated request and report "
        "what you did."
    )


_SYSTEM_PROMPT = """You are a silent relevance gate for one team bot in a Feishu group.

Your job is NOT to answer the user. Decide whether THIS bot should speak.

Return JSON only:
{
  "decision": "relevant" | "irrelevant" | "uncertain",
  "confidence": 0.0-1.0,
  "reason": "short internal reason",
  "reply_intent": "if relevant, the angle this bot should take"
}

Guidelines:
- relevant: the message needs this bot's role, responsibility, judgment, or action.
- irrelevant: this bot has no distinct value to add, or another role is clearly better.
- uncertain: the message may be related but there is not enough signal.
- Be conservative. Do not speak just because you can.
- Ordinary chatter, weather, acknowledgements, and broad FYI messages are usually irrelevant.
- If the message asks for cross-functional coordination or risk handling, a leadership or owner role may be relevant.
- If several roles could each add distinct value, this bot may be relevant.
"""


def load_ambient_gate_config() -> AmbientGateConfig:
    """Load ambient gate settings from profile config and env vars."""
    data = _read_profile_config()
    raw = _extract_ambient_config(data)

    enabled_default = is_truthy_value(os.getenv("FEISHU_AMBIENT_LISTENING", ""), default=False)
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    if not isinstance(raw, Mapping):
        raw = {}

    return AmbientGateConfig(
        enabled=_coerce_bool(raw.get("enabled"), enabled_default),
        confidence_threshold=_coerce_float(
            raw.get("confidence_threshold", raw.get("threshold")),
            _DEFAULT_THRESHOLD,
        ),
        uncertain_policy=_normalize_uncertain_policy(raw.get("uncertain_policy", raw.get("uncertain"))),
        timeout_seconds=max(
            1.0,
            _coerce_float(raw.get("timeout_seconds", raw.get("timeout")), _DEFAULT_TIMEOUT_SECONDS),
        ),
        retry_on_timeout=_coerce_bool(raw.get("retry_on_timeout"), True),
        retry_timeout_seconds=max(
            1.0,
            _coerce_float(
                raw.get("retry_timeout_seconds", raw.get("retry_timeout")),
                _DEFAULT_RETRY_TIMEOUT_SECONDS,
            ),
        ),
        timeout_notice=_coerce_bool(raw.get("timeout_notice"), True),
        timeout_notice_min_interval_seconds=max(
            0.0,
            _coerce_float(
                raw.get("timeout_notice_min_interval_seconds", raw.get("timeout_notice_min_interval")),
                _DEFAULT_TIMEOUT_NOTICE_MIN_INTERVAL_SECONDS,
            ),
        ),
        max_identity_chars=max(
            500,
            _coerce_int(raw.get("max_identity_chars"), _DEFAULT_MAX_IDENTITY_CHARS),
        ),
        max_message_chars=max(
            200,
            _coerce_int(raw.get("max_message_chars"), _DEFAULT_MAX_MESSAGE_CHARS),
        ),
    )


def is_feishu_ambient_candidate(event: Any) -> bool:
    """Return true for human, group, non-command Feishu messages with no @bot."""
    if not bool(getattr(event, "_feishu_ambient_candidate", False)):
        return False
    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", None))
    if platform != "feishu":
        return False
    if getattr(source, "is_bot", False):
        return False
    chat_type = str(getattr(source, "chat_type", "") or "").lower()
    if chat_type not in {"group", "forum", "channel"}:
        return False
    try:
        if event.is_command():
            return False
    except Exception:
        pass
    return bool((getattr(event, "text", "") or "").strip())


async def evaluate_ambient_relevance(event: Any) -> AmbientDecision:
    """Evaluate whether a candidate message should reach the main agent."""
    config = load_ambient_gate_config()
    if not config.enabled:
        return AmbientDecision("relevant", 1.0, "ambient gate disabled")

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _evaluate_sync, event, config)


def _evaluate_sync(event: Any, config: AmbientGateConfig) -> AmbientDecision:
    try:
        return _evaluate_once_sync(event, config, config.timeout_seconds)
    except Exception as exc:
        if config.retry_on_timeout and _is_retryable_gate_error(exc):
            jitter = _retry_jitter_seconds(event)
            logger.warning(
                "ambient gate evaluation failed: %s; retrying once in %.2fs",
                exc,
                jitter,
            )
            if jitter > 0:
                time.sleep(jitter)
            try:
                return _evaluate_once_sync(event, config, config.retry_timeout_seconds)
            except Exception as retry_exc:
                logger.warning("ambient gate retry failed: %s", retry_exc)
                logger.debug("ambient gate retry traceback", exc_info=True)
                return _gate_error_decision(retry_exc, retried=True)
        logger.warning("ambient gate evaluation failed: %s", exc)
        logger.debug("ambient gate evaluation traceback", exc_info=True)
        return _gate_error_decision(exc, retried=False)


def _evaluate_once_sync(
    event: Any,
    config: AmbientGateConfig,
    timeout_seconds: float,
) -> AmbientDecision:
    text = (getattr(event, "text", "") or "").strip()
    if not text:
        return AmbientDecision("irrelevant", 1.0, "empty ambient message")

    identity = _load_identity(config.max_identity_chars)
    source = getattr(event, "source", None)
    user_payload = {
        "profile": _profile_name(),
        "chat_name": getattr(source, "chat_name", "") or "",
        "sender_name": getattr(source, "user_name", "") or "",
        "bot_identity": identity,
        "message": text[: config.max_message_chars],
        "reply_to_text": (getattr(event, "reply_to_text", "") or "")[:1000],
    }

    from agent.auxiliary_client import call_llm

    response = call_llm(
        task="ambient_gate",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Decide if THIS bot should respond to this Feishu group message.\n"
                    + json.dumps(user_payload, ensure_ascii=False)
                ),
            },
        ],
        temperature=0,
        max_tokens=180,
        timeout=timeout_seconds,
    )
    raw = (response.choices[0].message.content or "").strip()
    parsed = _parse_json_object(raw)
    decision = _normalize_decision(str(parsed.get("decision", "uncertain")))
    confidence = _coerce_float(parsed.get("confidence"), 0.0)
    reason = str(parsed.get("reason", "") or "").strip()
    reply_intent = str(parsed.get("reply_intent", "") or "").strip()
    if decision == "relevant" and confidence < config.confidence_threshold:
        decision = "uncertain"
        reason = (reason + " " if reason else "") + "below confidence threshold"
    if decision == "uncertain" and config.uncertain_policy == "respond":
        decision = "relevant"
    return AmbientDecision(
        decision=decision,
        confidence=max(0.0, min(1.0, confidence)),
        reason=reason[:300],
        reply_intent=reply_intent[:300],
        raw=raw[:1000],
    )


def _gate_error_decision(exc: Exception, *, retried: bool) -> AmbientDecision:
    suffix = " after retry" if retried else ""
    return AmbientDecision("uncertain", 0.0, f"gate error: {type(exc).__name__}{suffix}")


def _is_retryable_gate_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    lowered = str(exc).lower()
    return any(marker in lowered for marker in ("timeout", "timed out", "connection", "connect"))


def _retry_jitter_seconds(event: Any) -> float:
    text = getattr(event, "text", "") or ""
    seed = f"{_profile_name()}:{text[:200]}".encode("utf-8", errors="ignore")
    digest = hashlib.sha1(seed).hexdigest()
    return (int(digest[:4], 16) % 1000) / 1000.0


def _read_profile_config() -> dict:
    try:
        import yaml
    except Exception:
        return {}
    path = get_hermes_home() / "config.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_ambient_config(data: Mapping[str, Any]) -> Any:
    for container_key in ("feishu", "gateway"):
        container = data.get(container_key)
        if isinstance(container, Mapping):
            for key in ("ambient_listening", "ambient_gate", "ambient"):
                if key in container:
                    return container[key]
    for key in ("FEISHU_AMBIENT_LISTENING", "ambient_listening", "ambient_gate"):
        if key in data:
            return data[key]
    return {}


def _load_identity(max_chars: int) -> str:
    home = get_hermes_home()
    soul_path = home / "SOUL.md"
    if soul_path.exists():
        try:
            text = soul_path.read_text(encoding="utf-8").strip()
            if text:
                return text[:max_chars]
        except OSError:
            pass
    return f"Profile name: {_profile_name()}"


def _profile_name() -> str:
    try:
        return Path(get_hermes_home()).name
    except Exception:
        return ""


def _parse_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("ambient gate response was not a JSON object")
    return parsed


def _normalize_decision(value: str) -> str:
    normalized = (value or "").strip().lower()
    aliases = {
        "respond": "relevant",
        "reply": "relevant",
        "yes": "relevant",
        "skip": "irrelevant",
        "ignore": "irrelevant",
        "no": "irrelevant",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"relevant", "irrelevant", "uncertain"}:
        return "uncertain"
    return normalized


def _normalize_uncertain_policy(value: Any) -> str:
    normalized = str(value or "ignore").strip().lower()
    return "respond" if normalized in {"respond", "reply", "relevant"} else "ignore"


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return is_truthy_value(value, default=default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
