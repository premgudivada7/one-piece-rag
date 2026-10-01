import json
import sys
import time

import requests

from app import config

_sent: set[tuple[str, str]] = set()  # (channel, text) already posted by this process


def backend() -> str:
    """How send_message posts: a Slack/Teams webhook, or a dry run."""
    if config.SEND_BACKEND != "auto":
        return config.SEND_BACKEND
    return "webhook" if config.WEBHOOK_URL else "dry_run"


def backend_label() -> str:
    if backend() == "webhook":
        return f"{config.WEBHOOK_FORMAT} webhook"
    return "dry run (nothing is posted; see data/sent_messages.jsonl)"


def validate(text: str, channel: str | None) -> tuple[str, str, str | None]:
    """Guardrails in code, checked before a person is even asked: allowed channel, not empty,
    not too long, not already sent. Returns (channel, text, error or None)."""
    channel = (channel or "").strip() or config.DEFAULT_CHANNEL
    text = (text or "").strip()
    if config.ALLOWED_CHANNELS and channel and channel not in config.ALLOWED_CHANNELS:
        return channel, text, (f"Error: channel {channel!r} is not allowed. "
                               f"Allowed: {', '.join(sorted(config.ALLOWED_CHANNELS))}.")
    if not text:
        return channel, text, "Error: refusing to send an empty message."
    if len(text) > config.MESSAGE_MAX_CHARS:
        return channel, text, (f"Error: message is {len(text)} characters, the limit is "
                               f"{config.MESSAGE_MAX_CHARS}. Shorten it.")
    if (channel, text) in _sent:
        return channel, text, "Error: this exact message was already sent to this channel. Not sending it twice."
    return channel, text, None


def _send_webhook(text: str) -> tuple[str, int, str]:
    """Post through the webhook and check that the service confirmed it. Returns (detail, HTTP status, reply)."""
    if config.WEBHOOK_FORMAT == "teams":
        payload = {"type": "message", "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {"$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard", "version": "1.4",
                        "body": [{"type": "TextBlock", "text": text, "wrap": True}]}}]}
    else:
        payload = {"text": text}
    response = requests.post(config.WEBHOOK_URL, json=payload, timeout=config.SEND_TIMEOUT)
    reply = response.text.strip()[:200]
    if response.status_code >= 300:
        raise RuntimeError(f"webhook returned HTTP {response.status_code}: {reply}")
    # A Slack webhook answers exactly "ok" once the message is in the channel; Teams answers 200 or 202.
    if config.WEBHOOK_FORMAT == "slack" and reply != "ok":
        raise RuntimeError(f"Slack did not confirm delivery (HTTP {response.status_code}, reply {reply!r})")
    confirmation = ("Slack confirmed the message is in the channel (HTTP 200 'ok')"
                    if config.WEBHOOK_FORMAT == "slack" else f"Teams accepted the message (HTTP {response.status_code})")
    return f"Sent via {config.WEBHOOK_FORMAT} webhook. {confirmation}.", response.status_code, reply


def send_message(text: str, channel: str | None = None) -> str:
    """Post the message. Only called after a person approved it. Returns text for the model."""
    channel, text, error = validate(text, channel)
    if error:
        return error
    name = backend()
    status, reply = None, None
    try:
        if name == "webhook":
            detail, status, reply = _send_webhook(text)
        else:
            detail = ("DRY RUN: nothing was actually posted (no webhook configured). "
                      "Tell the user the message was not sent to the channel.")
    except Exception as e:
        return f"Error: sending failed ({type(e).__name__}: {e}). The message was NOT sent."
    _sent.add((channel, text))
    append_jsonl(config.SENT_LOG_PATH, {"backend": name, "channel": channel, "http_status": status,
                                        "reply": reply, "text": text})
    return detail


def append_jsonl(path, record: dict):
    """Append one timestamped JSON line (sent messages, fake sends, feedback)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}, ensure_ascii=False) + "\n")


def fake_send(proposal: dict) -> bool:
    """Stands in for the person during tests and evals: logs the proposal and always says no,
    so nothing reaches the real channel."""
    append_jsonl(config.FAKE_SEND_LOG_PATH, {"declined": True, **proposal})
    return False


if __name__ == "__main__":
    if "--send" in sys.argv:
        message = sys.argv[sys.argv.index("--send") + 1]
        print(f"Backend: {backend_label()}\nChannel: {config.DEFAULT_CHANNEL or '(set by the webhook)'}\n\n{message}\n")
        print(send_message(message) if input("Send it? [y/N] ").strip().lower() == "y" else "Cancelled.")
    else:
        print('Usage: python -m app.messaging --send "hello"   (asks y/N, then posts through the webhook)')
