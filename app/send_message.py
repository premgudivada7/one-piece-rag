"""send_message tool with a human approval step (Yes/No).

Posting to Slack or Teams cannot be undone, so the agent can only
PROPOSE a message. Nothing is sent until a person approves it:

    proposal = propose_message("text")   # agent tool: never sends
    approve(proposal["proposal_id"])      # person said YES -> send
    reject(proposal["proposal_id"])       # person said NO  -> cancel

Delivery backends (settings go in .env, never in code):

1. Slack MCP server (primary path, how the assignment asks for it)
       SLACK_MCP_TOKEN=xoxp-...        user token from your Slack app's OAuth
       SLACK_CHANNEL_ID=C0123456789    channel to post in
       SLACK_MCP_URL=https://mcp.slack.com/mcp   (optional, this is the default)
       SLACK_MCP_SEND_TOOL=...         (optional, found automatically)
   Requires: uv pip install mcp

2. Webhook fallback (accepted if the MCP app/OAuth is not ready)
       SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
   or  TEAMS_WEBHOOK_URL=https://...   (Teams "Workflows" webhook)

MESSAGE_BACKEND=slack_mcp | slack_webhook | teams_webhook picks one
explicitly; otherwise the first configured backend in the order above
is used.

Terminal:
    python app/send_message.py --check        show which backend is configured
    python app/send_message.py "Hello team"   preview, then ask yes/no
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

import requests
from dotenv import load_dotenv


# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "data" / "message_log.json"

DEFAULT_MCP_URL = "https://mcp.slack.com/mcp"
MAX_MESSAGE_CHARS = 3000
TIMEOUT_SECONDS = 20

BACKENDS = ("slack_mcp", "slack_webhook", "teams_webhook")

# Proposals waiting for a Yes/No decision, by proposal id.
PENDING: dict[str, dict] = {}
LOCK = Lock()


def setting(name: str) -> str:
    """Read one setting from .env / the environment."""
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv(name, "").strip()


def choose_backend() -> str | None:
    """Return the configured delivery backend, or None."""
    chosen = setting("MESSAGE_BACKEND")
    if chosen:
        return chosen if chosen in BACKENDS else None

    if setting("SLACK_MCP_TOKEN") and setting("SLACK_CHANNEL_ID"):
        return "slack_mcp"
    if setting("SLACK_WEBHOOK_URL"):
        return "slack_webhook"
    if setting("TEAMS_WEBHOOK_URL"):
        return "teams_webhook"
    return None


# ---------- Log ----------

def log_event(event: dict) -> None:
    """Append one proposal or decision to data/message_log.json."""
    with LOCK:
        entries = []
        if LOG_FILE.exists():
            try:
                entries = json.loads(LOG_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                entries = []

        entries.append({"time": datetime.now().isoformat(timespec="seconds"),
                        **event})

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = LOG_FILE.with_suffix(".tmp")
        temporary_file.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_file.replace(LOG_FILE)


# ---------- Step 1: the agent proposes (never sends) ----------

def propose_message(text: str) -> dict:
    """Create a message proposal that waits for human approval."""
    if not isinstance(text, str) or not text.strip():
        return {"ok": False, "error": "The message text cannot be empty."}

    text = text.strip()

    if len(text) > MAX_MESSAGE_CHARS:
        return {"ok": False,
                "error": f"The message is longer than {MAX_MESSAGE_CHARS} characters."}

    backend = choose_backend()
    if backend is None:
        return {"ok": False,
                "error": "No Slack/Teams connection is configured in .env."}

    proposal = {
        "proposal_id": uuid.uuid4().hex[:8],
        "text": text,
        "backend": backend,
        "status": "pending_approval",
    }

    with LOCK:
        PENDING[proposal["proposal_id"]] = proposal

    log_event({"event": "proposed", **proposal})

    return {
        "ok": True,
        **proposal,
        "note": ("Not sent yet. A person must approve this message "
                 "before it is posted."),
    }


# ---------- Step 2: the person decides ----------

def approve(proposal_id: str) -> dict:
    """The person said YES: send the proposed message once."""
    with LOCK:
        proposal = PENDING.pop(proposal_id, None)

    if proposal is None:
        return {"ok": False,
                "error": "Unknown or already decided proposal."}

    try:
        details = deliver(proposal["text"], proposal["backend"])
        result = {"ok": True, "status": "sent", **details}
    except Exception as error:  # Network, auth or server errors.
        result = {"ok": False, "status": "failed",
                  "error": readable_error(error)}

    log_event({"event": result["status"],
               "proposal_id": proposal_id,
               "backend": proposal["backend"],
               **({"error": result["error"]} if not result["ok"] else {})})

    return {"proposal_id": proposal_id, "backend": proposal["backend"], **result}


def reject(proposal_id: str) -> dict:
    """The person said NO: cancel the proposal without sending."""
    with LOCK:
        proposal = PENDING.pop(proposal_id, None)

    if proposal is None:
        return {"ok": False,
                "error": "Unknown or already decided proposal."}

    log_event({"event": "rejected", "proposal_id": proposal_id})
    return {"ok": True, "proposal_id": proposal_id, "status": "rejected"}


def readable_error(error: BaseException) -> str:
    """Unwrap MCP task-group errors so the real reason is shown."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return f"{type(error).__name__}: {error}"


# ---------- Delivery ----------

def deliver(text: str, backend: str) -> dict:
    """Send the text with the chosen backend. Raises on failure."""
    if backend == "slack_mcp":
        return asyncio.run(send_with_slack_mcp(text))
    if backend == "slack_webhook":
        return send_with_webhook(setting("SLACK_WEBHOOK_URL"), {"text": text})
    if backend == "teams_webhook":
        return send_with_webhook(setting("TEAMS_WEBHOOK_URL"), teams_card(text))
    raise ValueError(f"Unknown backend: {backend}")


def send_with_webhook(url: str, payload: dict) -> dict:
    """POST a JSON payload to a Slack or Teams incoming webhook."""
    if not url.startswith("https://"):
        raise ValueError("The webhook URL must start with https://")

    response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)

    if response.status_code >= 300:
        raise RuntimeError(
            f"Webhook returned HTTP {response.status_code}: {response.text[:200]}"
        )

    return {"http_status": response.status_code}


def teams_card(text: str) -> dict:
    """Wrap text in the Adaptive Card format Teams Workflows expect."""
    return {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "type": "AdaptiveCard",
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "version": "1.4",
                "body": [{"type": "TextBlock", "text": text, "wrap": True}],
            },
        }],
    }


async def send_with_slack_mcp(text: str) -> dict:
    """Send through Slack's official MCP server as an MCP client."""
    try:
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as error:
        raise RuntimeError("Install the MCP client first: uv pip install mcp") from error

    token = setting("SLACK_MCP_TOKEN")
    channel = setting("SLACK_CHANNEL_ID")
    url = setting("SLACK_MCP_URL") or DEFAULT_MCP_URL

    if not token or not channel:
        raise ValueError("Set SLACK_MCP_TOKEN and SLACK_CHANNEL_ID in .env.")

    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(headers=headers, timeout=TIMEOUT_SECONDS) as http:
        async with streamable_http_client(url, http_client=http) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                listed = await session.list_tools()
                tool = find_send_tool(listed.tools)
                arguments = build_arguments(tool, channel, text)

                result = await session.call_tool(tool.name, arguments)

    output = " ".join(
        item.text for item in result.content if getattr(item, "text", None)
    )

    if result.isError:
        raise RuntimeError(f"Slack MCP error: {output[:300]}")

    return {"mcp_tool": tool.name, "server_reply": output[:300]}


def find_send_tool(tools: list):
    """Pick the server's send-message tool (name set in .env or found)."""
    wanted = setting("SLACK_MCP_SEND_TOOL")
    names = [tool.name for tool in tools]

    for tool in tools:
        if wanted and tool.name == wanted:
            return tool
        if not wanted and "send" in tool.name and "message" in tool.name \
                and "schedule" not in tool.name and "draft" not in tool.name:
            return tool

    raise RuntimeError(f"No send-message tool found. Server tools: {names}")


def build_arguments(tool, channel: str, text: str) -> dict:
    """Map channel and text onto the tool's own parameter names."""
    properties = (tool.inputSchema or {}).get("properties", {})

    channel_key = next((k for k in ("channel_id", "channel", "conversation_id")
                        if k in properties), "channel_id")
    text_key = next((k for k in ("message", "text", "content")
                     if k in properties), "text")

    return {channel_key: channel, text_key: text}


# ---------- Tool schema for the agent (registered in Phase 4) ----------

SEND_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": (
            "Propose posting a final answer to the team's Slack/Teams channel. "
            "This does NOT send: a person must approve the message first. "
            "Only use it when the user explicitly asks to share or send "
            "the answer. Include the answer and its citations in the text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The exact message to post, with citations.",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}


# ---------- Terminal: the Yes/No approval flow ----------

def ask_yes_no(prompt: str) -> bool:
    """Only an explicit 'yes' or 'y' counts as approval."""
    answer = input(prompt).strip().lower()
    return answer in ("yes", "y")


def main() -> None:
    """Preview a message, ask for approval, then send or cancel."""
    if len(sys.argv) < 2:
        print(__doc__)
        return

    if sys.argv[1] == "--check":
        backend = choose_backend()
        print(f"Configured backend: {backend or 'none (add settings to .env)'}")
        return

    proposal = propose_message(" ".join(sys.argv[1:]))
    if not proposal["ok"]:
        print(f"Cannot propose: {proposal['error']}")
        return

    print("\n----- Message to send -----")
    print(proposal["text"])
    print("---------------------------")
    print(f"Destination: {proposal['backend']}")

    if ask_yes_no("Send this message? (yes/no): "):
        print(json.dumps(approve(proposal["proposal_id"]), indent=2))
    else:
        print(json.dumps(reject(proposal["proposal_id"]), indent=2))


if __name__ == "__main__":
    main()
