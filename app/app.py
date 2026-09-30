import html
import re

import gradio as gr

from retrieval import load_data, retrieve
from generate_answer import generate_answer
from tools import send_message


# ============================================================
# Configuration
# ============================================================

EXAMPLES = [
    "What is the EU AI Act?",
    "What are the risk categories under the EU AI Act?",
    "What is the NIST AI Risk Management Framework?",
    "What are the requirements for high-risk AI systems?",
]

FALLBACK_MESSAGE = (
    "I don't have enough information in the provided "
    "documents to answer this question."
)

pending_message = ""
pending_platform = "slack"
pending_answer = ""


# ============================================================
# Helper functions
# ============================================================

def highlight_citations(text):
    """Highlight page citations such as [p. 12] or [Page 12]."""
    if not text:
        return text

    pattern = r"(\[(?:p\.?|page)\s*\d+(?:-\d+)?\])"

    return re.sub(
        pattern,
        r"<mark>\1</mark>",
        html.escape(text),
        flags=re.IGNORECASE,
    )


def format_origin(chunks, numbers=None):
    """Create a simple source section from retrieved chunks."""
    if not chunks:
        return ""

    lines = ["### Sources"]

    for index, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            source = (
                chunk.get("source")
                or chunk.get("file")
                or chunk.get("filename")
                or "Document"
            )

            page = (
                chunk.get("page")
                or chunk.get("page_number")
                or chunk.get("page_num")
            )

            if page is not None:
                lines.append(f"- {source} — page {page}")
            else:
                lines.append(f"- {source}")

        else:
            lines.append(f"- Source {index}")

    return "\n".join(lines)


# ============================================================
# Slack approval functions
# ============================================================

def approve_send():
    global pending_message
    global pending_platform
    global pending_answer

    if not pending_message:
        return "No answer is waiting to be sent."

    result = send_message(pending_message)

    if isinstance(result, dict):
        if result.get("success"):
            message = "Answer sent successfully to Slack."
        else:
            message = result.get(
                "error",
                "Failed to send the answer to Slack."
            )
    else:
        message = str(result)

    pending_message = ""
    pending_platform = "slack"
    pending_answer = ""

    return message


def reject_send():
    global pending_message
    global pending_platform
    global pending_answer

    pending_message = ""
    pending_platform = "slack"
    pending_answer = ""

    return "Send cancelled."


# ============================================================
# Main response function
# ============================================================

def respond(message: str, history: list):
    global pending_message
    global pending_platform
    global pending_answer

    if not message or not message.strip():
        return "", history

    message = message.strip()

    if history is None:
        history = []

    try:
        # Add user message
        history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        # Temporary assistant message
        history.append(
            {
                "role": "assistant",
                "content": "✍️ Writing the answer...",
            }
        )

        yield "", history

        # Retrieve relevant chunks
        chunks = retrieve(message)

        if not chunks:
            answer = FALLBACK_MESSAGE
        else:
            answer = generate_answer(message, chunks)

        if not answer:
            answer = FALLBACK_MESSAGE

        # Prepare Slack message
        pending_message = answer
        pending_answer = answer
        pending_platform = "slack"

        # Add source information
        source_text = format_origin(chunks)

        if source_text:
            content = (
                highlight_citations(answer)
                + "\n\n"
                + source_text
            )
        else:
            content = highlight_citations(answer)

        history[-1]["content"] = content

        yield "", history

    except Exception as error:
        history[-1]["content"] = (
            "<div class='error-msg'>"
            "⚠️ Something went wrong: "
            f"{html.escape(str(error))}"
            "</div>"
        )

        yield "", history


# ============================================================
# Example question function
# ============================================================

def use_example(example: str, history: list):
    """
    Clicking an example question sends it directly.
    """
    yield from respond(example, history)


# ============================================================
# Styling
# ============================================================

CSS = """
#hero {
    text-align: center;
    padding: 20px;
    margin-bottom: 20px;
}

#hero h1 {
    font-size: 2.2rem;
    margin-bottom: 10px;
}

#hero p {
    color: #475569;
    font-size: 1.05rem;
    line-height: 1.6;
}

.pill {
    display: inline-block;
    padding: 6px 12px;
    margin: 5px;
    border-radius: 999px;
    background: #eef2ff;
    color: #334155;
    font-size: 0.9rem;
}

#examples button:hover {
    background: #eef2ff !important;
}

.error-msg {
    padding: 10px;
    border-radius: 8px;
    background: #fee2e2;
    color: #991b1b;
}

mark {
    padding: 2px 5px;
    border-radius: 4px;
}
"""


THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="teal",
    neutral_hue="slate",
)


# ============================================================
# Layout
# ============================================================

with gr.Blocks(
    title="AI Regulation Assistant",
    theme=THEME,
    css=CSS,
) as demo:

    gr.HTML(
        """
        <div id="hero">
            <h1>🤖 AI Regulation Assistant</h1>

            <p>
                Ask anything about the
                <b>EU AI Act</b> and the
                <b>NIST AI Risk Management Framework</b>.
                Every answer cites the exact page it came from.
            </p>

            <span class="pill">🇪🇺 EU AI Act 2024/1689</span>

            <span class="pill">🇺🇸 NIST AI RMF 1.0</span>
        </div>
        """
    )

    chatbot = gr.Chatbot(
        elem_id="chat",
        height=520,
        show_label=False,
        avatar_images=(None, None),
        placeholder=(
            "<div style='text-align:center;color:#64748b'>"
            "<div style='font-size:2.4rem'>💬</div>"
            "<b>Ask your first question below</b><br>"
            "Type and press <b>Enter</b> to send."
            "</div>"
        ),
    )

    with gr.Row(elem_id="input-row", equal_height=True):

        question = gr.Textbox(
            show_label=False,
            placeholder="Type your question and press Enter...",
            lines=1,
            max_lines=4,
            scale=8,
            container=False,
            autofocus=True,
        )

        send = gr.Button(
            "Send ➤",
            variant="primary",
            scale=1,
            elem_id="send-btn",
        )

        clear = gr.Button(
            "🗑️ Clear",
            variant="secondary",
            scale=1,
        )

    # ========================================================
    # Slack approval section
    # ========================================================

    gr.Markdown("### Send answer to Slack")

    with gr.Row():

        approve_button = gr.Button(
            "Yes, Send",
            variant="primary",
        )

        reject_button = gr.Button(
            "No, Cancel",
            variant="secondary",
        )

    approval_status = gr.Textbox(
        label="Message status",
        interactive=False,
    )

    # ========================================================
    # Example questions
    # ========================================================

    gr.Markdown("**Try one of these:**")

    with gr.Row(elem_id="examples"):

        example_buttons = [
            gr.Button(text, size="sm")
            for text in EXAMPLES
        ]

    # ========================================================
    # Events
    # ========================================================

    ask_events = [
        question.submit(
            respond,
            [question, chatbot],
            [question, chatbot],
        ),

        send.click(
            respond,
            [question, chatbot],
            [question, chatbot],
        ),
    ]

    for button in example_buttons:

        ask_events.append(
            button.click(
                use_example,
                [button, chatbot],
                [question, chatbot],
            )
        )

    # Clear button
    clear.click(
        lambda: ("", []),
        None,
        [question, chatbot],
        queue=False,
        cancels=ask_events,
    )

    # Slack approval
    approve_button.click(
        approve_send,
        None,
        approval_status,
        queue=False,
    )

    reject_button.click(
        reject_send,
        None,
        approval_status,
        queue=False,
    )


# ============================================================
# Force light theme
# ============================================================

FORCE_LIGHT = """
() => {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""


# ============================================================
# Run application
# ============================================================

if __name__ == "__main__":
    demo.launch(
        js=FORCE_LIGHT,
    )