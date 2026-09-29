"""Gradio interface for the AI regulation RAG assistant."""

import html
import logging
import re

import gradio as gr

from generate_answer import generate_answer
from retrieval import load_data, retrieve


# ---------- Configuration ----------

LOGGER = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "I don't have enough information in the provided "
    "documents to answer this question."
)

EXAMPLES = [
    "What are the four core functions of the NIST AI RMF?",
    "How does Article 3 of the EU AI Act define risk?",
    "Which AI practices are prohibited under Article 5?",
    "What is the difference between a provider and a deployer?",
]

SOURCE_LABELS = {
    "eu_ai_act_2024.pdf": ("EU AI Act", "src-eu"),
    "nist_ai_rmf_1_0.pdf": ("NIST AI RMF", "src-nist"),
}


# ---------- Formatting helpers ----------

def cited_numbers(answer: str, total: int) -> list[int]:
    """Find valid references such as [2] or [1, 3]."""
    numbers = set()

    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for part in group.split(","):
            part = part.strip()

            if part.isdigit():
                number = int(part)

                if 1 <= number <= total:
                    numbers.add(number)

    return sorted(numbers)


def doc_info(source: str) -> tuple[str, str]:
    """Return an escaped label and a predefined CSS class."""
    label, css_class = SOURCE_LABELS.get(
        source, (source, "src-other")
    )
    return html.escape(label), css_class


def source_card(number: int, chunk: dict, cited: bool) -> str:
    """Render one passage, escaping document content before display."""
    label, css_class = doc_info(chunk["source"])
    filename = html.escape(chunk["source"])
    text = html.escape(chunk["text"])
    status = "Cited in answer" if cited else "Retrieved, not cited"
    page = int(chunk["page"])
    score = float(chunk["score"])

    return (
        f'<div class="source-card {css_class}">'
        '<div class="source-heading">'
        f'<strong>[{number}]</strong>'
        f'<span class="source-badge">{label}</span>'
        f'<span>PDF page {page}</span>'
        f'<span class="score">Similarity: {score:.3f}</span>'
        '</div>'
        f'<div class="source-status">{status} · {filename}</div>'
        f'<div class="source-text">{text}</div>'
        '</div>'
    )


def format_sources(chunks: list[dict], numbers: list[int]) -> str:
    """Show cited passages and all search results for the latest question."""
    all_chunks, _ = load_data()
    cited = set(numbers)

    cited_cards = "".join(
        source_card(number, chunks[number - 1], cited=True)
        for number in numbers
    )

    all_cards = "".join(
        source_card(number, chunk, cited=number in cited)
        for number, chunk in enumerate(chunks, start=1)
    )

    if not cited_cards:
        cited_cards = (
            '<p>No numbered citations were detected in this answer. '
            'Review the retrieved passages below.</p>'
        )

    return (
        '<section class="source-panel">'
        '<h3>Sources for the latest question</h3>'
        '<div class="summary-row">'
        f'<span>{len(all_chunks):,} indexed passages</span>'
        f'<span>{len(chunks)} retrieved</span>'
        f'<span>{len(cited)} cited in answer</span>'
        '</div>'
        '<p class="source-note">'
        'A citation identifies a passage referenced by the model; '
        'check that the passage supports the claim. '
        'Page numbers are PDF positions, starting at 1.'
        '</p>'
        '<details open><summary>Passages cited in the answer</summary>'
        f'{cited_cards}</details>'
        '<details><summary>All retrieved passages</summary>'
        f'{all_cards}</details>'
        '</section>'
    )


# ---------- Chat logic ----------

def respond(message: str, history: list):
    """Retrieve passages, generate an answer, and update the interface."""
    message = (message or "").strip()

    if not message:
        yield "", history or [], gr.skip()
        return

    # Copy messages so this callback does not modify the input history.
    history = [dict(item) for item in (history or [])]

    # Escape user content before passing it to the Markdown renderer.
    history.extend([
        {"role": "user", "content": html.escape(message)},
        {
            "role": "assistant",
            "content": "🔎 Searching the documents…",
        },
    ])

    yield "", history, gr.update(value="", visible=False)

    try:
        chunks = retrieve(message)

        history[-1]["content"] = "✍️ Writing the answer…"
        yield "", history, gr.update(value="", visible=False)

        if chunks:
            answer = generate_answer(message, chunks)
        else:
            answer = FALLBACK_MESSAGE

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer generation returned no text.")

        answer = answer.strip()

        # Keep the exact refusal visible instead of replacing its meaning.
        history[-1]["content"] = html.escape(answer)

        if chunks:
            numbers = cited_numbers(answer, len(chunks))
            sources = format_sources(chunks, numbers)
            yield "", history, gr.update(value=sources, visible=True)
        else:
            yield "", history, gr.update(value="", visible=False)

    except Exception:
        # Details stay in the terminal rather than being exposed in the UI.
        LOGGER.exception("Unable to answer the question")

        history[-1]["content"] = (
            "⚠️ I couldn’t complete this request. "
            "Please check the terminal for the error details."
        )

        yield "", history, gr.update(value="", visible=False)


def use_example(example: str, history: list):
    """Run a question selected from the example buttons."""
    yield from respond(example, history)


# ---------- Styling ----------

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="teal",
    neutral_hue="slate",
)

CSS = """
.gradio-container {
    max-width: 1050px !important;
    margin: 0 auto !important;
}

#hero {
    background: linear-gradient(120deg, #312e81, #4f46e5, #0d9488);
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 12px;
}

#hero h1 {
    color: white;
    margin: 0 0 8px;
    font-size: 1.9rem;
}

#hero p {
    color: #e0e7ff;
    margin: 0;
}

#hero .pill {
    display: inline-block;
    margin: 12px 8px 0 0;
    padding: 4px 12px;
    color: white;
    background: rgba(255, 255, 255, .16);
    border: 1px solid rgba(255, 255, 255, .3);
    border-radius: 999px;
    font-size: .8rem;
}

#chat {
    border-radius: 16px;
}

#input-row {
    padding: 8px;
    border: 1px solid #c7d2fe;
    border-radius: 14px;
}

#examples button {
    border-radius: 999px;
}

.source-panel {
    background: #f8fafc;
    color: #0f172a;
    border: 1px solid #c7d2fe;
    border-radius: 16px;
    padding: 16px;
}

.source-panel h3 {
    color: #312e81;
    margin-top: 0;
}

.summary-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.summary-row span {
    background: #e0e7ff;
    color: #3730a3;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: .85rem;
}

.source-note {
    color: #475569;
    font-size: .82rem;
    margin: 12px 0;
}

.source-panel summary {
    cursor: pointer;
    font-weight: 600;
    padding: 8px 0;
}

.source-card {
    background: white;
    border-left: 4px solid #9333ea;
    border-radius: 10px;
    padding: 12px;
    margin: 10px 0;
}

.source-card.src-eu {
    border-left-color: #2563eb;
}

.source-card.src-nist {
    border-left-color: #0d9488;
}

.source-heading {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    font-size: .85rem;
}

.source-badge {
    background: #4338ca;
    color: white;
    border-radius: 999px;
    padding: 2px 8px;
}

.src-eu .source-badge {
    background: #2563eb;
}

.src-nist .source-badge {
    background: #0d9488;
}

.score {
    color: #64748b;
    margin-left: auto;
}

.source-status {
    margin-top: 6px;
    color: #64748b;
    font-size: .78rem;
    overflow-wrap: anywhere;
}

.source-text {
    margin-top: 8px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.6;
    font-size: .88rem;
}
"""


# ---------- Layout ----------

with gr.Blocks(title="AI Regulation Assistant") as demo:
    gr.HTML(
        """
        <div id="hero">
            <h1>AI Regulation Assistant</h1>
            <p>Ask questions about the EU AI Act and NIST AI RMF,
            then inspect the passages behind the answer.</p>
            <span class="pill">EU AI Act 2024/1689</span>
            <span class="pill">NIST AI RMF 1.0</span>
        </div>
        """
    )

    chatbot = gr.Chatbot(
        elem_id="chat",
        height=480,
        show_label=False,
        sanitize_html=True,
        placeholder="Ask your first question below.",
    )

    with gr.Row(elem_id="input-row"):
        question = gr.Textbox(
            show_label=False,
            placeholder="Type your question and press Enter…",
            lines=1,
            max_lines=4,
            scale=8,
            autofocus=True,
        )
        send = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear", scale=1)

    gr.Markdown(
        "Each question is answered independently. "
        "Include the context needed for follow-up questions."
    )

    gr.Markdown("**Try an example:**")

    with gr.Row(elem_id="examples"):
        example_buttons = [
            gr.Button(text, size="sm") for text in EXAMPLES
        ]

    sources_panel = gr.HTML(value="", visible=False)

    # Share one queue group to avoid overlapping answer-generation callbacks.
    event_options = {
        "concurrency_id": "rag_requests",
        "concurrency_limit": 1,
    }

    ask_events = [
        question.submit(
            respond,
            inputs=[question, chatbot],
            outputs=[question, chatbot, sources_panel],
            **event_options,
        ),
        send.click(
            respond,
            inputs=[question, chatbot],
            outputs=[question, chatbot, sources_panel],
            **event_options,
        ),
    ]

    for button in example_buttons:
        ask_events.append(
            button.click(
                use_example,
                inputs=[button, chatbot],
                outputs=[question, chatbot, sources_panel],
                **event_options,
            )
        )

    clear.click(
        fn=lambda: ("", [], gr.update(value="", visible=False)),
        inputs=None,
        outputs=[question, chatbot, sources_panel],
        queue=False,
        cancels=ask_events,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo.queue().launch(theme=THEME, css=CSS)