import html
import re

import gradio as gr

from retrieval import load_data, retrieve
from generate_answer import generate_answer


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
    """Return citation numbers like [2] or [1, 3] used in the answer."""
    numbers = []

    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                n = int(part)
                if 1 <= n <= total and n not in numbers:
                    numbers.append(n)

    return sorted(numbers)


def doc_info(source: str) -> tuple[str, str]:
    """Return a friendly document name and its colour class."""
    return SOURCE_LABELS.get(source, (source, "src-other"))


def total_passages() -> int:
    """How many passages the search index holds."""
    try:
        chunks, _ = load_data()
        return len(chunks)
    except Exception:
        return 0


def format_origin(chunks: list[dict], numbers: list[int]) -> str:
    """Show which documents and pages the answer was built from."""
    cited = set(numbers)
    total = total_passages()

    # ---- Step-by-step trace ----
    trace = (
        '<div class="trace">'
        f'<span class="step">🔎 Searched <b>{total:,}</b> passages</span>'
        '<span class="arrow">→</span>'
        f'<span class="step">🎯 <b>{len(chunks)}</b> closest matches</span>'
        '<span class="arrow">→</span>'
        f'<span class="step used">✅ <b>{len(cited)}</b> used in this answer</span>'
        "</div>"
    )

    # ---- Documents and pages, grouped ----
    by_doc: dict[str, dict] = {}
    for i, chunk in enumerate(chunks, start=1):
        doc = by_doc.setdefault(chunk["source"], {"cited": [], "other": []})
        bucket = "cited" if i in cited else "other"
        if chunk["page"] not in doc[bucket]:
            doc[bucket].append(chunk["page"])

    rows = []
    # Documents that were actually cited come first.
    for source, pages in sorted(
        by_doc.items(), key=lambda item: -len(item[1]["cited"])
    ):
        label, css_class = doc_info(source)
        other = [p for p in pages["other"] if p not in pages["cited"]]
        cited_chips = "".join(
            f'<span class="page-chip cited">p. {p}</span>'
            for p in sorted(pages["cited"])
        )
        other_chips = "".join(
            f'<span class="page-chip">p. {p}</span>' for p in sorted(other)
        )
        status = (
            '<span class="doc-status yes">used</span>'
            if pages["cited"]
            else '<span class="doc-status no">searched, not used</span>'
        )
        rows.append(
            f'<div class="doc-row {css_class}">'
            f'<div class="doc-name"><span class="src-badge">{label}</span>{status}</div>'
            f'<div class="doc-pages">{cited_chips}{other_chips}</div>'
            "</div>"
        )

    origin = (
        '<div class="origin"><div class="origin-title">'
        "🧭 Where this answer comes from</div>"
        f"{trace}{''.join(rows)}"
        '<div class="legend"><span class="page-chip cited">p. N</span> cited in the answer '
        '&nbsp; <span class="page-chip">p. N</span> retrieved but not cited</div>'
        "</div>"
    )

    # ---- Full text of every cited reference ----
    cards = []
    for n in numbers:
        chunk = chunks[n - 1]
        label, css_class = doc_info(chunk["source"])
        text = html.escape(chunk["text"].replace("\n", " "))
        cards.append(
            f'<div class="src-card {css_class}">'
            f'<div class="src-head"><span class="src-num">[{n}]</span>'
            f'<span class="src-badge">{label}</span>'
            f'<span class="src-page">Page {chunk["page"]}</span>'
            f'<span class="src-score">match {chunk["score"]:.2f}</span></div>'
            f'<div class="src-text">“{text}”</div>'
            "</div>"
        )

    references = (
        '<details class="src-box" open><summary>📌 References used '
        f"({len(numbers)})</summary>{''.join(cards)}</details>"
        if cards
        else ""
    )

    # ---- Everything the search returned ----
    all_rows = []
    for i, chunk in enumerate(chunks, start=1):
        label, css_class = doc_info(chunk["source"])
        mark = "✅" if i in cited else "▫️"
        snippet = html.escape(chunk["text"][:150].replace("\n", " "))
        all_rows.append(
            f'<div class="all-row {css_class}{" is-cited" if i in cited else ""}">'
            f'<span class="all-mark">{mark}</span>'
            f'<span class="src-num">[{i}]</span>'
            f'<span class="src-badge">{label}</span>'
            f'<span class="src-page">p. {chunk["page"]}</span>'
            f'<span class="src-score">{chunk["score"]:.2f}</span>'
            f'<div class="all-snippet">{snippet}…</div>'
            "</div>"
        )

    retrieved = (
        '<details class="src-box"><summary>🗂️ All retrieved passages '
        f"({len(chunks)}) — click to expand</summary>{''.join(all_rows)}</details>"
    )

    return origin + references + retrieved


def highlight_citations(answer: str) -> str:
    """Turn [1] into small coloured citation pills."""
    return re.sub(
        r"\[([\d,\s]+)\]",
        lambda m: f'<span class="cite">{m.group(1).strip()}</span>',
        answer,
    )


# ---------- Chat logic ----------

def respond(message: str, history: list):
    """Add the question to the chat, then stream in the answer."""
    message = (message or "").strip()

    if not message:
        yield "", history
        return

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "🔍 *Searching the documents…*"},
    ]
    yield "", history

    try:
        chunks = retrieve(message)

        history[-1]["content"] = "✍️ *Writing the answer…*"
        yield "", history

        answer = generate_answer(message, chunks)

    except Exception as error:  # Show errors in the chat, not a crash.
        history[-1]["content"] = (
            f'<div class="error-msg">⚠️ Something went wrong: '
            f"{html.escape(str(error))}</div>"
        )
        yield "", history
        return

    if answer.strip().strip('"') == FALLBACK_MESSAGE:
        content = (
            '<div class="no-info">🤷 I couldn’t find this in the '
            "EU AI Act or the NIST AI RMF. Try rephrasing, or ask "
            "about something these documents cover.</div>"
        )
    else:
        numbers = cited_numbers(answer, len(chunks))
        content = (
            highlight_citations(answer)
            + "\n\n" + format_origin(chunks, numbers)
        )

    history[-1]["content"] = content
    yield "", history


def use_example(example: str, history: list):
    """Clicking an example chip asks it straight away."""
    yield from respond(example, history)


# ---------- Styling ----------

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
body, .gradio-container {
    background: linear-gradient(160deg, #eef2ff 0%, #f0fdfa 55%, #fdf4ff 100%) !important;
}

#hero {
    background: linear-gradient(120deg, #312e81 0%, #4f46e5 50%, #0d9488 100%);
    border-radius: 20px; padding: 26px 30px; color: white;
    box-shadow: 0 12px 30px rgba(79, 70, 229, .25);
}
#hero h1 { color: white; margin: 0 0 6px; font-size: 1.9rem; }
#hero p  { color: #e0e7ff !important; margin: 0; font-size: 1rem; }
#hero b, #hero .pill { color: white !important; }
#hero .pill {
    display: inline-block; margin: 12px 8px 0 0; padding: 4px 12px;
    border-radius: 999px; font-size: .8rem; font-weight: 600;
    background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3);
}

#chat { border-radius: 18px !important; border: 1px solid #e0e7ff !important;
        box-shadow: 0 6px 24px rgba(15, 23, 42, .06); background: white !important; }
#chat .message.user, #chat [data-testid="user"] {
    background: linear-gradient(120deg, #4f46e5, #6366f1) !important;
    color: white !important; border: none !important;
}
#chat [data-testid="user"] * { color: white !important; }
#chat [data-testid="bot"] {
    background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
}

.cite {
    display: inline-block; min-width: 1.4em; padding: 0 6px; margin: 0 1px;
    border-radius: 999px; font-size: .72rem; font-weight: 700; line-height: 1.5;
    text-align: center; background: #e0e7ff; color: #3730a3; vertical-align: 2px;
}

.src-box { margin-top: 10px; }
.src-box summary { cursor: pointer; font-weight: 600; color: #475569; margin-bottom: 6px; }
.src-card { border-radius: 12px; padding: 10px 12px; margin: 8px 0;
            border-left: 5px solid; background: white; }
.src-eu    { border-color: #2563eb; background: #eff6ff; }
.src-nist  { border-color: #0d9488; background: #f0fdfa; }
.src-other { border-color: #9333ea; background: #faf5ff; }
.src-head  { font-size: .82rem; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.src-num   { font-weight: 700; color: #334155; }
.src-badge { color: white !important; padding: 2px 9px; border-radius: 999px; color: white; font-weight: 600; font-size: .72rem; }
.src-eu .src-badge    { background: #2563eb; }
.src-nist .src-badge  { background: #0d9488; }
.src-other .src-badge { background: #9333ea; }
.src-page  { color: #475569; font-weight: 600; }
.src-score { color: #94a3b8; margin-left: auto; }
.src-text  { font-size: .85rem; color: #334155; margin-top: 6px; line-height: 1.5; }

#chat [data-testid="bot"], #chat .answer { color: #0f172a !important; }
.answer { font-size: 1rem; line-height: 1.6; margin-bottom: 12px; }

.origin { background: linear-gradient(135deg, #eef2ff, #f0fdfa); border: 1px solid #c7d2fe;
          border-radius: 14px; padding: 12px 14px; margin: 6px 0 10px; }
.origin-title { font-weight: 700; color: #312e81; margin-bottom: 8px; }
.trace { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 10px; font-size: .82rem; }
.trace .step { background: white; border: 1px solid #e0e7ff; border-radius: 999px; padding: 3px 10px; color: #334155; }
.trace .step.used { background: #dcfce7; border-color: #86efac; color: #166534; }
.trace .arrow { color: #818cf8; font-weight: 700; }
.doc-row { background: white; border-radius: 10px; padding: 8px 10px; margin: 6px 0; border-left: 5px solid #9333ea; }
.doc-row.src-eu, .all-row.src-eu { border-left-color: #2563eb; }
.doc-row.src-nist, .all-row.src-nist { border-left-color: #0d9488; }
.doc-name { display: flex; gap: 8px; align-items: center; margin-bottom: 6px; }
.doc-status { font-size: .72rem; font-weight: 600; border-radius: 999px; padding: 1px 8px; }
.doc-status.yes { background: #dcfce7; color: #166534; }
.doc-status.no  { background: #f1f5f9; color: #64748b; }
.doc-pages { display: flex; flex-wrap: wrap; gap: 4px; }
.page-chip { font-size: .72rem; padding: 1px 8px; border-radius: 6px; background: #f1f5f9; color: #64748b; border: 1px solid #e2e8f0; }
.page-chip.cited { background: #4f46e5; color: white !important; border-color: #4f46e5; font-weight: 700; }
.legend { font-size: .72rem; color: #64748b; margin-top: 8px; }

.all-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 6px 10px;
           border-radius: 8px; margin: 4px 0; background: white; border-left: 4px solid #9333ea; font-size: .8rem; }
.all-row.is-cited { background: #f0fdf4; }
.all-snippet { flex-basis: 100%; color: #64748b; font-size: .76rem; }

.no-info   { background: #fffbeb; border: 1px solid #fcd34d; color: #92400e;
             border-radius: 12px; padding: 10px 14px; }
.error-msg { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b;
             border-radius: 12px; padding: 10px 14px; }

#input-row { background: white; border-radius: 16px; padding: 8px;
             border: 1px solid #c7d2fe; box-shadow: 0 4px 16px rgba(79,70,229,.08); }
#send-btn { min-width: 110px; }
#examples button { border-radius: 999px !important; background: white !important;
                   border: 1px solid #c7d2fe !important; color: #3730a3 !important; }
#examples button:hover { background: #eef2ff !important; }
"""


# ---------- Layout ----------

with gr.Blocks(title="AI Regulation Assistant") as demo:

    gr.HTML(
        """
        <div id="hero">
          <h1>🤖 AI Regulation Assistant</h1>
          <p>Ask anything about the <b>EU AI Act</b> and the
             <b>NIST AI Risk Management Framework</b>.
             Every answer cites the exact page it came from.</p>
          <span class="pill">🇪🇺 EU AI Act 2024/1689</span>
          <span class="pill">🇺🇸 NIST AI RMF 1.0</span>
        </div>
        """
    )

    chatbot = gr.Chatbot(
        elem_id="chat",
        height=520,
        show_label=False,
        sanitize_html=False,
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
            placeholder="Type your question and press Enter…",
            lines=1,
            max_lines=4,
            scale=8,
            container=False,
            autofocus=True,
        )
        send = gr.Button("Send ➤", variant="primary", scale=1, elem_id="send-btn")
        clear = gr.Button("🗑️ Clear", variant="secondary", scale=1)

    gr.Markdown("**Try one of these:**")
    with gr.Row(elem_id="examples"):
        example_buttons = [gr.Button(text, size="sm") for text in EXAMPLES]

    # Enter key and Send button both ask the question.
    ask_events = [
        question.submit(respond, [question, chatbot], [question, chatbot]),
        send.click(respond, [question, chatbot], [question, chatbot]),
    ]

    for button in example_buttons:
        ask_events.append(
            button.click(use_example, [button, chatbot], [question, chatbot])
        )

    # Clear runs instantly (no queue) and stops any answer in progress.
    clear.click(
        lambda: ("", []),
        None,
        [question, chatbot],
        queue=False,
        cancels=ask_events,
    )


# Always use the light theme so the colours stay readable.
FORCE_LIGHT = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, js=FORCE_LIGHT)
