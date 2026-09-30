import html
import json
import time

import gradio as gr

from agent import FALLBACK_MESSAGE, run_agent
from generate_answer import generate_answer
from multi_agent import run_multi_agent
from retrieval import SEARCH_MODE, TOP_K, retrieve
from send_message import approve, reject


# =========================================================
# CONFIGURATION
# =========================================================

MODES = [
    "🤖 Single agent",
    "👥 Multi-agent team",
    "📚 RAG pipeline (no agent)",
]

EXAMPLES = [
    "What is the fine in Article 99(3), what is half of it, and how "
    "many days ago did the EU AI Act start to apply generally?",
    "How many times does each document use the term 'trustworthy', "
    "and what is the total?",
    "What are the four core functions of the NIST AI RMF?",
    "What is the difference between a provider and a deployer?",
    "What are the four core functions of the NIST AI RMF? "
    "Send the answer to our Slack channel.",
]

TOOL_ICONS = {
    "search_corpus": "🔍",
    "quote_exact": "📄",
    "get_article": "📜",
    "count_term": "🔢",
    "calculator": "🧮",
    "date_tool": "📅",
    "send_message": "📤",
    "ask_eu_ai_act_agent": "🇪🇺",
    "ask_nist_ai_rmf_agent": "🇺🇸",
}

AGENT_TITLES = {
    "eu_ai_act_agent": "🇪🇺 EU AI Act agent",
    "nist_ai_rmf_agent": "🇺🇸 NIST AI RMF agent",
}

WAITING_PANEL = (
    "### 🤖 Agent Steps\n\n"
    "Ask a question to see which tools the agent uses, step by step."
)

# Long texts are shortened in the "View code" dropdowns.
MAX_TEXT_IN_CODE = 300


# =========================================================
# FORMAT AGENT STEPS
# =========================================================

def shorten(value):
    """Shorten long strings inside a tool result for display."""
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT_IN_CODE \
            else value[:MAX_TEXT_IN_CODE] + " …"
    if isinstance(value, list):
        return [shorten(item) for item in value]
    if isinstance(value, dict):
        return {key: shorten(item) for key, item in value.items()}
    return value


def code_dropdown(step: dict) -> str:
    """A collapsible block with the exact tool call and its result."""
    call = {"tool": step["tool"], "arguments": step["arguments"]}
    code = (
        "// Tool call\n"
        + json.dumps(call, ensure_ascii=False, indent=2)
        + "\n// Result\n"
        + json.dumps(shorten(step.get("result", {})),
                     ensure_ascii=False, indent=2)
    )
    # One HTML line (no blank lines), so Markdown leaves the code untouched.
    code = html.escape(code).replace("\n", "&#10;")
    return (
        '<details class="step-code"><summary>View code</summary>'
        f"<pre><code>{code}</code></pre></details>"
    )


def render_steps(steps: list) -> list[str]:
    """Group tool calls by step; mark steps that ran tools in parallel."""
    lines = []
    by_step = {}
    for step in steps:
        by_step.setdefault(step["step"], []).append(step)

    for number, calls in by_step.items():
        parallel = (f" · ⚡ {len(calls)} tools in parallel"
                    if len(calls) > 1 else "")
        lines += [f"#### Step {number}{parallel}", ""]

        for call in calls:
            mark = "✅" if call["ok"] else "❌"
            icon = TOOL_ICONS.get(call["tool"], "🛠️")
            lines += [
                f"{mark} {icon} **`{call['tool']}`** — "
                f"{html.escape(str(call['summary']))}",
                "",
                code_dropdown(call),
                "",
            ]

    return lines


def tool_chain(steps: list) -> str:
    """'get_article ×2 → calculator → date_tool' in the order used."""
    order, counts = [], {}
    for step in steps:
        name = step["tool"]
        if name not in counts:
            order.append(name)
        counts[name] = counts.get(name, 0) + 1

    return " → ".join(
        f"{name} ×{counts[name]}" if counts[name] > 1 else name
        for name in order
    ) or "no tools"


def format_single_run(run: dict) -> str:
    """Agent Steps panel for the single agent."""
    distinct = len({step["tool"] for step in run["steps"]})
    lines = [
        "### 🤖 Agent Steps — single agent",
        "",
        f"**🧰 Tools used ({distinct}):** {tool_chain(run['steps'])}  ",
        f"**Model calls:** {run['model_calls']} · "
        f"**Time:** {run.get('seconds', 0)} s · "
        f"**Stopped:** {run['stop_reason']}",
        "",
    ]

    if not run["steps"]:
        lines += ["No tools were needed for this question.", ""]
    lines += render_steps(run["steps"])

    lines += final_section(run)
    return "\n".join(lines)


def format_multi_run(run: dict) -> str:
    """Agent Steps panel for the coordinator + specialist agents."""
    agents = [AGENT_TITLES.get(name, name) for name in run["agents_used"]]
    lines = [
        "### 👥 Agent Steps — multi-agent team",
        "",
        f"**Agents used ({len(agents)}):** "
        f"{', '.join(agents) or 'coordinator only'}  ",
        f"**Model calls:** {run['model_calls']} · "
        f"**Time:** {run.get('seconds', 0)} s · "
        f"**Stopped:** {run['stop_reason']}",
        "",
        "### 🧭 Coordinator",
        "",
        f"**🧰 Tools used:** {tool_chain(run['coordinator_steps'])}",
        "",
    ]
    lines += render_steps(run["coordinator_steps"])

    for specialist in run["specialist_runs"]:
        title = AGENT_TITLES.get(specialist["agent"], specialist["agent"])
        lines += [
            f"### {title}",
            "",
            f"**Asked:** {html.escape(specialist['question'])}  ",
            f"**🧰 Tools used:** {tool_chain(specialist['steps'])}",
            "",
        ]
        lines += render_steps(specialist["steps"])

    lines += final_section(run)
    return "\n".join(lines)


def final_section(run: dict) -> list[str]:
    """Closing lines: citations found in the final answer."""
    citations = ", ".join(
        f"{c['source']} p. {c['page']}" for c in run["citations"]
    ) or "none"
    return ["#### ✅ Final answer", "", f"**Citations:** {citations}", ""]


def tools_line(run: dict, multi: bool) -> str:
    """One short line under the chat answer listing the tools used."""
    if multi:
        parts = [f"coordinator: {tool_chain(run['coordinator_steps'])}"]
        parts += [
            f"{AGENT_TITLES.get(s['agent'], s['agent'])}: {tool_chain(s['steps'])}"
            for s in run["specialist_runs"]
        ]
        return "🧰 " + " · ".join(parts)
    return f"🧰 Tools used: {tool_chain(run['steps'])}"


# =========================================================
# RAG PIPELINE PANEL (no agent)
# =========================================================

def format_rag_run(chunks: list, seconds: float) -> str:
    """Steps panel for the fixed RAG pipeline: retrieve, then generate."""
    lines = [
        "### 📚 RAG pipeline — no agent",
        "",
        f"**Fixed steps:** retrieve (top {TOP_K}, {SEARCH_MODE} search) "
        f"→ generate_answer (gpt-4o-mini) · **Time:** {seconds} s",
        "",
        "#### Step 1 · 🔍 retrieve",
        "",
    ]
    for number, chunk in enumerate(chunks, start=1):
        ranks = (f"vector rank {chunk.get('vector_rank')}, "
                 f"keyword rank {chunk.get('keyword_rank') or '—'}")
        lines += [
            f"**[{number}]** `{chunk['source']}` — page **{chunk['page']}** "
            f"— similarity {chunk['score']:.3f} ({ranks})",
            "",
            code_dropdown({"tool": "retrieved passage",
                           "arguments": {"rank": number},
                           "result": {"text": chunk["text"]}}),
            "",
        ]
    lines += ["#### Step 2 · ✍️ generate_answer", "",
              "The model answers only from the passages above, citing [1]–"
              f"[{len(chunks)}].", ""]
    return "\n".join(lines)


# =========================================================
# HUMAN APPROVAL FOR send_message
# =========================================================

def approval_view(pending: list):
    """Show the first proposed message with Yes/No buttons, or hide the box."""
    if not pending:
        return [], gr.update(visible=False), ""

    proposal = pending[0]
    quoted = "\n".join(f"> {line}" if line else ">"
                       for line in proposal["text"].splitlines())
    text = (
        "### 📤 The agent wants to send this message\n\n"
        f"**Destination:** {proposal['backend']} · "
        f"**Proposal:** `{proposal['proposal_id']}`\n\n"
        f"{quoted}\n\n"
        "Nothing is sent until you click **✅ Yes, send it**."
    )
    return pending, gr.update(visible=True), text


def decide_message(pending: list, history: list, send_it: bool):
    """The person's decision: send the proposed message, or cancel it."""
    if not pending:
        return (history, *approval_view([]))

    proposal = pending[0]

    if send_it:
        result = approve(proposal["proposal_id"])
        note = ("📤 Message sent ✅" if result.get("ok")
                else f"⚠️ Sending failed: {result.get('error')}")
    else:
        reject(proposal["proposal_id"])
        note = "❌ Message cancelled: nothing was sent."

    history = history + [{"role": "assistant", "content": note}]
    return (history, *approval_view(pending[1:]))


# =========================================================
# CHAT LOGIC
# =========================================================

def respond(message: str, history: list, mode: str):
    """Run the selected mode and show its answer, steps and approvals.

    Yields: question box, chat, steps panel, pending proposals,
    approval box visibility, approval text.
    """
    message = (message or "").strip()

    if not message:
        yield ("", history, WAITING_PANEL, *approval_view([]))
        return

    multi = mode == MODES[1]
    rag = mode == MODES[2]

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "🤖 Working..."},
    ]

    if rag:
        status = "retrieving passages and generating the answer"
    elif multi:
        status = "coordinator is asking the specialist agents"
    else:
        status = "agent is choosing which tools to use"

    yield ("", history, f"### 🤖 Agent Steps\n\n⏳ The {status}...",
           *approval_view([]))

    try:
        if rag:
            started = time.time()
            chunks = retrieve(message)
            answer = generate_answer(message, chunks) or FALLBACK_MESSAGE
            history[-1]["content"] = (
                f"{answer}\n\n<sub>📚 RAG pipeline: retrieve → "
                "generate_answer</sub>"
            )
            panel = format_rag_run(chunks, round(time.time() - started, 2))
            yield ("", history, panel, *approval_view([]))
            return

        run = run_multi_agent(message) if multi else run_agent(message)
        answer = run["answer"] or FALLBACK_MESSAGE

        history[-1]["content"] = (
            f"{answer}\n\n<sub>{html.escape(tools_line(run, multi))}</sub>"
        )
        panel = format_multi_run(run) if multi else format_single_run(run)
        yield ("", history, panel,
               *approval_view(run.get("pending_messages", [])))

    except Exception as error:  # Show errors in the chat, not a crash.
        history[-1]["content"] = f"⚠️ Something went wrong: {error}"
        yield (
            "",
            history,
            "### 🤖 Agent Steps\n\n❌ An error occurred while running.",
            *approval_view([]),
        )


# =========================================================
# EXAMPLE BUTTON
# =========================================================

def use_example(example: str, history: list, mode: str):
    yield from respond(example, history, mode)


# =========================================================
# CLEAR
# =========================================================

def clear_all():
    return ("", [], WAITING_PANEL, *approval_view([]))


# =========================================================
# THEME
# =========================================================

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Inter"),
        "system-ui",
        "sans-serif"
    ],
)


# =========================================================
# CSS
# =========================================================

CSS = """

.gradio-container {
    max-width: 1050px !important;
    margin: 0 auto !important;
}

body,
.gradio-container {
    background:
        linear-gradient(
            160deg,
            #eef2ff 0%,
            #f0fdfa 55%,
            #fdf4ff 100%
        ) !important;
}


/* HEADER */

#hero {
    background:
        linear-gradient(
            120deg,
            #312e81 0%,
            #4f46e5 50%,
            #0d9488 100%
        );

    border-radius: 20px;
    padding: 26px 30px;
    color: white;

    box-shadow:
        0 12px 30px
        rgba(79, 70, 229, .25);
}


#hero h1 {
    color: white;
    margin: 0 0 6px;
    font-size: 1.9rem;
}


#hero p {
    color: #e0e7ff !important;
    margin: 0;
}


#hero b {
    color: white !important;
}


#hero .pill {
    display: inline-block;

    margin:
        12px 8px 0 0;

    padding:
        4px 12px;

    border-radius:
        999px;

    font-size:
        .8rem;

    font-weight:
        600;

    color:
        white !important;

    background:
        rgba(255,255,255,.16);

    border:
        1px solid
        rgba(255,255,255,.3);
}


/* CHAT */

#chat {
    border-radius:
        18px !important;

    border:
        1px solid
        #e0e7ff !important;

    box-shadow:
        0 6px 24px
        rgba(15,23,42,.06);

    background:
        white !important;
}


/* AGENT PANEL */

#agent-panel {
    background:
        white !important;

    border:
        1px solid
        #c7d2fe !important;

    border-radius:
        16px !important;

    padding:
        8px 18px !important;

    box-shadow:
        0 5px 20px
        rgba(79,70,229,.07);
}


#agent-panel h3 {
    color:
        #3730a3 !important;
}


#agent-panel h4 {
    color:
        #0f766e !important;

    margin-top:
        18px !important;
}


#agent-panel code {
    background:
        #eef2ff !important;

    color:
        #3730a3 !important;

    border-radius:
        5px;

    padding:
        2px 5px;
}


/* INPUT */

#input-row {
    background:
        white;

    border-radius:
        16px;

    padding:
        8px;

    border:
        1px solid
        #c7d2fe;

    box-shadow:
        0 4px 16px
        rgba(79,70,229,.08);
}


#send-btn {
    min-width:
        110px;
}


/* EXAMPLES */

#examples button {
    border-radius:
        999px !important;

    background:
        white !important;

    border:
        1px solid
        #c7d2fe !important;

    color:
        #3730a3 !important;
}


#examples button:hover {
    background:
        #eef2ff !important;
}

/* AGENT STEP CODE DROPDOWNS */
#agent-panel details.step-code {
    margin: 4px 0 12px 22px;
}
#agent-panel details.step-code summary {
    cursor: pointer;
    color: #4f46e5;
    font-size: .82rem;
    font-weight: 600;
}
#agent-panel details.step-code pre {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: .76rem;
    max-height: 320px;
    overflow: auto;
}
#agent-panel details.step-code pre code {
    background: transparent !important;
    color: #e2e8f0 !important;
    padding: 0;
}
#approval-box {
    border: 2px solid #f59e0b !important;
    background: #fffbeb !important;
    border-radius: 16px !important;
    padding: 10px 16px !important;
}
#approval-box blockquote {
    background: white;
    border-left: 4px solid #f59e0b;
    padding: 6px 12px;
}
#mode-row {
    margin-top: 6px;
}
"""


# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(
    title="AI Regulation Assistant"
) as demo:


    # =====================================================
    # HEADER
    # =====================================================

    gr.HTML(
        """
        <div id="hero">

            <h1>
                🤖 AI Regulation Assistant
            </h1>

            <p>
                Ask questions about the
                <b>EU AI Act</b>
                and
                <b>NIST AI Risk Management Framework</b>.
                The Agent Steps panel shows every tool
                and agent used, with the code of each step.
            </p>

            <span class="pill">
                🇪🇺 EU AI Act
            </span>

            <span class="pill">
                🇺🇸 NIST AI RMF
            </span>

            <span class="pill">
                🔍 Agentic RAG
            </span>

        </div>
        """
    )


    # =====================================================
    # CHATBOT
    # =====================================================

    chatbot = gr.Chatbot(
        elem_id="chat",

        height=500,

        show_label=False,

        placeholder=(
            "<div style='text-align:center;color:#64748b'>"
            "<div style='font-size:2.4rem'>🤖</div>"
            "<b>Ask your first question</b><br>"
            "The answer appears here and the agent actions "
            "appear below."
            "</div>"
        ),
    )


    # =====================================================
    # HUMAN APPROVAL (send_message)
    # =====================================================

    pending = gr.State([])

    with gr.Group(visible=False, elem_id="approval-box") as approval_box:
        approval_text = gr.Markdown()
        with gr.Row():
            yes_button = gr.Button("✅ Yes, send it", variant="primary")
            no_button = gr.Button("❌ No, cancel", variant="stop")


    # =====================================================
    # AGENT STEPS PANEL
    # =====================================================

    with gr.Accordion(
        "🤖 Agent Steps",
        open=True
    ):

        agent_steps = gr.Markdown(
            value=(
                "### 🤖 Agent Steps\n\n"
                "Ask a question to see the agent's actions."
            ),

            elem_id="agent-panel"
        )


    # =====================================================
    # AGENT MODE
    # =====================================================

    with gr.Row(elem_id="mode-row"):
        mode = gr.Radio(
            MODES,
            value=MODES[0],
            label="Mode",
            info="Single agent with its tools, a coordinator with EU AI Act "
                 "and NIST specialist agents, or the original RAG pipeline.",
        )


    # =====================================================
    # INPUT
    # =====================================================

    with gr.Row(
        elem_id="input-row",
        equal_height=True
    ):

        question = gr.Textbox(
            show_label=False,

            placeholder=(
                "Ask about the EU AI Act or NIST AI RMF..."
            ),

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
            elem_id="send-btn"
        )


        clear = gr.Button(
            "🗑️ Clear",
            variant="secondary",
            scale=1
        )


    # =====================================================
    # EXAMPLES
    # =====================================================

    gr.Markdown(
        "**Try one of these:**"
    )


    with gr.Row(
        elem_id="examples"
    ):

        example_buttons = [
            gr.Button(
                example,
                size="sm"
            )

            for example in EXAMPLES
        ]


    # =====================================================
    # ENTER KEY
    # =====================================================

    enter_event = question.submit(
        respond,

        [
            question,
            chatbot,
            mode
        ],

        [
            question,
            chatbot,
            agent_steps,
            pending,
            approval_box,
            approval_text
        ]
    )


    # =====================================================
    # SEND BUTTON
    # =====================================================

    send_event = send.click(
        respond,

        [
            question,
            chatbot,
            mode
        ],

        [
            question,
            chatbot,
            agent_steps,
            pending,
            approval_box,
            approval_text
        ]
    )


    ask_events = [
        enter_event,
        send_event
    ]


    # =====================================================
    # EXAMPLE BUTTONS
    # =====================================================

    for button in example_buttons:

        event = button.click(
            use_example,

            [
                button,
                chatbot,
                mode
            ],

            [
                question,
                chatbot,
                agent_steps,
                pending,
                approval_box,
                approval_text
            ]
        )

        ask_events.append(
            event
        )


    # =====================================================
    # YES / NO FOR PROPOSED MESSAGES
    # =====================================================

    yes_button.click(
        lambda proposals, history: decide_message(proposals, history, True),
        [pending, chatbot],
        [chatbot, pending, approval_box, approval_text],
    )

    no_button.click(
        lambda proposals, history: decide_message(proposals, history, False),
        [pending, chatbot],
        [chatbot, pending, approval_box, approval_text],
    )


    # =====================================================
    # CLEAR BUTTON
    # =====================================================

    clear.click(
        clear_all,

        None,

        [
            question,
            chatbot,
            agent_steps,
            pending,
            approval_box,
            approval_text
        ],

        queue=False,

        cancels=ask_events,
    )


# =========================================================
# FORCE LIGHT THEME
# =========================================================

FORCE_LIGHT = """
() => {

    const url = new URL(window.location);

    if (
        url.searchParams.get('__theme')
        !==
        'light'
    ) {

        url.searchParams.set(
            '__theme',
            'light'
        );

        window.location.href =
            url.href;
    }
}
"""


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    demo.launch(
        theme=THEME,
        css=CSS,
        js=FORCE_LIGHT
    )