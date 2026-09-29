import gradio as gr

from agent import run_agent


# =========================================================
# CONFIGURATION
# =========================================================

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


# =========================================================
# FORMAT AGENT TRACE
# =========================================================

def format_agent_trace(trace: list) -> str:
    """
    Display observable agent actions:
    - searches
    - queries
    - retrieval results
    - pages
    - similarity scores
    - final status
    """

    if not trace:
        return (
            "### 🤖 Agent Steps\n\n"
            "No agent actions recorded."
        )

    lines = [
        "### 🤖 Agent Steps",
        "",
        "This shows the agent's **tool and retrieval actions**.",
        "",
    ]

    step_number = 1

    for event in trace:

        # -------------------------------------------------
        # SEARCH
        # -------------------------------------------------

        if event["type"] == "search":

            lines.append(
                f"#### {step_number}. 🔍 Search Attempt "
                f"{event['attempt']}"
            )
            lines.append("")

            lines.append(
                f"**Tool:** `{event.get('tool', 'search_corpus')}`"
            )
            lines.append("")

            lines.append(
                f"**Query:** `{event.get('query', '')}`"
            )
            lines.append("")

            lines.append(
                f"**Top K:** {event.get('k', 5)}"
            )
            lines.append("")

            step_number += 1

        # -------------------------------------------------
        # RETRIEVAL SUCCESS
        # -------------------------------------------------

        elif event["type"] == "retrieval":

            lines.append(
                f"#### {step_number}. 📚 Retrieval Result"
            )
            lines.append("")

            matches = event.get("matches", [])

            lines.append(
                f"Found **{event.get('count', len(matches))} passages**."
            )
            lines.append("")

            pages = []
            documents = []

            for match in matches:

                page = match.get("page")
                source = match.get("source")

                if page is not None and page not in pages:
                    pages.append(page)

                if source and source not in documents:
                    documents.append(source)

            if documents:
                lines.append(
                    "**Document:** "
                    + ", ".join(
                        f"`{document}`"
                        for document in documents
                    )
                )
                lines.append("")

            if pages:
                lines.append(
                    "**Pages retrieved:** "
                    + ", ".join(
                        str(page)
                        for page in pages
                    )
                )
                lines.append("")

            if matches:

                lines.append("**Retrieved matches:**")
                lines.append("")

                for number, match in enumerate(
                    matches,
                    start=1
                ):

                    lines.append(
                        f"- Match {number}: "
                        f"`{match.get('source', 'Unknown')}` — "
                        f"page **{match.get('page', '?')}** — "
                        f"score **{match.get('score', '?')}**"
                    )

                lines.append("")

            step_number += 1

        # -------------------------------------------------
        # RETRIEVAL FAILURE
        # -------------------------------------------------

        elif event["type"] == "retrieval_failed":

            lines.append(
                f"#### {step_number}. ⚠️ Retrieval Insufficient"
            )
            lines.append("")

            lines.append(
                event.get(
                    "message",
                    "No useful passages were found."
                )
            )
            lines.append("")

            lines.append(
                "The agent may try a different search query."
            )
            lines.append("")

            step_number += 1

        # -------------------------------------------------
        # FINAL ANSWER
        # -------------------------------------------------

        elif event["type"] == "final":

            lines.append(
                f"#### {step_number}. ✅ Final Answer"
            )
            lines.append("")

            lines.append(
                f"**Searches performed:** "
                f"{event.get('search_count', 0)}"
            )
            lines.append("")

            lines.append(
                event.get(
                    "message",
                    "Final answer generated."
                )
            )
            lines.append("")

            step_number += 1

        # -------------------------------------------------
        # NO SEARCH
        # -------------------------------------------------

        elif event["type"] == "no_search":

            lines.append(
                f"#### {step_number}. 🚫 No Document Search"
            )
            lines.append("")

            lines.append(
                event.get(
                    "message",
                    "No document search was performed."
                )
            )
            lines.append("")

            step_number += 1

        # -------------------------------------------------
        # SEARCH LIMIT
        # -------------------------------------------------

        elif event["type"] == "limit":

            lines.append(
                f"#### {step_number}. 🛑 Search Limit"
            )
            lines.append("")

            lines.append(
                event.get(
                    "message",
                    "Maximum search attempts reached."
                )
            )
            lines.append("")

            step_number += 1

    return "\n".join(lines)


# =========================================================
# CHAT
# =========================================================

def respond(message: str, history: list):

    message = (message or "").strip()

    history = history or []

    if not message:

        yield (
            "",
            history,
            "### 🤖 Agent Steps\n\nWaiting for a question..."
        )

        return

    # Make a copy of the history
    history = [
        dict(item)
        for item in history
    ]

    # Add user question
    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    # Temporary assistant message
    history.append(
        {
            "role": "assistant",
            "content": "🤖 Agent is working..."
        }
    )

    # Show immediately
    yield (
        "",
        history,
        (
            "### 🤖 Agent Steps\n\n"
            "⏳ The agent is deciding whether it needs "
            "to search the documents..."
        )
    )

    try:

        # -------------------------------------------------
        # RUN AGENT
        # -------------------------------------------------

        final_message, trace = run_agent(
            message
        )

        answer = final_message.content

        if not answer:
            answer = FALLBACK_MESSAGE

        answer = answer.strip()

        # -------------------------------------------------
        # PUT ONLY ANSWER IN CHAT
        # -------------------------------------------------

        history[-1]["content"] = answer

        # -------------------------------------------------
        # PUT STEPS IN SEPARATE PANEL
        # -------------------------------------------------

        trace_markdown = format_agent_trace(
            trace
        )

        yield (
            "",
            history,
            trace_markdown
        )

    except Exception as error:

        history[-1]["content"] = (
            "⚠️ Something went wrong while running the agent."
        )

        yield (
            "",
            history,
            (
                "### 🤖 Agent Steps\n\n"
                "❌ Agent error:\n\n"
                f"`{str(error)}`"
            )
        )


# =========================================================
# EXAMPLE BUTTON
# =========================================================

def use_example(example: str, history: list):

    yield from respond(
        example,
        history
    )


# =========================================================
# CLEAR
# =========================================================

def clear_all():

    return (
        "",
        [],
        (
            "### 🤖 Agent Steps\n\n"
            "Ask a question to see the agent's actions."
        )
    )


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


/* =======================================================
   HEADER
   ======================================================= */

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

    margin-bottom: 14px;

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

    margin: 12px 8px 0 0;

    padding: 4px 12px;

    border-radius: 999px;

    font-size: .8rem;

    font-weight: 600;

    color: white !important;

    background:
        rgba(255, 255, 255, .16);

    border:
        1px solid
        rgba(255, 255, 255, .3);
}


/* =======================================================
   CHAT
   ======================================================= */

#chat {
    border-radius: 18px !important;

    border:
        1px solid
        #e0e7ff !important;

    background:
        white !important;

    box-shadow:
        0 6px 24px
        rgba(15, 23, 42, .06);
}


/* =======================================================
   AGENT PANEL
   ======================================================= */

#agent-panel {
    background:
        white !important;

    border:
        1px solid
        #c7d2fe !important;

    border-radius:
        16px !important;

    padding:
        10px 18px !important;

    box-shadow:
        0 5px 20px
        rgba(79, 70, 229, .07);
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


/* =======================================================
   INPUT
   ======================================================= */

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
        rgba(79, 70, 229, .08);
}


#send-btn {
    min-width:
        110px;
}


/* =======================================================
   EXAMPLES
   ======================================================= */

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

"""


# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(
    title="AI Regulation Assistant"
) as demo:

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

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
                The agent can decide when and how to
                search the documents.
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


    # -----------------------------------------------------
    # CHATBOT
    # -----------------------------------------------------

    chatbot = gr.Chatbot(
        elem_id="chat",
        height=500,
        show_label=False,

        placeholder=(
            "<div style='text-align:center;color:#64748b'>"
            "<div style='font-size:2.4rem'>🤖</div>"
            "<b>Ask your first question</b><br>"
            "The final answer appears here."
            "</div>"
        ),
    )


    # -----------------------------------------------------
    # AGENT STEPS
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # INPUT
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # EXAMPLES
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # ENTER
    # -----------------------------------------------------

    enter_event = question.submit(
        respond,

        inputs=[
            question,
            chatbot
        ],

        outputs=[
            question,
            chatbot,
            agent_steps
        ],
    )


    # -----------------------------------------------------
    # SEND
    # -----------------------------------------------------

    send_event = send.click(
        respond,

        inputs=[
            question,
            chatbot
        ],

        outputs=[
            question,
            chatbot,
            agent_steps
        ],
    )


    ask_events = [
        enter_event,
        send_event
    ]


    # -----------------------------------------------------
    # EXAMPLE BUTTONS
    # -----------------------------------------------------

    for button in example_buttons:

        example_event = button.click(
            use_example,

            inputs=[
                button,
                chatbot
            ],

            outputs=[
                question,
                chatbot,
                agent_steps
            ],
        )

        ask_events.append(
            example_event
        )


    # -----------------------------------------------------
    # CLEAR
    # -----------------------------------------------------

    clear.click(
        clear_all,

        inputs=None,

        outputs=[
            question,
            chatbot,
            agent_steps
        ],

        queue=False,

        cancels=ask_events,
    )


# =========================================================
# FORCE LIGHT MODE
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

    demo.queue().launch(
        theme=THEME,
        css=CSS,
        js=FORCE_LIGHT
    )