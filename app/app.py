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
# FORMAT AGENT TRACE AS MARKDOWN
# =========================================================

def format_agent_trace(trace: list) -> str:

    if not trace:

        return (
            "### 🤖 Agent Steps\n\n"
            "No agent actions recorded."
        )


    lines = [
        "### 🤖 Agent Steps",
        "",
        "This panel shows the agent's **tool and retrieval actions**.",
        "",
    ]


    step_number = 1


    for event in trace:

        # =================================================
        # SEARCH
        # =================================================

        if event["type"] == "search":

            lines.append(
                f"#### {step_number}. 🔍 Search Attempt "
                f"{event['attempt']}"
            )

            lines.append("")

            lines.append(
                f"**Tool:** `{event['tool']}`"
            )

            lines.append("")

            lines.append(
                f"**Query:** `{event['query']}`"
            )

            lines.append("")

            lines.append(
                f"**Top K:** {event['k']}"
            )

            lines.append("")

            step_number += 1


        # =================================================
        # RETRIEVAL
        # =================================================

        elif event["type"] == "retrieval":

            lines.append(
                f"#### {step_number}. 📚 Retrieval Result"
            )

            lines.append("")

            lines.append(
                f"Found **{event['count']} passages**."
            )

            lines.append("")


            # Get unique pages
            pages = []

            # Get unique documents
            documents = []


            for match in event["matches"]:

                page = match["page"]
                source = match["source"]

                if page not in pages:
                    pages.append(page)

                if source not in documents:
                    documents.append(source)


            if documents:

                lines.append(
                    "**Document:** "
                    + ", ".join(
                        f"`{doc}`"
                        for doc in documents
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


            # Show each retrieved match
            lines.append(
                "**Retrieved matches:**"
            )

            lines.append("")


            for number, match in enumerate(
                event["matches"],
                start=1
            ):

                lines.append(
                    f"- Match {number}: "
                    f"`{match['source']}` — "
                    f"page **{match['page']}** — "
                    f"score **{match['score']}**"
                )


            lines.append("")

            step_number += 1


        # =================================================
        # RETRIEVAL FAILED
        # =================================================

        elif event["type"] == "retrieval_failed":

            lines.append(
                f"#### {step_number}. ⚠️ Retrieval Insufficient"
            )

            lines.append("")

            lines.append(
                event["message"]
            )

            lines.append("")

            lines.append(
                "The agent can reformulate the search query."
            )

            lines.append("")

            step_number += 1


        # =================================================
        # FINAL ANSWER
        # =================================================

        elif event["type"] == "final":

            lines.append(
                f"#### {step_number}. ✅ Final Answer"
            )

            lines.append("")

            lines.append(
                f"**Searches performed:** "
                f"{event['search_count']}"
            )

            lines.append("")

            lines.append(
                event["message"]
            )

            lines.append("")

            step_number += 1


        # =================================================
        # NO SEARCH
        # =================================================

        elif event["type"] == "no_search":

            lines.append(
                f"#### {step_number}. 🚫 No Document Search"
            )

            lines.append("")

            lines.append(
                event["message"]
            )

            lines.append("")

            step_number += 1


    return "\n".join(lines)


# =========================================================
# CHAT LOGIC
# =========================================================

def respond(message: str, history: list):

    message = (
        message
        or ""
    ).strip()


    if not message:

        yield (
            "",
            history,
            "### 🤖 Agent Steps\n\nWaiting for a question..."
        )

        return


    # =====================================================
    # ADD USER MESSAGE
    # =====================================================

    history = history + [
        {
            "role": "user",
            "content": message
        },

        {
            "role": "assistant",
            "content": "🤖 Agent is working..."
        }
    ]


    # Show working status
    yield (
        "",
        history,
        (
            "### 🤖 Agent Steps\n\n"
            "⏳ The agent is deciding whether to search "
            "the document corpus..."
        )
    )


    try:

        # =================================================
        # RUN AGENT
        # =================================================

        final_message, trace = run_agent(
            message
        )


        answer = (
            final_message.content
            or FALLBACK_MESSAGE
        )


        # =================================================
        # FINAL CHAT ANSWER
        # =================================================

        history[-1]["content"] = answer


        # =================================================
        # AGENT TRACE
        # =================================================

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
            f"⚠️ Something went wrong: {error}"
        )


        yield (
            "",
            history,
            (
                "### 🤖 Agent Steps\n\n"
                "❌ An error occurred while running the agent."
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
                The Agent Steps panel shows how the
                assistant searches the documents.
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
            chatbot
        ],

        [
            question,
            chatbot,
            agent_steps
        ]
    )


    # =====================================================
    # SEND BUTTON
    # =====================================================

    send_event = send.click(
        respond,

        [
            question,
            chatbot
        ],

        [
            question,
            chatbot,
            agent_steps
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
                chatbot
            ],

            [
                question,
                chatbot,
                agent_steps
            ]
        )

        ask_events.append(
            event
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
            agent_steps
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