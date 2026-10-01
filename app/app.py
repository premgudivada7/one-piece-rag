import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from app import config
from app.agent import AgentRun
from app.chunking import page_label
from app.index import get_index
from app.llm import load_costs
from app.messaging import append_jsonl, backend_label
from app.rag import answer_question, citation_labels
from app.retrieval import retrieve, score_name, threshold
from app.ui_components import (CSS, document_link_markdown, document_url, how_it_works_markdown, reranker_name,
                               source_cards, status_line, steps_markdown)


def text_of(content) -> str:
    """Gradio 6 gives message content as a list of {"type": "text", "text": ...} blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content or "")


SOURCES_FOOTER = "\n\n📄 _Sources:"


def plain_history(chat: list) -> list[dict]:
    """Earlier questions and final answers only (no tool-call messages, no source footer)."""
    return [{"role": m["role"], "content": text_of(m["content"]).split(SOURCES_FOOTER)[0]}
            for m in chat if m["role"] in ("user", "assistant") and not m.get("metadata")]


def spend_line() -> str:
    costs = load_costs()
    return f"💰 Tracked spend: **${costs['total']:.4f}** / ${config.BUDGET_USD:.2f} ({costs['calls']} API calls)"


def header_markdown() -> str:
    title = config.APP_TITLE + (f" by Team {config.TEAM_NAME}" if config.TEAM_NAME else "")
    try:
        index = get_index()
        kb = config.KNOWLEDGE_BASE or " · ".join(index.source_names)
        corpus = f"{len(index.source_names)} documents, {len(index.chunks)} chunks"
        links = [f"[📄 {name}]({document_url(name)})" for name in index.source_names if document_url(name)]
    except Exception as e:
        kb, corpus, links = config.KNOWLEDGE_BASE or "-", f"⚠️ index not loaded: {e}", []
    documents = f"**Documents:** {' · '.join(links)}\n\n" if links else ""
    rerank = (f"reranked by `{reranker_name()}`, refusal below reranker score {config.RERANK_MIN_SCORE}"
              if config.USE_RERANK else f"refusal below similarity {config.MIN_SCORE}")
    warning = ("\n\n⚠️ **INJECTION_TEST is on**: a poisoned test passage (injection_test.txt) is in the index."
               if config.INJECTION_TEST else "")
    return (f"# {title}\n\n**Knowledge base:** {kb} ({corpus})\n\n{documents}"
            f"**Models:** chat `{config.CHAT_MODEL}` · embeddings `{config.EMBED_MODEL}` (local) · "
            f"{config.RETRIEVAL_MODE} search (BM25 + vectors + exact references), top-{config.TOP_K}, {rerank}{warning}")


def with_citation_footer(answer: str, results: list[dict], refused: bool) -> str:
    """Document + page links under the answer, even if the model forgot to cite."""
    if refused or not results:
        return answer
    refs = citation_labels(answer) or [f"{r['source']} {page_label(r)}" for r in results[:3]]
    return f"{answer}{SOURCES_FOOTER} {', '.join(document_link_markdown(ref) for ref in refs)}_"


EMPTY_SOURCES = "<div class='src-summary'>Sources appear here after your first question.</div>"
HIDDEN = gr.update(visible=False)


def rag_respond(message: str, chat: list):
    """Outputs: chat, input box, sources panel, status line, 3 follow-up buttons."""
    message = (message or "").strip()
    if not message:
        yield chat, "", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    history = plain_history(chat)
    chat = chat + [{"role": "user", "content": message}]
    searching = "<div class='status-line'>🔎 Searching the documents…</div>"
    yield (chat + [{"role": "assistant", "content": "🔎 Searching the documents…"}], "", gr.skip(), searching,
           HIDDEN, HIDDEN, HIDDEN)
    try:
        result = answer_question(message, history=history or None, with_follow_ups=True)
    except Exception as e:
        yield (chat + [{"role": "assistant", "content": f"⚠️ Error: {type(e).__name__}: {e}"}], "", gr.skip(),
               f"<div class='status-line'>⚠️ {type(e).__name__}</div>", HIDDEN, HIDDEN, HIDDEN)
        return
    title, body = steps_markdown(result)
    answer = with_citation_footer(result["answer"], result["results"], result["refused"])
    chat = chat + [{"role": "assistant", "content": body, "metadata": {"title": title}},
                   {"role": "assistant", "content": answer}]
    note = ""
    if result["below_threshold"]:
        best = result["top_score"]
        note = (f"Nothing in the documents is close enough: best {score_name(result['results'])} "
                f"{'none' if best is None else f'{best:.2f}'}, needed {threshold(result['results'])}. "
                "The model was not called.")
    elif result["refused"]:
        note = "The model read the passages below and found no answer to this question in them."
    elif result["search_query"] != message:
        note = f"Searched for: {result['search_query']}"
    sources = source_cards(result["results"], result["answer"], result["search_query"], note=note,
                           refused=result["refused"])
    buttons = [gr.update(value=q, visible=True) for q in result["follow_ups"]]
    buttons += [HIDDEN] * (3 - len(buttons))
    yield (chat, "", sources, status_line(result) + f"<div class='status-line'>{spend_line()}</div>", *buttons)


TOOL_ICONS = {"search_corpus": "🔎", "calculator": "🧮", "quote_exact": "📄", "get_article": "📜",
              "date_tool": "📅", "send_message": "📤"}
APPROVAL_TITLE = "📤 send_message: approval needed"


def _args_text(args: dict) -> str:
    return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())


def trajectory_markdown(run: AgentRun | None) -> str:
    if run is None:
        return "### Tool calls\n_Ask a question: every tool call the agent makes appears here, in order._"
    lines = ["### Tool calls"]
    n = 0
    for e in run.trace:
        if e["type"] == "model":
            decided = ", ".join(e["tool_calls"]) if e["tool_calls"] else "final answer"
            lines.append(f"<sub>model call {e['step']} → {decided} · {e['prompt_tokens']} prompt tokens</sub>")
        elif e["type"] == "tool":
            n += 1
            result = html.escape(e["result"])
            flag = "" if "approved" not in e else (" ✅ approved" if e["approved"] else " ❌ rejected")
            icon = TOOL_ICONS.get(e["tool"], "🛠️")
            lines.append(f"**{n}. {icon} {e['tool']}**({html.escape(_args_text(e['args']))}){flag}"
                         f"\n<details><summary>result</summary><pre>{result}</pre></details>")
        elif e["type"] == "approval_needed":
            lines.append("⏸️ **waiting for human approval**")
        elif e["type"] == "check":
            lines.append(f"🔢 number check: {', '.join(e['numbers'])} not in any tool result → asked to use the calculator")
        elif e["type"] == "final":
            cap = " · ⚠️ hit the step cap" if e["hit_step_cap"] else ""
            lines.append(f"✅ **answered after {e['steps']} model calls**{cap}")
        elif e["type"] == "error":
            lines.append(f"⚠️ **error:** {html.escape(e['error'])}")
    lines.append(f"<sub>run cost ${run.cost:.5f} · trace: data/traces/{run.id}.json</sub>")
    return "\n\n".join(lines)


def _tool_message(event: dict) -> dict:
    """A collapsible chat bubble for one tool call."""
    title = f"{TOOL_ICONS.get(event['tool'], '🛠️')} {event['tool']}({_args_text(event['args'])})"
    if len(title) > 120:
        title = title[:117] + "…"
    return {"role": "assistant", "content": event["result"][:1500], "metadata": {"title": title, "status": "done"}}


def _outputs(chat, run, state, approval_visible=False, approval_text=""):
    """Everything the agent handlers update, in the order of `outputs` in build_ui()."""
    sources = (source_cards(run.sources, run.final_answer or "", run.question, mode="agent")
               if run and run.sources else EMPTY_SOURCES)
    return (chat, gr.update(value="", interactive=not approval_visible), trajectory_markdown(run),
            sources, gr.update(visible=approval_visible), approval_text, state)


def _proposal_markdown(run: AgentRun, warning: str = "") -> str:
    """The approval card: exactly what will be posted, where, and how."""
    proposal = run.approval
    quoted = "> " + proposal["text"].replace("\n", "\n> ")
    return (f"{warning}### ⏸️ The agent wants to post this message\n"
            f"**Channel:** `{proposal['channel'] or '(default)'}` · **via** {proposal['backend']}\n\n{quoted}"
            "\n\nNothing is sent until you click **Approve & send**.")


def _drive(run: AgentRun, chat: list, state: dict):
    """Stream each agent step to the UI until it answers or asks for approval."""
    try:
        for event in run.advance():
            if event["type"] == "tool":
                chat = chat + [_tool_message(event)]
            yield _outputs(chat, run, state)
    except Exception as e:
        run.fail(e)
    if run.status == "awaiting_approval":
        chat = chat + [{"role": "assistant", "content": "⏸️ Waiting for your approval before sending (see below).",
                        "metadata": {"title": APPROVAL_TITLE}}]
        yield _outputs(chat, run, state, approval_visible=True, approval_text=_proposal_markdown(run))
        return
    summary = run.summary()
    answer = with_citation_footer(run.final_answer or "", run.sources, summary["refused"])
    chat = chat + [{"role": "assistant", "content": answer}]
    state["history"] = state["history"] + [{"role": "user", "content": run.question},
                                           {"role": "assistant", "content": run.final_answer or ""}]
    yield _outputs(chat, run, state)


def agent_respond(message: str, chat: list, state: dict):
    message = (message or "").strip()
    state = state or {"run": None, "history": []}
    if not message:
        yield _outputs(chat, state.get("run"), state)
        return
    run = state.get("run")
    if run is not None and run.status == "awaiting_approval":
        warning = "⚠️ **Approve or reject the pending message first.**\n\n"
        yield _outputs(chat, run, state, approval_visible=True, approval_text=_proposal_markdown(run, warning))
        return
    chat = chat + [{"role": "user", "content": message}]
    try:
        run = AgentRun(message, history=state["history"])
    except Exception as e:
        yield _outputs(chat + [{"role": "assistant", "content": f"⚠️ Error: {e}"}], None, state)
        return
    state = {**state, "run": run}
    yield _outputs(chat, run, state)
    yield from _drive(run, chat, state)


def agent_decide(approved: bool):
    def handler(chat: list, state: dict):
        run = (state or {}).get("run")
        if run is None or run.status != "awaiting_approval":
            yield _outputs(chat, run, state)
            return
        chat = [m for m in chat if (m.get("metadata") or {}).get("title") != APPROVAL_TITLE]
        if approved:
            yield _outputs(chat + [{"role": "assistant", "content": "📤 Sending…"}], run, state)
        run.resolve_approval(approved)
        chat = chat + [_tool_message(run.trace[-1])]
        yield from _drive(run, chat, state)
    return handler


def reset_agent():
    return _outputs([], None, {"run": None, "history": []})


def log_feedback(tab: str):
    def handler(data: gr.LikeData):
        append_jsonl(config.FEEDBACK_PATH, {"tab": tab, "liked": data.liked, "message": text_of(data.value)})
    return handler


def example_questions(limit=5) -> list[str]:
    """The 5 priority questions from questions.json, as demo buttons."""
    try:
        questions = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    priority = [q["question"] for q in questions if q.get("priority")]
    return (priority or [q["question"] for q in questions])[:limit]


def build_ui() -> gr.Blocks:
    examples = example_questions()
    with gr.Blocks(title=config.APP_TITLE) as demo:
        gr.Markdown(header_markdown())

        with gr.Tab("💬 Chat (RAG)"):
            with gr.Row():
                with gr.Column(scale=3):
                    rag_chat = gr.Chatbot(height=560, label="Conversation", buttons=["copy"])
                    with gr.Row():
                        rag_input = gr.Textbox(placeholder="Ask a question… (follow-up questions work too)",
                                               show_label=False, lines=1, max_lines=4, scale=5)
                        rag_send = gr.Button("Send", variant="primary", scale=1)
                    rag_status = gr.HTML("<div class='status-line'>Ready.</div>")
                    with gr.Row():
                        follow_ups = [gr.Button(visible=False, size="sm") for _ in range(3)]
                    if examples:
                        gr.Examples(examples, inputs=rag_input, label="Try these")
                    rag_reset = gr.Button("🗑️ New conversation", size="sm")
                with gr.Column(scale=2, elem_classes="panel"):
                    gr.Markdown("### Sources")
                    rag_sources = gr.HTML(EMPTY_SOURCES)
            rag_outputs = [rag_chat, rag_input, rag_sources, rag_status, *follow_ups]
            rag_input.submit(rag_respond, [rag_input, rag_chat], rag_outputs)
            rag_send.click(rag_respond, [rag_input, rag_chat], rag_outputs)
            for button in follow_ups:
                button.click(lambda q: q, button, rag_input).then(rag_respond, [rag_input, rag_chat], rag_outputs)
            rag_reset.click(lambda: ([], "", EMPTY_SOURCES, "<div class='status-line'>Ready.</div>",
                                     HIDDEN, HIDDEN, HIDDEN), None, rag_outputs)
            rag_chat.like(log_feedback("rag"))

        with gr.Tab("🤖 Agent"):
            gr.Markdown(f"Tools: search_corpus (top {config.AGENT_TOP_K}, reranked) · calculator · quote_exact · "
                        f"get_article · date_tool · send_message (backend: **{backend_label()}**, always asks for "
                        f"approval) · step cap {config.AGENT_MAX_STEPS} · full results of every call in the panel →")
            agent_state = gr.State({"run": None, "history": []})
            with gr.Row():
                with gr.Column(scale=3):
                    agent_chat = gr.Chatbot(height=520, label="Agent", buttons=["copy"])
                    with gr.Column(visible=False, elem_classes="approval") as approval_box:
                        approval_md = gr.Markdown()
                        with gr.Row():
                            approve_btn = gr.Button("✅ Approve & send", variant="primary")
                            reject_btn = gr.Button("❌ Reject", variant="stop")
                    agent_input = gr.Textbox(placeholder="Ask anything; e.g. “What is 15% of the tuition? "
                                                         "Then post the answer to the team channel.”",
                                             show_label=False, submit_btn=True)
                    agent_examples = examples[:3] + ["Send your last answer to the team channel."]
                    gr.Examples(agent_examples, inputs=agent_input, label="Try")
                    agent_reset = gr.Button("🗑️ New conversation", size="sm")
                with gr.Column(scale=2, elem_classes="panel"):
                    trajectory = gr.Markdown(trajectory_markdown(None))
                    gr.Markdown("### Passages the agent read")
                    agent_sources = gr.HTML(EMPTY_SOURCES)
            outputs = [agent_chat, agent_input, trajectory, agent_sources, approval_box, approval_md, agent_state]
            agent_input.submit(agent_respond, [agent_input, agent_chat, agent_state], outputs)
            approve_btn.click(agent_decide(True), [agent_chat, agent_state], outputs)
            reject_btn.click(agent_decide(False), [agent_chat, agent_state], outputs)
            agent_reset.click(reset_agent, None, outputs)
            agent_chat.like(log_feedback("agent"))
        with gr.Tab("🧰 How it works"):
            try:
                index = get_index()
                gr.Markdown(how_it_works_markdown(len(index.source_names), len(index.chunks)))
            except Exception:
                gr.Markdown(how_it_works_markdown())
    return demo


def warm_up():
    """Load the index, the embedding model and the reranker before the first question."""
    try:
        retrieve("warm up")
    except Exception as e:
        print(f"warm-up skipped: {e}")


if __name__ == "__main__":
    warm_up()
    # allowed_paths lets the "Open PDF" links serve the original documents (data/sources/ only)
    build_ui().queue().launch(theme=gr.themes.Soft(), css=CSS, allowed_paths=[str(config.SOURCES_DIR.resolve())])
