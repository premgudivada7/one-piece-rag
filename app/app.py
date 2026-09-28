import gradio as gr

from retrieval import retrieve
from generate_answer import generate_answer


def ask_question(question: str):
    """Run the complete RAG pipeline."""

    if not question.strip():
        return (
            gr.update(
                value="Please enter a question.",
                visible=True,
            ),
            gr.update(
                value="",
                visible=False,
            ),
        )

    # Step 1: Retrieve relevant chunks from FAISS
    retrieved_chunks = retrieve(question)

    # Step 2: Generate answer using GPT-4o-mini
    final_answer = generate_answer(
        question,
        retrieved_chunks,
    )

    # Step 3: Check if documents contain the answer
    fallback_message = (
        "I don't have enough information in the provided "
        "documents to answer this question."
    )

    if final_answer.strip() == fallback_message:
        sources_text = "No relevant sources found."

    else:
        source_lines = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            source_lines.append(
                f"[{i}] {chunk['source']} — "
                f"Page {chunk['page']} "
                f"(score: {chunk['score']:.4f})"
            )

        sources_text = "\n\n".join(source_lines)

    # Show Answer and Sources only after Ask
    return (
        gr.update(
            value=final_answer,
            visible=True,
        ),
        gr.update(
            value=sources_text,
            visible=True,
        ),
    )


with gr.Blocks(
    title="AI Regulation RAG Assistant"
) as demo:

    gr.Markdown(
        """
        # 🤖 AI Regulation RAG Assistant

        Ask questions about the **EU AI Act** and
        **NIST AI Risk Management Framework**.
        """
    )

    # ---------------- QUESTION ----------------

    with gr.Group():
        gr.Markdown("## ❓ Question")

        question = gr.Textbox(
            label="Enter your question",
            placeholder=(
                "Example: What are the four core functions "
                "of NIST AI RMF?"
            ),
            lines=2,
        )

        ask_button = gr.Button(
            "Ask",
            variant="primary",
        )

    # ---------------- ANSWER ----------------
    # Hidden when page first opens

    answer = gr.Textbox(
        label="💡 Answer",
        lines=6,
        interactive=False,
        visible=False,
    )

    # ---------------- SOURCES ----------------
    # Hidden when page first opens

    sources = gr.Textbox(
        label="📚 Retrieved Sources",
        lines=10,
        max_lines=10,
        interactive=False,
        visible=False,
    )

    # ---------------- EVENTS ----------------

    ask_button.click(
        fn=ask_question,
        inputs=question,
        outputs=[answer, sources],
    )

    question.submit(
        fn=ask_question,
        inputs=question,
        outputs=[answer, sources],
    )


if __name__ == "__main__":
    demo.launch()
