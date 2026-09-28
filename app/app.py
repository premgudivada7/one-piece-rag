import gradio as gr


def ask_question(question: str):
    """Temporary function until retrieval and LLM are connected."""
    if not question.strip():
        return "Please enter a question.", ""

    return (
        "Retrieval and LLM will be connected here.",
        "Sources will appear here."
    )


with gr.Blocks(title="AI Regulation RAG Assistant") as demo:

    gr.Markdown("# AI Regulation RAG Assistant")
    gr.Markdown(
        "Ask questions about the EU AI Act and "
        "NIST AI Risk Management Framework."
    )

    question = gr.Textbox(
        label="Your Question",
        placeholder="Example: What are the four core functions of NIST AI RMF?"
    )

    ask_button = gr.Button("Ask")

    answer = gr.Markdown(label="Answer")
    sources = gr.Markdown(label="Sources")

    ask_button.click(
        fn=ask_question,
        inputs=question,
        outputs=[answer, sources],
    )


if __name__ == "__main__":
    demo.launch()