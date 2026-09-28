import os

from dotenv import load_dotenv
from openai import OpenAI


# Load OpenAI API key from .env / environment
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an AI Regulation RAG assistant.

Answer the user's question using ONLY the information provided
in the retrieved context.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain enough information to answer,
   say exactly:
   "I don't have enough information in the provided documents to answer this question."
3. Every factual claim must have a citation.
4. Citations must refer only to the sources and page numbers provided
   in the context.
5. Do not invent sources, page numbers, quotations, or citations.
6. Keep the answer clear and concise.
7. If multiple retrieved chunks support an answer, cite all relevant chunks.

Use citations in this format:

[1]
[2]

where the number corresponds to the retrieved context.
"""


def build_context(retrieved_chunks):
    """
    Convert retrieved chunks from retrieval.py into context
    that can be given to the LLM.
    """

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "")
        source = chunk.get("source", "Unknown source")
        page = chunk.get("page", "Unknown page")
        score = chunk.get("score", None)

        part = f"""
[{i}]
Source: {source}
Page: {page}
"""

        if score is not None:
            part += f"Similarity score: {score}\n"

        part += f"Text:\n{text}\n"

        context_parts.append(part)

    return "\n".join(context_parts)


def generate_answer(question, retrieved_chunks):
    """
    Generate an answer using GPT-4o-mini from retrieved RAG chunks.

    Parameters
    ----------
    question : str
        User's question.

    retrieved_chunks : list
        Top retrieved chunks returned by retrieval.py.

    Returns
    -------
    str
        Final answer with citations.
    """

    if not retrieved_chunks:
        return (
            "I don't have enough information in the provided "
            "documents to answer this question."
        )

    context = build_context(retrieved_chunks)

    user_prompt = f"""
Question:
{question}

Retrieved context:
{context}

Answer the question using only the retrieved context.

Remember:
- Cite factual statements using [1], [2], etc.
- Do not invent citations.
- If the context is insufficient, use the required
  "I don't have enough information..." response.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.choices[0].message.content.strip()
