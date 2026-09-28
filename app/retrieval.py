import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CHUNKS_FILE = DATA_DIR / "chunks.json"
INDEX_FILE = DATA_DIR / "my_index.faiss"

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
TOP_K = 15

# ---------- Load data ----------

def load_data():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(str(INDEX_FILE))

    return chunks, index

# ---------- Create question embedding ----------

def embed_question(question):
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        model=MODEL,
        input=question,
        dimensions=DIMENSIONS
    )

    vector = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    return vector

# ---------- Retrieve top chunks ----------

def retrieve(question, k=TOP_K):
    chunks, index = load_data()

    question_vector = embed_question(question)

    scores, indices = index.search(question_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": float(score)
        })

    return results

# ---------- Test retrieval ----------

if __name__ == "__main__":
    question = "What is a high-risk AI system?"

    results = retrieve(question)

    print(f"\nQuestion: {question}\n")

    for i, result in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']}")
        print(f"Text: {result['text'][:500]}")
        print()
