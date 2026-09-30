
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

sys.path.insert(0, str(PROJECT_ROOT / "app"))
from retrieval import embed_question  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed the evaluation questions once and save the vectors."
    )
    parser.add_argument("--questions", default=str(DATA_DIR / "questions.json"))
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    vectors = np.vstack([embed_question(q["question"]) for q in questions])

    name = Path(args.questions).stem
    suffix = "" if name == "questions" else f"_{name}"
    np.save(DATA_DIR / f"question_vectors{suffix}.npy", vectors)
    (DATA_DIR / f"question_vectors{suffix}.json").write_text(
        json.dumps([q["question"] for q in questions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(questions)} vectors to data/question_vectors{suffix}.npy")


if __name__ == "__main__":
    main()
