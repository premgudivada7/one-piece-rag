"""Prints saved agent runs step by step.

    python scripts/view_trace.py --last 3
    python scripts/view_trace.py data/traces/<id>.json --full
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.agent import format_trace


def show(path: Path, full: bool):
    run = json.loads(path.read_text(encoding="utf-8"))
    print("=" * 78)
    print(f"{path.name}\nQ: {run['question']}")
    print(format_trace(run["trace"], max_result=100_000 if full else 250))
    print(f"A: {run['answer']}")
    print(f"steps {run['steps']} · cap hit {run['hit_step_cap']} · tools {run['tools_used']} · "
          f"cost ${run['cost']:.5f} · {run['duration_s']}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--last", type=int, default=1)
    parser.add_argument("--full", action="store_true", help="print full tool results")
    args = parser.parse_args()
    files = args.files or sorted(config.TRACES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)[-args.last:]
    if not files:
        sys.exit("No traces yet: ask the agent something first.")
    for path in files:
        show(path, args.full)


if __name__ == "__main__":
    main()
