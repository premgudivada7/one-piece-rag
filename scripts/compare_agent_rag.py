"""Agent vs RAG on the same questions (agent questions + priority questions), from two saved runs.

    python scripts/eval_runner.py --set agent --judge       RAG pipeline on the agent question set
    python scripts/eval_runner.py --mode agent --judge      agent on the same set
    python scripts/compare_agent_rag.py                     table -> data/agent_vs_rag.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.evaluation import mean_text, pct, should_refuse


def newest(mode: str) -> Path | None:
    """The newest run of this mode, preferring one made on the agent question set."""
    paths = sorted(config.EVAL_RUNS_DIR.glob(f"*_{mode}.json"), reverse=True)
    on_agent_set = [p for p in paths if json.loads(p.read_text(encoding="utf-8")).get("question_set") == "agent"]
    return (on_agent_set or paths or [None])[0]


def load(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, {r["id"]: r for r in data["results"] if r.get("run", 0) == 0}


def summary(rows: list[dict]) -> dict:
    answered = [r for r in rows if r.get("answer") and not r.get("refused")]
    return {
        "Questions": str(len(rows)),
        "Correct answers": pct([r.get("passed") for r in rows]),
        "Correct refusals": pct([r.get("refused") for r in rows if should_refuse(r)]),
        "Answers citing a page": pct([r.get("has_citation") for r in answered]),
        "Model calls per question": mean_text([r.get("model_calls") for r in rows]),
        "Latency per question (s)": mean_text([r.get("latency_s") for r in rows]),
        "Cost per question ($)": mean_text([r.get("cost") for r in rows], "{:.4f}"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rag", type=Path)
    parser.add_argument("--agent", type=Path)
    args = parser.parse_args()
    rag_path, agent_path = args.rag or newest("rag"), args.agent or newest("agent")
    if not rag_path or not agent_path:
        sys.exit("Need a RAG run and an agent run in data/eval_runs/ (see the commands at the top of this file).")
    rag_data, rag = load(rag_path)
    agent_data, agent = load(agent_path)
    common = [i for i in agent if i in rag]
    if not common:
        sys.exit(f"{rag_path.name} and {agent_path.name} have no question in common: run the RAG eval with --set agent.")
    if not (rag_data.get("judge") and agent_data.get("judge")):
        print("⚠️  One of the runs was made without --judge: correctness is incomplete.")

    rag_summary, agent_summary = summary([rag[i] for i in common]), summary([agent[i] for i in common])
    lines = [f"Agent vs RAG on the same {len(common)} questions (RAG run {rag_data['time']}, agent run "
             f"{agent_data['time']}, {agent_data.get('settings', {}).get('chat_model', '?')})", "",
             "| | RAG pipeline | Agent |", "|---|:---:|:---:|"]
    lines += [f"| {key} | {rag_summary[key]} | {agent_summary[key]} |" for key in rag_summary]
    lines += ["", "| Question | Scenario | RAG | Agent | Calls RAG / agent | Latency RAG / agent (s) |",
              "|---|---|:---:|:---:|:---:|:---:|"]
    icon = {True: "✅", False: "❌", None: "❔"}
    for i in common:
        r, a = rag[i], agent[i]
        scenario = a.get("sub_type") if a.get("category") == "agent" else "priority"
        lines.append(f"| {i}: {a['question'][:60]} | {scenario} | {icon[r.get('passed')]} | {icon[a.get('passed')]} | "
                     f"{r.get('model_calls', '-')} / {a.get('model_calls', '-')} | "
                     f"{r.get('latency_s', '-')} / {a.get('latency_s', '-')} |")
    only_agent = [i for i in common if agent[i].get("passed") and rag[i].get("passed") is False]
    only_rag = [i for i in common if rag[i].get("passed") and agent[i].get("passed") is False]
    lines += ["", f"Agent right, RAG wrong ({len(only_agent)}): {', '.join(only_agent) or '-'}",
              f"RAG right, agent wrong ({len(only_rag)}): {', '.join(only_rag) or '-'}"]
    text = "\n".join(lines)
    print(text)
    out = config.DATA_DIR / "agent_vs_rag.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
