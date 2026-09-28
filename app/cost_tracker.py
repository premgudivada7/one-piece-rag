import json
from pathlib import Path
from threading import Lock
 
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COST_FILE = PROJECT_ROOT / "data" / "cost_tracker.json"
 
# Estimated USD per one million tokens from the course reference.
EMBEDDING_RATE = 0.02
CHAT_INPUT_RATE = 0.15
CHAT_OUTPUT_RATE = 0.60
 
FILE_LOCK = Lock()
 
 
def load_total() -> float:
    """Read the estimated spending recorded on this computer."""
    if not COST_FILE.exists():
        return 0.0
 
    data = json.loads(COST_FILE.read_text(encoding="utf-8"))
    return float(data["total"])
 
 
def track_cost(response, is_embedding: bool = False) -> float:
    """Calculate and save the estimated cost of one API response."""
    usage = response.usage
 
    if usage is None:
        raise ValueError("The API response did not include token usage.")
 
    if is_embedding:
        cost = usage.total_tokens * EMBEDDING_RATE / 1_000_000
    else:
        cost = (
            usage.prompt_tokens * CHAT_INPUT_RATE
            + usage.completion_tokens * CHAT_OUTPUT_RATE
        ) / 1_000_000
 
    with FILE_LOCK:
        total = load_total() + cost
 
        COST_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = COST_FILE.with_suffix(".tmp")
 
        temporary_file.write_text(
            json.dumps({"total": total}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_file.replace(COST_FILE)
 
    print(f"Estimated cost of this call: ${cost:.6f}")
    print(f"Estimated spending recorded locally: ${total:.4f}")
 
    return cost
 
 
if __name__ == "__main__":
    print(f"Estimated spending recorded locally: ${load_total():.4f}")