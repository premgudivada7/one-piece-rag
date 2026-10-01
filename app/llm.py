import json
import os
import threading
import time
from functools import lru_cache

import numpy as np
from openai import OpenAI

from app import config

_clients = {}
_local_model = None
_cost_lock = threading.Lock()  # Gradio answers in several threads


class BudgetExceeded(RuntimeError):
    """Raised before a call once the tracked spend has reached BUDGET_USD."""


def _require(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is missing in .env (see .env.example).")
    return value


def _client(kind: str) -> OpenAI:
    """One client per kind ("chat" or "embed"), created on first use."""
    if kind not in _clients:
        if kind == "chat" and config.LLM_PROVIDER == "xai":
            _clients[kind] = OpenAI(api_key=_require("XAI_API_KEY"), base_url=config.XAI_BASE_URL,
                                    timeout=config.OPENAI_TIMEOUT, max_retries=3)
        else:
            _clients[kind] = OpenAI(api_key=_require("OPENAI_API_KEY"), timeout=config.OPENAI_TIMEOUT,
                                    max_retries=3)
    return _clients[kind]


def cost_of(response, is_embedding=False) -> float:
    """Dollar cost of one API response, from the token counts it reports."""
    usage = response.usage
    if usage is None:
        return 0.0
    if is_embedding:
        return usage.total_tokens * config.PRICE_EMBED
    return usage.prompt_tokens * config.PRICE_CHAT_INPUT + usage.completion_tokens * config.PRICE_CHAT_OUTPUT


def load_costs() -> dict:
    if config.COST_PATH.exists():
        return json.loads(config.COST_PATH.read_text(encoding="utf-8"))
    return {"total": 0.0, "calls": 0, "by_operation": {}}


def track_cost(response, operation: str, is_embedding=False) -> float:
    """Add the call to data/cost_tracker.json (totals) and data/cost_log.jsonl (one line per call)."""
    cost = cost_of(response, is_embedding)
    usage = response.usage
    with _cost_lock:
        data = load_costs()
        data["total"] += cost
        data["calls"] += 1
        data["by_operation"][operation] = data["by_operation"].get(operation, 0.0) + cost
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.COST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with config.COST_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "operation": operation,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "cost": round(cost, 8),
            }) + "\n")
    if data["total"] > 0.8 * config.BUDGET_USD:
        print(f"⚠️  Spend ${data['total']:.4f} is above 80% of the ${config.BUDGET_USD:.2f} budget")
    return cost


def check_budget():
    total = load_costs()["total"]
    if total >= config.BUDGET_USD:
        raise BudgetExceeded(f"Tracked spend ${total:.4f} reached BUDGET_USD=${config.BUDGET_USD:.2f}. "
                             "Raise BUDGET_USD in .env only if the team agrees.")


def chat(messages, operation: str, tools=None, tool_choice=None, max_tokens=400, response_format=None):
    """One chat completion at temperature 0, with a token cap and cost tracking."""
    check_budget()
    kwargs = {"model": config.CHAT_MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"
    if response_format:
        kwargs["response_format"] = response_format
    response = _client("chat").chat.completions.create(**kwargs)
    track_cost(response, operation)
    return response


def ask(messages, operation: str, max_tokens=400) -> tuple[str, float]:
    """Plain text answer of one chat call, and its cost."""
    response = chat(messages, operation=operation, max_tokens=max_tokens)
    return (response.choices[0].message.content or "").strip(), cost_of(response)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _embed_local(texts: list[str]) -> np.ndarray:
    """Embed on this machine (the model downloads once, about 130 MB)."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"loading local embedding model {config.LOCAL_EMBED_MODEL} ...")
        _local_model = SentenceTransformer(config.LOCAL_EMBED_MODEL)
    vectors = _local_model.encode(texts, batch_size=32, normalize_embeddings=True,
                                  show_progress_bar=len(texts) > 100)
    return np.asarray(vectors, dtype="float32")


def embed_texts(texts, operation="embed", batch_size=100) -> np.ndarray:
    """Embed a list of texts into an L2-normalized float32 matrix (inner product = cosine)."""
    if config.EMBED_PROVIDER == "local":
        return _embed_local(list(texts))
    vectors = []
    for start in range(0, len(texts), batch_size):
        check_budget()
        response = _client("embed").embeddings.create(model=config.EMBED_MODEL, input=texts[start:start + batch_size])
        track_cost(response, operation, is_embedding=True)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return _normalize(np.array(vectors, dtype="float32"))


@lru_cache(maxsize=4096)
def _embed_query_cached(text: str) -> tuple:
    return tuple(embed_texts([config.QUERY_PREFIX + text], operation="embed_query")[0].tolist())


def embed_query(text: str) -> np.ndarray:
    """Embed one question; repeated questions come from memory."""
    return np.array(_embed_query_cached(text.strip()), dtype="float32")
