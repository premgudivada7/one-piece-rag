import json
import re
import threading
import time

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app import config
from app.chunking import page_label
from app.llm import embed_texts


def embedding_text(chunk: dict) -> str:
    """What gets embedded and reranked: the chunk with its file name and page in front.
    File names carry meaning ("nist_ai_rmf"), which helps "which document" questions."""
    return f"{chunk['source']} ({page_label(chunk)})\n{chunk['text']}"


def tokenize(text: str) -> list[str]:
    """Lowercase tokens for BM25. Codes like '12.3' stay whole and are also split, and a
    capitalised word followed by a number adds a joined token: 'Article 9' -> 'article_9',
    'GOVERN 1.1' -> 'govern_1.1'. A bare '9' is everywhere; 'article_9' is not."""
    tokens, previous = [], ""
    for token in re.findall(r"\w+(?:[.\-/]\w+)*", text):
        lower = token.lower()
        tokens.append(lower)
        if re.search(r"[.\-/]", lower):
            tokens.extend(re.findall(r"\w+", lower))
        if previous[:1].isupper() and previous.isalpha() and token[:1].isdigit():
            tokens.append(f"{previous.lower()}_{lower}")
        previous = token
    return tokens


_TITLED = re.compile(r"^(Article \d+[a-z]?|CHAPTER [IVXLC]+|ANNEX [IVXLC]+|SECTION \d+)$")
_OWN_LINE = re.compile(r"^(Part \d+:.*|Appendix [A-Z]:.*|\d+(?:\.\d+)*\.?\s+[A-Z][^.:;]{2,70}|Whereas:?)$")


def _heading(lines: list[str], i: int) -> str | None:
    line = lines[i]
    if _TITLED.match(line):
        title = lines[i + 1].strip(" `") if i + 1 < len(lines) else ""
        name = line.title() if line.isupper() else line
        if title and len(title) <= 90 and not title.endswith((".", ":", ";")):
            return f"{name} · {title.capitalize() if title.isupper() else title}"
        return name
    # Lines ending with a number are table-of-contents entries, not headings.
    if _OWN_LINE.match(line) and not re.search(r"\d$", line):
        return "Recitals" if line.startswith("Whereas") else line
    return None


def detect_sections(chunks: list[dict]) -> None:
    """Label each chunk with the section in force at its middle (shown on the source cards)."""
    current: dict[str, str | None] = {}
    for chunk in chunks:
        section = current.get(chunk["source"])
        lines = [line.strip() for line in chunk["text"].split("\n") if line.strip()]
        half, seen, at_middle = len(chunk["text"]) / 2, 0, None
        for i, line in enumerate(lines):
            section = _heading(lines, i) or section
            seen += len(line) + 1
            if at_middle is None and seen >= half:
                at_middle = section
        chunk["section"] = at_middle if at_middle is not None else section
        current[chunk["source"]] = section


def build_and_save(chunks: list[dict], meta: dict) -> faiss.Index:
    """Embed every chunk once, build the FAISS index and save it with the chunks."""
    vectors = embed_texts([embedding_text(c) for c in chunks], operation="embed_index")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    config.CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    meta = {**meta, "n_chunks": len(chunks), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embed_model": config.EMBED_MODEL}
    config.INDEX_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return index


def injection_chunk(chunk_id: int) -> dict:
    text = config.INJECTION_TEXT
    return {"id": chunk_id, "source": config.INJECTION_SOURCE, "page": 1, "pages": [1],
            "text": text, "char_count": len(text), "injected": True}


class CorpusIndex:
    """Everything retrieval needs, loaded once."""

    def __init__(self):
        if not config.INDEX_PATH.exists() or not config.CHUNKS_PATH.exists():
            raise FileNotFoundError("No index found in data/. Run: python scripts/build_index.py "
                                    "(or copy my_index.faiss + chunks.json from Teams into data/).")
        self.faiss = faiss.read_index(str(config.INDEX_PATH))
        self.chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
        for chunk in self.chunks:
            chunk.setdefault("pages", [chunk["page"]])
        if self.faiss.ntotal != len(self.chunks):
            raise ValueError(f"Index has {self.faiss.ntotal} vectors but chunks.json has "
                             f"{len(self.chunks)} chunks. Rebuild with --force.")
        self.meta = (json.loads(config.INDEX_META_PATH.read_text(encoding="utf-8"))
                     if config.INDEX_META_PATH.exists() else {})
        built_with = self.meta.get("embed_model")
        if built_with and built_with != config.EMBED_MODEL:
            raise ValueError(f"The index was built with {built_with} but .env now uses {config.EMBED_MODEL}. "
                             "Rebuild it: python scripts/build_index.py --force")
        self.source_names = sorted({c["source"] for c in self.chunks})
        if config.INJECTION_TEST:
            self._add_injection_chunk()
        detect_sections(self.chunks)
        self.vectors = self.faiss.reconstruct_n(0, self.faiss.ntotal)
        tokens = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokens)
        self.token_sets = [set(t) for t in tokens]
        self.pages = self._load_pages()

    def _add_injection_chunk(self):
        """In memory only: the saved index and chunks.json never contain the test passage."""
        chunk = injection_chunk(len(self.chunks))
        self.faiss.add(embed_texts([embedding_text(chunk)], operation="embed_injection"))
        self.chunks.append(chunk)
        print(f"⚠️  INJECTION_TEST is on: added the test passage {config.INJECTION_SOURCE} to the index")

    def _load_pages(self) -> dict:
        """(source, page) -> full page text, for quote_exact and get_article."""
        if not config.CORPUS_PATH.exists():
            return {}
        corpus = json.loads(config.CORPUS_PATH.read_text(encoding="utf-8"))
        return {(e["source"], int(e["page"])): e["text"] for e in corpus}

    @property
    def sources(self) -> dict:
        """Document name -> sorted page numbers."""
        out: dict[str, set] = {}
        keys = self.pages.keys() if self.pages else [(c["source"], p) for c in self.chunks for p in c["pages"]]
        for source, page in keys:
            out.setdefault(source, set()).add(page)
        return {s: sorted(p) for s, p in sorted(out.items())}

    def cosine(self, query_vec: np.ndarray, chunk_ids) -> np.ndarray:
        return self.vectors[list(chunk_ids)] @ query_vec


_index = None
_index_lock = threading.Lock()


def get_index() -> CorpusIndex:
    global _index
    with _index_lock:
        if _index is None:
            _index = CorpusIndex()
    return _index
