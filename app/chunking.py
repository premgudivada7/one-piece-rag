import re

from app import config

_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")


def _split_long(text: str, max_len: int) -> list[str]:
    """Split a unit longer than max_len: first by sentences, then by words."""
    pieces = []
    for sentence in _SENTENCE_BREAK.split(text):
        while len(sentence) > max_len:
            cut = sentence.rfind(" ", 0, max_len)
            cut = cut if cut > max_len // 2 else max_len
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def split_units(text: str, max_len: int) -> list[str]:
    """Units we never cut inside: lines (paragraphs, list items, table rows), or sentences
    when a paragraph is too long."""
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            units.extend([line] if len(line) <= max_len else _split_long(line, max_len))
    return units


def _tail(units: list[tuple[int, str]], overlap: int) -> list[tuple[int, str]]:
    """The last units of a chunk (at most `overlap` characters), repeated at the start of the next one."""
    tail, size = [], 0
    for page, unit in reversed(units):
        if size + len(unit) + 1 > overlap:
            if not tail:  # always carry something, so a fact on the boundary is in both chunks
                cut = unit.find(" ", len(unit) - overlap)
                tail.insert(0, (page, unit[cut + 1:] if cut != -1 else unit[-overlap:]))
            break
        tail.insert(0, (page, unit))
        size += len(unit) + 1
    return tail


def _make_chunk(units: list[tuple[int, str]], source: str) -> dict:
    pages = sorted({page for page, _ in units})
    text = "\n".join(unit for _, unit in units)
    return {"text": text, "source": source, "page": pages[0], "pages": pages, "char_count": len(text)}


def chunk_document(source: str, pages: list[tuple[int, str]],
                   chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP) -> list[dict]:
    """Chunk one document given as [(page_number, page_text), ...] in page order."""
    units = [(page, unit) for page, text in pages for unit in split_units(text, chunk_size)]
    chunks, current, size = [], [], 0
    for page, unit in units:
        if current and size + len(unit) + 1 > chunk_size:
            chunks.append(_make_chunk(current, source))
            current = _tail(current, overlap)
            size = sum(len(u) + 1 for _, u in current)
        current.append((page, unit))
        size += len(unit) + 1
    if current:
        chunks.append(_make_chunk(current, source))
    return chunks


def chunk_corpus(corpus: list[dict], chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP) -> list[dict]:
    """Chunk every document of corpus.json; a chunk may run across a page break."""
    documents: dict[str, list[tuple[int, str]]] = {}
    for entry in corpus:
        documents.setdefault(entry["source"], []).append((int(entry["page"]), entry["text"] or ""))
    chunks = []
    for source, pages in documents.items():
        chunks.extend(chunk_document(source, sorted(pages), chunk_size, overlap))
    for i, chunk in enumerate(chunks):
        chunk["id"] = i  # same position as in the FAISS index
    return chunks


def page_label(chunk: dict) -> str:
    """'p.4' or 'p.4-5', as used in citations."""
    pages = chunk.get("pages") or [chunk["page"]]
    return f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"
