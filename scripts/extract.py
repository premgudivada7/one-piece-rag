
import json
import re
from pathlib import Path
from typing import TypedDict

import pymupdf


# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "sources"

MIN_CHARACTERS = 50
SAMPLE_SIZE = 10

# Header/footer patterns for the two selected PDFs.
MARGIN_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"EN",
        r"OJ L,.*",
        r"ELI:\s*https?://.*",
        r"\d+/\d+",
        r"NIST AI 100-1",
        r"AI RMF 1\.0",
        r"Page \d+",
        r"[ivxlcdm]+",
    )
)

WORD_PATTERN = re.compile(
    r"(?<![\w-])[A-Za-z]+(?:-[A-Za-z]+)*(?![\w-])"
)
SPLIT_WORD_PATTERN = re.compile(
    r"\b([A-Za-z]+)-\s+([A-Za-z]+)\b"
)


# ---------- Data structure ----------

class CorpusEntry(TypedDict):
    """Required metadata and cleaned text for one PDF page."""

    source: str
    page: int
    char_count: int
    text: str


# ---------- Source discovery ----------

def find_pdf_sources(source_dir: Path) -> list[Path]:
    """Find PDFs without silently ignoring unsupported documents."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source folder: {source_dir}")

    files = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )

    if not files:
        raise ValueError("No documents found in data/sources/.")

    unsupported = [
        path.name for path in files
        if path.suffix.lower() != ".pdf"
    ]

    if unsupported:
        raise ValueError(
            "This script supports PDFs only: "
            + ", ".join(unsupported)
        )

    return files


# ---------- Text cleaning ----------

def build_vocabulary(sources: list[Path]) -> set[str]:
    """Collect complete word spellings directly from the source PDFs."""
    vocabulary: set[str] = set()
    flags = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_DEHYPHENATE

    for source in sources:
        with pymupdf.open(source) as document:
            for page in document:
                text = page.get_text("text", flags=flags).lower()
                vocabulary.update(WORD_PATTERN.findall(text))

    return vocabulary


def repair_split_words(text: str, vocabulary: set[str]) -> str:
    """Repair only splits supported by spellings in the source corpus."""

    def replace_match(match: re.Match[str]) -> str:
        first, second = match.group(1, 2)
        joined = first + second
        hyphenated = first + "-" + second

        if joined.lower() in vocabulary:
            return joined

        if hyphenated.lower() in vocabulary:
            return hyphenated

        # Leave uncertain cases unchanged instead of guessing.
        return match.group(0)

    return SPLIT_WORD_PATTERN.sub(replace_match, text)


def clean_block(raw_text: str, in_margin: bool) -> str:
    """Remove extra whitespace and known margin labels."""
    lines = []

    for raw_line in raw_text.splitlines():
        line = " ".join(raw_line.replace("\u00ad", "").split())

        if not line:
            continue

        is_margin_label = any(
            pattern.fullmatch(line)
            for pattern in MARGIN_PATTERNS
        )

        if in_margin and is_margin_label:
            continue

        lines.append(line)

    return " ".join(lines)


# ---------- PDF extraction ----------

def extract_page_text(
    page: pymupdf.Page,
    vocabulary: set[str],
) -> str:
    """Extract text blocks and conservatively repair split words."""
    # Disable automatic dehyphenation so meaningful hyphens survive.
    flags = (
        pymupdf.TEXTFLAGS_BLOCKS
        & ~pymupdf.TEXT_DEHYPHENATE
        & ~pymupdf.TEXT_PRESERVE_IMAGES
    )

    blocks = page.get_text("blocks", sort=True, flags=flags)
    paragraphs = []

    for block in blocks:
        _, top, _, bottom, raw_text, _, block_type = block[:7]

        if block_type != 0:
            continue

        # Thresholds checked against the selected source documents.
        in_margin = bottom < 65 or top > page.rect.height * 0.87
        paragraph = clean_block(raw_text, in_margin)

        if paragraph:
            paragraphs.append(paragraph)

    text = "\n\n".join(paragraphs)
    return repair_split_words(text, vocabulary)


def extract_pdf(
    path: Path,
    vocabulary: set[str],
) -> list[CorpusEntry]:
    """Keep one entry for every PDF page, including empty pages."""
    entries: list[CorpusEntry] = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = extract_page_text(page, vocabulary)

            entries.append(
                {
                    "source": path.name,
                    "page": page_number,
                    "char_count": len(text),
                    "text": text,
                }
            )

    return entries


# ---------- Output ----------

def save_json(entries: list[CorpusEntry], output_path: Path) -> None:
    """Save readable UTF-8 JSON without escaping accented characters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_statistics(
    entries: list[CorpusEntry],
    source_count: int,
) -> None:
    """Print the five required statistics and near-empty page details."""
    total_entries = len(entries)
    total_characters = sum(entry["char_count"] for entry in entries)
    average = total_characters / total_entries if total_entries else 0

    near_empty = [
        entry for entry in entries
        if entry["char_count"] < MIN_CHARACTERS
    ]

    print(f"source files: {source_count}")
    print(f"total entries: {total_entries}")
    print(f"total characters: {total_characters}")
    print(f"avg characters/entry: {average:.2f}")
    print(
        f"near-empty entries (<{MIN_CHARACTERS} chars): "
        f"{len(near_empty)}"
    )

    for entry in near_empty:
        print(
            f"  Review: {entry['source']}, page {entry['page']} "
            f"({entry['char_count']} characters)"
        )


# ---------- Main workflow ----------

def main() -> None:
    """Discover sources, clean text, save JSON, and report statistics."""
    sources = find_pdf_sources(SOURCE_DIR)
    vocabulary = build_vocabulary(sources)
    corpus: list[CorpusEntry] = []

    for source in sources:
        corpus.extend(extract_pdf(source, vocabulary))

    save_json(corpus, DATA_DIR / "corpus.json")
    save_json(corpus[:SAMPLE_SIZE], DATA_DIR / "sample.json")
    print_statistics(corpus, source_count=len(sources))


if __name__ == "__main__":
    main()