"""Chunks data/corpus.json, embeds every chunk once and saves the FAISS index.

    python scripts/build_index.py --dry-run     chunk statistics only
    python scripts/build_index.py               build data/my_index.faiss, chunks.json, index_meta.json
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.chunking import chunk_corpus
from app.index import build_and_save, embedding_text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="only chunk and estimate the cost")
    parser.add_argument("--force", action="store_true", help="overwrite an existing index")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    if not config.CORPUS_PATH.exists():
        sys.exit("data/corpus.json not found. Run scripts/extract.py first (or copy it from Teams).")
    raw = config.CORPUS_PATH.read_bytes()
    corpus = json.loads(raw.decode("utf-8"))
    chunks = chunk_corpus(corpus, args.chunk_size, args.overlap)
    sizes = [c["char_count"] for c in chunks]
    print(f"corpus: {len(corpus)} entries from {len({e['source'] for e in corpus})} files")
    print(f"chunks: {len(chunks)} (size {args.chunk_size}, overlap {args.overlap}) | "
          f"chars min {min(sizes)} / median {int(statistics.median(sizes))} / max {max(sizes)}")
    print(f"chunks spanning a page break: {sum(len(c['pages']) > 1 for c in chunks)}")

    if config.EMBED_PROVIDER == "local":
        print(f"embedding model: {config.EMBED_MODEL} (local, free)")
    else:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = sum(len(encoder.encode(embedding_text(c))) for c in chunks)
        print(f"embedding tokens: {tokens:,} -> estimated cost ${tokens * config.PRICE_EMBED:.4f}")

    if args.dry_run:
        return
    if config.INDEX_PATH.exists() and not args.force:
        sys.exit("An index already exists in data/. Use --force to rebuild it (this re-embeds everything).")

    meta = {"chunk_size": args.chunk_size, "overlap": args.overlap, "embed_provider": config.EMBED_PROVIDER,
            "corpus_sha1": hashlib.sha1(raw).hexdigest()}
    build_and_save(chunks, meta)
    print(f"saved {config.INDEX_PATH.name}, {config.CHUNKS_PATH.name}, {config.INDEX_META_PATH.name} in data/")


if __name__ == "__main__":
    main()
