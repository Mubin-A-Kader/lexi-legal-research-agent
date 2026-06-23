"""One-time: parse all PDFs -> extracted_docs.json + chunks.json, then build Chroma index.

Run: python scripts/ingest.py
Idempotent. Commits JSON to repo so deploy startup skips PDF parsing.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag import ingestion


def main(rebuild_chroma: bool = True):
    t0 = time.time()
    print(f"Parsing PDFs from {config.CORPUS_DIR} ...")
    docs, chunks = ingestion.build_corpus()
    print(f"  {len(docs)} docs, {len(chunks)} chunks in {time.time()-t0:.1f}s")

    # Generate contextual summaries for retrieval
    t_sum = time.time()
    print("Generating contextual summaries ...")
    chunks = ingestion.summarize_chunks(chunks)
    print(f"  summaries in {time.time()-t_sum:.1f}s")

    ingestion.save_extracted(docs, chunks)
    print(f"  wrote {config.EXTRACTED_JSON.name}, {config.CHUNKS_JSON.name}")

    if rebuild_chroma:
        from rag.retriever import build_chroma
        t1 = time.time()
        print("Building Chroma dense index ...")
        n = build_chroma(chunks)
        print(f"  embedded {n} chunks in {time.time()-t1:.1f}s")

    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main(rebuild_chroma="--no-chroma" not in sys.argv)
