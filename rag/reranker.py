"""Minimal reranker. Torch-free: re-scores hybrid candidates by cosine similarity
between the query and each chunk, reusing the SAME OpenAI embedder as dense retrieval
(text-embedding-3-small via the API — no local model). Works identically local and on the
512 MB hosted tier, since nothing loads into the process.

This is a bi-encoder rerank (lighter than a cross-encoder) — it tightens ordering after
RRF fusion without pulling any model into the runtime; it costs one batched embedding-API
call per rerank. Gated by ENABLE_RERANKER so it can still be switched off.
"""
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.ingestion import DocChunk

_enabled = None


def _ready() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = bool(config.ENABLE_RERANKER)
    return _enabled


def rerank(query: str, chunks: List[DocChunk], top_k: int = None) -> List[tuple[DocChunk, float]]:
    """Return [(chunk, cosine_score)] sorted desc, truncated to top_k."""
    top_k = top_k or config.RERANK_TOPK
    if not _ready() or not chunks:
        return [(c, 0.0) for c in chunks[:top_k]]

    import numpy as np
    from rag.retriever import _embed_fn  # shared OpenAI embedder singleton — no extra model load

    ef = _embed_fn()
    vecs = np.asarray(ef([query] + [c.text for c in chunks]), dtype=np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)  # L2 normalize → dot = cosine
    scores = vecs[1:] @ vecs[0]

    ranked = sorted(zip(chunks, scores), key=lambda x: -float(x[1]))
    return [(c, float(s)) for c, s in ranked[:top_k]]
