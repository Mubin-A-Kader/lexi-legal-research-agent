"""Cross-encoder reranker. Boosts precision after hybrid recall.

Falls back to lighter TinyBERT if primary OOMs (512MB Render). Degrades to identity if both fail.
"""
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.ingestion import DocChunk

_reranker = None
_FALLBACK = "cross-encoder/ms-marco-TinyBERT-L-2-v2"


def _load():
    global _reranker
    if _reranker is not None:
        return _reranker
    if not config.ENABLE_RERANKER:
        _reranker = "identity"  # disabled (e.g. low-memory hosted tier) — skip torch entirely
        return _reranker
    from sentence_transformers import CrossEncoder
    for name in (config.RERANK_MODEL, _FALLBACK):
        try:
            _reranker = CrossEncoder(name, max_length=512)
            return _reranker
        except Exception as e:  # pragma: no cover
            print(f"[reranker] failed to load {name}: {e}")
    _reranker = "identity"  # sentinel: skip reranking
    return _reranker


def rerank(query: str, chunks: List[DocChunk], top_k: int = None) -> List[tuple[DocChunk, float]]:
    """Return [(chunk, score)] sorted desc, truncated to top_k."""
    top_k = top_k or config.RERANK_TOPK
    model = _load()
    if model == "identity" or not chunks:
        return [(c, 0.0) for c in chunks[:top_k]]
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: -float(x[1]))
    return [(c, float(s)) for c, s in ranked[:top_k]]
