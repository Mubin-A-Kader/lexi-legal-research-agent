"""Hybrid retrieval: BM25 (lexical) + Chroma dense (semantic), fused by RRF.

Singleton retriever built once at startup. Chroma rebuilt in-memory if no persisted dir.
"""
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.ingestion import DocChunk, load_chunks

_ef = None        # chroma OpenAI embedding function (text-embedding-3-small, API — no local model)
_retriever = None  # singleton HybridRetriever


def _embed_fn():
    """OpenAI embeddings API (text-embedding-3-small) via Chroma's EF. No model loads into the
    process — no onnxruntime, no torch — so the hosted runtime fits a 512 MB free tier. The
    reranker reuses this same embedder, so query/doc vectors are compatible.

    Chroma's OpenAIEmbeddingFunction defaults to CHROMA_OPENAI_API_KEY, so we pass OPENAI_API_KEY
    explicitly; api_base honours OPENAI_BASE_URL for OpenAI-compatible providers."""
    global _ef
    if _ef is None:
        import os
        from chromadb.utils import embedding_functions
        _ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=config.EMBED_MODEL,
            api_base=config.OPENAI_BASE_URL,
        )
    return _ef


def _chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def build_chroma(chunks: List[DocChunk]) -> int:
    """Index chunks in Chroma. Chroma embeds via the OpenAI EF (API). Returns count."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = _chroma_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    coll = client.create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_embed_fn(),
    )
    # Batch the add: the OpenAI embeddings API caps each request at 300k tokens, and the full
    # corpus exceeds that, so we send in fixed-size slices.
    BATCH = 100
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        coll.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {"doc_id": c.doc_id, "title": c.title, "page_start": c.page_start, "page_end": c.page_end}
                for c in batch
            ],
        )
    return len(chunks)


class HybridRetriever:
    """BM25 + dense, fused via Reciprocal Rank Fusion."""

    def __init__(self, chunks: List[DocChunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self._tokenized = [self._tok(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._tokenized)

        # Dense: connect to Chroma; if missing, build in-memory from chunks.
        self.coll = self._ensure_chroma(chunks)

    @staticmethod
    def _tok(text: str) -> List[str]:
        import re
        return re.findall(r"[a-z0-9]+", text.lower())

    def _ensure_chroma(self, chunks):
        ef = _embed_fn()
        try:
            client = _chroma_client()
            coll = client.get_collection(config.COLLECTION_NAME, embedding_function=ef)
            if coll.count() == len(chunks):
                return coll
        except Exception:
            pass
        # Rebuild (in-memory persistent dir) — happens on fresh deploy.
        build_chroma(chunks)
        return _chroma_client().get_collection(config.COLLECTION_NAME, embedding_function=ef)

    def bm25_search(self, query: str, n: int) -> List[tuple[str, float]]:
        scores = self.bm25.get_scores(self._tok(query))
        ranked = sorted(zip(range(len(scores)), scores), key=lambda x: -x[1])[:n]
        return [(self.chunks[i].chunk_id, float(s)) for i, s in ranked if s > 0]

    def dense_search(self, query: str, n: int) -> List[tuple[str, float]]:
        # Chroma embeds the query via the same OpenAI EF (API call, no local model).
        res = self.coll.query(query_texts=[query], n_results=n)
        ids = res["ids"][0]
        dists = res["distances"][0]
        return [(cid, 1.0 - d) for cid, d in zip(ids, dists)]  # cosine sim

    def hybrid_search(self, query: str, n_each: int = None, k_rrf: int = 60) -> List[DocChunk]:
        """RRF-fuse BM25 + dense, return deduped DocChunks ordered by fused score."""
        n_each = n_each or max(config.BM25_TOPK, config.DENSE_TOPK)
        bm = self.bm25_search(query, n_each)
        dn = self.dense_search(query, n_each)

        rrf: dict[str, float] = {}
        for ranklist in (bm, dn):
            for rank, (cid, _) in enumerate(ranklist):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)

        order = sorted(rrf.items(), key=lambda x: -x[1])
        return [self.by_id[cid] for cid, _ in order if cid in self.by_id]


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(load_chunks())
    return _retriever
