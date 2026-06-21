"""Hybrid retrieval: BM25 (lexical) + Chroma dense (semantic), fused by RRF.

Singleton retriever built once at startup. Chroma rebuilt in-memory if no persisted dir.
"""
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.ingestion import DocChunk, load_chunks

_embedder = None  # lazy SentenceTransformer
_retriever = None  # singleton HybridRetriever


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def _chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def build_chroma(chunks: List[DocChunk]) -> int:
    """Embed chunks, write to persistent Chroma. Returns count."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = _chroma_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    coll = client.create_collection(config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    embedder = get_embedder()
    texts = [c.text for c in chunks]
    embs = embedder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    coll.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=[e.tolist() for e in embs],
        documents=texts,
        metadatas=[
            {"doc_id": c.doc_id, "title": c.title, "page_start": c.page_start, "page_end": c.page_end}
            for c in chunks
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
        try:
            client = _chroma_client()
            coll = client.get_collection(config.COLLECTION_NAME)
            if coll.count() == len(chunks):
                return coll
        except Exception:
            pass
        # Rebuild (in-memory persistent dir) — happens on fresh deploy.
        build_chroma(chunks)
        return _chroma_client().get_collection(config.COLLECTION_NAME)

    def bm25_search(self, query: str, n: int) -> List[tuple[str, float]]:
        scores = self.bm25.get_scores(self._tok(query))
        ranked = sorted(zip(range(len(scores)), scores), key=lambda x: -x[1])[:n]
        return [(self.chunks[i].chunk_id, float(s)) for i, s in ranked if s > 0]

    def dense_search(self, query: str, n: int) -> List[tuple[str, float]]:
        emb = get_embedder().encode([query], normalize_embeddings=True)[0]
        res = self.coll.query(query_embeddings=[emb.tolist()], n_results=n)
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
