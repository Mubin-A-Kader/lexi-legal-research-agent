"""Agent tools. The agent composes these freely — no hard-coded pipeline.

Tools:
  semantic_search   - dense+lexical hybrid + rerank, returns ranked chunks
  keyword_search    - pure BM25 lexical (exact terms, citations)
  get_document      - full text of one judgment
  analyze_precedent - LLM classifies one judgment vs a case position (supporting/adverse/neutral)
  list_documents    - corpus inventory (doc_id + title)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool

import config
from rag.retriever import get_retriever
from rag.reranker import rerank
from rag.ingestion import load_docs
from agent.prompts import ANALYZE_PRECEDENT_PROMPT

_docs_cache = None


def _docs():
    global _docs_cache
    if _docs_cache is None:
        _docs_cache = load_docs()
    return _docs_cache


def _fmt_hits(hits) -> str:
    out = []
    for c, score in hits:
        out.append(
            {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "title": c.title,
                "pages": f"{c.page_start}-{c.page_end}",
                "rerank_score": round(score, 3),
                "snippet": c.text[:500],
            }
        )
    return json.dumps(out, ensure_ascii=False, indent=2)


@tool
def semantic_search(query: str, n: int = 8) -> str:
    """Search the judgment corpus by meaning. Hybrid (semantic + keyword) retrieval with
    cross-encoder reranking. Best for conceptual queries ("contributory negligence",
    "policy void due to unlicensed driver"). Returns top-n ranked chunks with doc_id,
    title, snippet, and relevance score. Use this first for most research."""
    retriever = get_retriever()
    candidates = retriever.hybrid_search(query)[:30]
    reranked = rerank(query, candidates, top_k=n)
    return _fmt_hits(reranked)


@tool
def keyword_search(query: str, n: int = 8) -> str:
    """Exact-term lexical (BM25) search. Best when you need precise tokens: a case name,
    a statute/section number, or a specific phrase. Returns top-n chunks with doc_id and snippet."""
    retriever = get_retriever()
    hits = retriever.bm25_search(query, n)
    chunks = [(retriever.by_id[cid], score) for cid, score in hits]
    return _fmt_hits(chunks)


@tool
def get_document(doc_id: str) -> str:
    """Return the full text of one judgment by its doc_id (e.g. "doc_017"). Use to read a
    precedent in full before relying on it or distinguishing it. Truncated to ~12k chars."""
    d = _docs().get(doc_id)
    if not d:
        return f"No document with doc_id={doc_id}. Use list_documents to see valid ids."
    text = d["full_text"]
    if len(text) > 12000:
        text = text[:12000] + "\n...[truncated]"
    return f"# {d['title']} ({doc_id})\n\n{text}"


@tool
def analyze_precedent(doc_id: str, case_context: str) -> str:
    """Classify ONE judgment as supporting / adverse / neutral for a given client position.
    Pass the doc_id and a short description of the client's case/argument (case_context).
    Returns JSON with stance, the legal principle, specific reasoning, and risk/strength.
    Use this on the most relevant judgments during deep precedent research."""
    d = _docs().get(doc_id)
    if not d:
        return f"No document with doc_id={doc_id}."
    # Use reranked excerpts of this doc against the case_context to keep prompt tight.
    retriever = get_retriever()
    doc_chunks = [c for c in retriever.chunks if c.doc_id == doc_id]
    ranked = rerank(case_context, doc_chunks, top_k=4)
    excerpt = "\n\n---\n\n".join(c.text for c, _ in ranked) or d["full_text"][:6000]

    from agent.llm import complete

    prompt = ANALYZE_PRECEDENT_PROMPT.format(
        case_context=case_context, doc_id=doc_id, doc_text=excerpt
    )
    text = complete(prompt, model=config.EVAL_MODEL, max_tokens=1024)
    # Pass through; the agent reads the JSON. Prefix with doc_id for traceability.
    return f"[analyze_precedent doc_id={doc_id}]\n{text}"


@tool
def list_documents() -> str:
    """List every judgment in the corpus as doc_id + title. Use to get an inventory or when
    the user asks how many / which documents exist."""
    items = [{"doc_id": k, "title": v["title"]} for k, v in sorted(_docs().items())]
    return json.dumps(items, ensure_ascii=False, indent=2)


ALL_TOOLS = [semantic_search, keyword_search, get_document, analyze_precedent, list_documents]
