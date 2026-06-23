"""Shared eval helpers: run the agent, extract doc_ids it surfaced / relied on."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

GOLD = json.loads((Path(__file__).parent / "gold_standard.json").read_text())
DOC_RE = re.compile(r"doc_\d{3}")


def run_agent(query: str) -> dict:
    """Run the full agent. Returns {output, retrieved_ids, cited_ids, intermediate_steps}.

    retrieved_ids = every doc_id that appeared in any tool observation (what the agent looked at).
    cited_ids     = doc_ids appearing in the FINAL answer (what the agent actually presented).
    """
    from agent.agent import build_agent

    executor = build_agent()
    result = executor.invoke({"input": query})
    output = result["output"]
    if isinstance(output, list):
        output = " ".join(b.get("text", "") for b in output if isinstance(b, dict))

    retrieved, steps_dump = set(), []
    for action, observation in result.get("intermediate_steps", []):
        obs = str(observation)
        retrieved.update(DOC_RE.findall(obs))
        steps_dump.append({"tool": getattr(action, "tool", "?"),
                           "input": getattr(action, "tool_input", ""),
                           "obs_doc_ids": sorted(set(DOC_RE.findall(obs)))})

    cited = sorted(set(DOC_RE.findall(output)))
    return {
        "output": output,
        "retrieved_ids": sorted(retrieved),
        "cited_ids": cited,
        "intermediate_steps": steps_dump,
    }


def prf(predicted: set, gold: set) -> dict:
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "tp": tp, "n_pred": len(predicted), "n_gold": len(gold)}


def retriever_topk_ids(query: str, k: int = 20, pool: int = 120) -> list[str]:
    """Retrieval-only doc_ids (no LLM) — fast precision/recall of the RAG layer alone.

    Dedup chunks -> one-per-doc BEFORE reranking. The reranker has a fixed budget; if fed many
    chunks of the same doc it ranks DEPTH (best passage within a few docs) and buries lower-ranked
    relevant docs, tanking recall. One chunk per doc makes it rank BREADTH across docs instead.
    """
    from rag.retriever import get_retriever
    from rag.reranker import rerank

    r = get_retriever()
    cands = r.hybrid_search(query)[:pool]
    seen, uniq = set(), []
    for c in cands:
        if c.doc_id not in seen:
            seen.add(c.doc_id)
            uniq.append(c)
    reranked = rerank(query, uniq, top_k=k)
    out = []
    for c, _ in reranked:
        if c.doc_id not in out:
            out.append(c.doc_id)
    return out
