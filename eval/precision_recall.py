"""Dimensions 1 (Precision) & 2 (Recall).

Two layers:
  - retrieval-only: does the RAG layer surface the gold-relevant docs? (no LLM, deterministic)
  - agent-level: of the docs the AGENT presents in its answer, how many are relevant (precision),
    and how many gold docs did it surface (recall). Recall averaged over 3 query phrasings.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import GOLD, prf, run_agent, retriever_topk_ids

GOLD_RELEVANT = set(GOLD["relevant_doc_ids"])

PHRASINGS = [
    "Find precedents on whether an insurer can deny a death claim because the truck driver had no valid driving licence.",
    "Which judgments decide insurer liability when the driver was unlicensed and the policy is said to be void?",
    "Cases about pay-and-recover and insurer liability for an accident caused by an unlicensed commercial-vehicle driver.",
]


def retrieval_eval(k: int = 30) -> dict:
    rows = []
    for q in PHRASINGS:
        ids = set(retriever_topk_ids(q, k=k))
        rows.append(prf(ids, GOLD_RELEVANT))
    avg = lambda key: round(sum(r[key] for r in rows) / len(rows), 3)
    return {"k": k, "per_query": rows,
            "avg_precision": avg("precision"), "avg_recall": avg("recall"), "avg_f1": avg("f1")}


def agent_eval() -> dict:
    rows = []
    for q in PHRASINGS:
        res = run_agent(q)
        cited = set(res["cited_ids"])
        retrieved = set(res["retrieved_ids"])
        rows.append({
            "query": q,
            "cited_precision_recall": prf(cited, GOLD_RELEVANT),   # precision = quality of presented set
            "retrieved_recall": prf(retrieved, GOLD_RELEVANT)["recall"],
            "cited_ids": sorted(cited),
        })
    avg_p = round(sum(r["cited_precision_recall"]["precision"] for r in rows) / len(rows), 3)
    avg_r = round(sum(r["cited_precision_recall"]["recall"] for r in rows) / len(rows), 3)
    return {"avg_cited_precision": avg_p, "avg_cited_recall": avg_r, "per_query": rows}


def main(agent: bool = True) -> dict:
    out = {"dimension": "precision_recall", "retrieval_layer": retrieval_eval()}
    if agent:
        out["agent_layer"] = agent_eval()
    return out


if __name__ == "__main__":
    run_agent_layer = "--retrieval-only" not in sys.argv
    print(json.dumps(main(agent=run_agent_layer), indent=2))
