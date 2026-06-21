"""Dimension 4 (Adverse Identification).

Does the agent surface the known-adverse precedents AND label them as adverse (not as supporting)?
Two checks per gold-adverse doc:
  surfaced  — doc_id appears anywhere in the agent's answer.
  labeled_adverse — doc_id appears in the answer's adverse/against-client section (not the supporting one).
We locate the adverse section heuristically and confirm the doc_id sits there.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import GOLD, run_agent

GOLD_ADVERSE = GOLD["adverse_doc_ids"]
DEFAULT_TASK = (
    "Research this matter for the claimant. List supporting precedents AND adverse precedents "
    "(judgments the insurer could use against us), then give a strategy recommendation. "
    + GOLD["case_context"]
)


def _adverse_span(answer: str) -> str:
    """Return the slice of the answer that talks about adverse/against-client precedents."""
    low = answer.lower()
    m = re.search(r"adverse|against (the |our )?(client|us)|unfavou?rable|opposing", low)
    if not m:
        return ""
    start = m.start()
    # End at the next top-level section (strategy/recommendation/supporting) after the adverse heading.
    rest = low[start + 10:]
    end_m = re.search(r"\n\s*(#+\s*)?(strategy|recommendation|conclusion|supporting precedent)", rest)
    end = start + 10 + end_m.start() if end_m else len(answer)
    return answer[start:end]


def evaluate(query: str = DEFAULT_TASK) -> dict:
    res = run_agent(query)
    answer = res["output"]
    adverse_span = _adverse_span(answer)

    rows, surfaced_n, labeled_n = [], 0, 0
    for did in GOLD_ADVERSE:
        surfaced = did in answer
        labeled = did in adverse_span
        surfaced_n += surfaced
        labeled_n += labeled
        rows.append({"doc_id": did, "surfaced": surfaced, "labeled_adverse": labeled})

    n = len(GOLD_ADVERSE)
    return {
        "dimension": "adverse_identification",
        "query": query,
        "n_gold_adverse": n,
        "surfaced_recall": round(surfaced_n / n, 3),
        "labeled_adverse_recall": round(labeled_n / n, 3),
        "found_an_adverse_section": bool(adverse_span),
        "per_doc": rows,
        "cited_ids": res["cited_ids"],
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
