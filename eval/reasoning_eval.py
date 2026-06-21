"""Dimension 3 (Reasoning Quality) — LLM-as-judge.

We run the agent on the deep-research task, then have a separate judge model score the agent's
precedent reasoning on 4 rubrics (1-5 each). The judge is given the agent's answer AND the actual
text of the cited judgments, so it can catch hallucinated holdings (grounding/faithfulness).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from eval.common import GOLD, run_agent
from rag.ingestion import load_docs

JUDGE_PROMPT = """You are a senior litigator grading a junior's precedent research memo.

CLIENT POSITION:
{case_context}

THE JUNIOR'S MEMO:
{answer}

ACTUAL TEXT OF THE JUDGMENTS THEY CITED (ground truth — use to catch misstatements):
{cited_texts}

Score the memo 1-5 (5=excellent) on each rubric. Be strict.
- faithfulness: Are the stated holdings/facts actually supported by the cited judgment text? Penalise hallucination.
- specificity: Does it cite concrete facts/principles per precedent, not vague generalities?
- legal_logic: Does the reasoning for why a precedent supports/opposes the client hold up?
- adverse_handling: Does it honestly surface and assess precedents AGAINST the client (not only favorable ones)?

Respond JSON only:
{{"faithfulness": int, "specificity": int, "legal_logic": int, "adverse_handling": int, "justification": "<2-3 sentences>"}}"""

DOC_RE = re.compile(r"doc_\d{3}")
DEFAULT_TASK = (
    "Research this matter. Identify supporting precedents, adverse precedents, and give a "
    "strategy recommendation. " + GOLD["case_context"]
)


def evaluate(query: str = DEFAULT_TASK) -> dict:
    res = run_agent(query)
    answer = res["output"]
    docs = load_docs()
    cited = res["cited_ids"] or list(DOC_RE.findall(answer))
    cited_texts = "\n\n".join(
        f"=== {cid} ({docs[cid]['title']}) ===\n{docs[cid]['full_text'][:3500]}"
        for cid in cited[:8] if cid in docs
    ) or "(agent cited no retrievable documents)"

    from agent.llm import complete

    prompt = JUDGE_PROMPT.format(
        case_context=GOLD["case_context"], answer=answer, cited_texts=cited_texts
    )
    raw = complete(prompt, model=config.EVAL_MODEL, max_tokens=800)
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
    try:
        scores = json.loads(raw)
    except Exception:
        scores = {"error": "judge returned non-JSON", "raw": raw}
    rubric_keys = ["faithfulness", "specificity", "legal_logic", "adverse_handling"]
    nums = [scores[k] for k in rubric_keys if isinstance(scores.get(k), (int, float))]
    scores["mean_score"] = round(sum(nums) / len(nums), 2) if nums else None
    return {"dimension": "reasoning_quality", "query": query,
            "cited_ids": cited, "scores": scores}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
