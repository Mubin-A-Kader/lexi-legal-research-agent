"""Run all four eval dimensions, print + save a consolidated report.

Usage:
  python eval/run_all.py                 # full (needs ANTHROPIC_API_KEY; runs the agent)
  python eval/run_all.py --retrieval-only  # dims 1-2 retrieval layer only, no API calls
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from eval import precision_recall, reasoning_eval, adverse_eval

REPORT = Path(__file__).parent / "results.json"


def _has_key() -> bool:
    var = "ANTHROPIC_API_KEY" if config.LLM_PROVIDER == "anthropic" else "OPENAI_API_KEY"
    return bool(os.getenv(var))


def main():
    retrieval_only = "--retrieval-only" in sys.argv or not _has_key()
    t0 = time.time()
    report = {"mode": "retrieval-only" if retrieval_only else "full-agent", "dimensions": {}}

    print("== Dim 1+2: Precision / Recall ==")
    report["dimensions"]["precision_recall"] = precision_recall.main(agent=not retrieval_only)

    if not retrieval_only:
        print("== Dim 3: Reasoning Quality (LLM judge) ==")
        report["dimensions"]["reasoning_quality"] = reasoning_eval.evaluate()
        print("== Dim 4: Adverse Identification ==")
        report["dimensions"]["adverse_identification"] = adverse_eval.evaluate()
    else:
        print("(skipping Dim 3 + 4 — no API key / retrieval-only mode)")

    report["elapsed_s"] = round(time.time() - t0, 1)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {REPORT}  ({report['elapsed_s']}s)")
    print(json.dumps(_summary(report), indent=2))


def _summary(report: dict) -> dict:
    d = report["dimensions"]
    s = {"mode": report["mode"]}
    pr = d.get("precision_recall", {})
    if "retrieval_layer" in pr:
        rl = pr["retrieval_layer"]
        s["retrieval_precision"] = rl["avg_precision"]
        s["retrieval_recall"] = rl["avg_recall"]
    if "agent_layer" in pr:
        s["agent_precision"] = pr["agent_layer"]["avg_cited_precision"]
        s["agent_recall"] = pr["agent_layer"]["avg_cited_recall"]
    if "reasoning_quality" in d:
        s["reasoning_mean"] = d["reasoning_quality"]["scores"].get("mean_score")
    if "adverse_identification" in d:
        s["adverse_surfaced_recall"] = d["adverse_identification"]["surfaced_recall"]
        s["adverse_labeled_recall"] = d["adverse_identification"]["labeled_adverse_recall"]
    return s


if __name__ == "__main__":
    main()
