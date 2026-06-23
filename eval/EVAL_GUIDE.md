# Eval Guide — How to Run the Evaluation Suite

Step-by-step guide to running the four-dimension evaluation framework and reading the results.
For *why* these dimensions and the result analysis, see [`../docs/ADR.md`](../docs/ADR.md) §6–7.

---

## What gets measured

| Dim | Name | What it answers | File |
|---|---|---|---|
| 1 | **Precision** | Of the docs the agent presents, how many are gold-relevant? | `precision_recall.py` |
| 2 | **Recall** | Of the gold-relevant docs, how many did it surface? (avg of 3 phrasings) | `precision_recall.py` |
| 3 | **Reasoning Quality** | LLM-judge scores the memo 1–5 on faithfulness, specificity, legal logic, adverse handling — judge is given the cited source texts to catch hallucinated holdings | `reasoning_eval.py` |
| 4 | **Adverse Identification** | Does it surface the known-adverse precedents *and* label them adverse (not supporting)? | `adverse_eval.py` |

All four score the agent against a hand-curated gold standard: [`gold_standard.json`](gold_standard.json).

---

## Step 1 — Environment

From the repo root, with the venv active:

```bash
source .venv/bin/activate
pip install -r requirements.txt        # core (ONNX embeddings, torch-free)
```

Set the LLM key. Default provider is OpenAI:

```bash
export OPENAI_API_KEY=sk-...           # or put it in .env
```

(Switching provider: `export LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY=...`. The runner picks
the key for whichever provider is active.)

---

## Step 2 — Run

### A. Full run — all 4 dimensions (calls the agent + the LLM judge)

```bash
python eval/run_all.py
```

- Runs the real tool-calling agent over the gold queries, then scores all four dimensions.
- Costs API calls; takes ~250 s on `gpt-4o-mini`.
- Needs the active provider's API key — if no key is found it **auto-falls back to retrieval-only**.

### B. Retrieval-only — dimensions 1 & 2, no API, no cost

```bash
python eval/run_all.py --retrieval-only
```

- Scores the RAG layer (hybrid BM25 + dense + RRF + rerank) directly, skipping the agent and judge.
- No API key needed. Fast. Use this to iterate on retrieval/chunking/rerank changes.

### C. One dimension at a time (debugging)

```bash
python -m eval.precision_recall      # dims 1 + 2
python -m eval.reasoning_eval        # dim 3  (needs key)
python -m eval.adverse_eval          # dim 4  (needs key)
```

---

## Step 3 — Read the results

Every run writes [`results.json`](results.json) and prints a summary. Shape:

```json
{
  "mode": "full-agent",
  "dimensions": {
    "precision_recall":      { "retrieval_layer": {...}, "agent_layer": {...} },
    "reasoning_quality":     { "scores": { "mean_score": 3.75, ... } },
    "adverse_identification":{ "surfaced_recall": 0.33, "labeled_recall": 0.0 }
  },
  "elapsed_s": 248.1
}
```

What "good" looks like (and the known weak spots) is documented in ADR §6 "Reading the numbers"
and §7 "Failure analysis". Headline: precision high, recall is the ceiling, adverse *labelling* is
the priority failure.

---

## Step 4 — Extend the gold standard

`gold_standard.json` fields:

- `relevant_doc_ids` — every judgment relevant to the case theme (drives precision/recall).
- `supporting_doc_ids` / `adverse_doc_ids` — stance labels (drive dims 3–4).
- `case_context` — the client position the agent/judge reason against.
- `general_query_gold` — non-research queries (e.g. "which docs involve commercial vehicles?") to
  prove the agent handles general lookups, not just deep research.

To add a test case: add the query + expected doc_ids to the relevant list, re-run. Stance labels are
curated from each judgment's **outcome** (pay-and-recover ⇒ supporting; insurer exonerated ⇒ adverse).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Runs retrieval-only despite wanting full | No API key for the active provider. `echo $OPENAI_API_KEY`; check `LLM_PROVIDER`. |
| `ModuleNotFoundError` | Wrong interpreter — use the venv: `.venv/bin/python eval/run_all.py`. |
| Slow first run | Cold start: ONNX model load + Chroma index load. Subsequent runs reuse the singletons. |
| Adverse recall = 0.0 | Known finding, not a bug — see ADR §6/§7 (stance-classification, not retrieval). |
