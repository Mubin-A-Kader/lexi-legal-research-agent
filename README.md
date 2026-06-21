# Lexi — Legal Precedent Research Agent

A flexible research agent over a corpus of **56 Indian motor-accident / insurance court judgments**.
It handles both general lookups ("Which judgments involve commercial vehicles?") and deep precedent
research ("Find precedents supporting our claim where the truck driver was unlicensed and the insurer
says the policy is void"). The agent **decides its own workflow per query** — there is no hard-coded
pipeline.

**Hosted app:** _<ADD RENDER URL HERE AFTER DEPLOY>_

---

## What it does

A Claude tool-calling agent is given five tools and chooses which to call, in what order, and how
many times:

| Tool | Purpose |
|---|---|
| `semantic_search` | Hybrid (BM25 + dense) retrieval + cross-encoder rerank |
| `keyword_search` | Exact-term BM25 (case names, statute sections) |
| `get_document` | Full text of one judgment |
| `analyze_precedent` | LLM-classify one judgment vs. a client position → supporting / adverse / neutral + reasoning |
| `list_documents` | Corpus inventory |

A simple query triggers one search; deep research triggers broad search → read/analyze the top
judgments → synthesize supporting precedents, **adverse precedents**, and a strategy recommendation.
The agent loop is model-driven, so new query types are handled by design, not by `if/else`.

The web UI shows **every intermediate step** — each tool call, the documents retrieved with relevance
scores, and the precedent classifications — not just the final answer (a hard requirement).

## Architecture

```
PDF corpus (56) ──ingest──► extracted_docs.json + chunks.json (committed)
                                     │
                    ┌────────────────┴───────────────┐
              BM25 index (in-mem)            Chroma dense index (committed, 24M)
                    └────────────────┬───────────────┘
                          Hybrid retrieve (RRF fuse)
                                     │
                       Cross-encoder rerank → top-k
                                     │
        Claude tool-calling agent (model-driven loop) ──► Streamlit UI (live trace)
```

- **Embeddings:** `all-MiniLM-L6-v2` via ChromaDB's **ONNX** runtime (no torch) — keeps the hosted
  runtime under the 512 MB free-tier limit.
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (torch) — **local only**; OFF on hosted via
  `ENABLE_RERANKER=false`. Install it with `pip install -r requirements-local.txt`.
- **LLM:** OpenAI `gpt-4o-mini` (cheapest with reliable tool-calling) for both the agent and the
  eval/classify calls. Provider is a one-line switch (`LLM_PROVIDER=anthropic` → Claude); set
  `OPENAI_BASE_URL` to reuse the same code against any OpenAI-compatible API (e.g. DeepSeek).
- See [`docs/ADR.md`](docs/ADR.md) for the full design rationale, tradeoffs, and scaling plan.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                 # core (ONNX embeddings, no torch)
pip install -r requirements-local.txt           # optional: cross-encoder reranker (torch)
cp .env.example .env                             # add your OPENAI_API_KEY

# (Optional) re-build indexes from the raw PDFs. Pre-built artifacts are committed,
# so this is only needed if the corpus changes. Set CORPUS_DIR to the PDF folder.
python scripts/ingest.py

streamlit run app.py
```

The Chroma index and extracted text are committed under `data/`, so the app boots without re-parsing
PDFs and without an embed-all memory spike.

## Evaluation

```bash
python eval/run_all.py                 # all 4 dimensions (needs ANTHROPIC_API_KEY)
python eval/run_all.py --retrieval-only  # dims 1-2, RAG layer only, no API calls
```

Four dimensions, each with an automated eval against a hand-curated gold standard
(`eval/gold_standard.json`):

1. **Precision** — of the docs the agent presents, how many are gold-relevant.
2. **Recall** — of the gold-relevant docs, how many the agent surfaced (averaged over 3 phrasings).
3. **Reasoning Quality** — LLM-as-judge scores the agent's memo (1-5) on faithfulness, specificity,
   legal logic, and adverse handling, **with the cited judgment texts supplied** so it can catch
   hallucinated holdings.
4. **Adverse Identification** — does the agent surface the known-adverse precedents *and* label them
   in its adverse section (not as supporting)?

Results + failure analysis: see [`docs/ADR.md`](docs/ADR.md).

## Layout

```
app.py                  Streamlit UI + live agent trace
config.py               Models, paths, retrieval params (env-overridable)
agent/  {agent,tools,prompts}.py   Tool-calling agent + 5 tools + generic prompts
rag/    {ingestion,retriever,reranker}.py   PDF→chunks, hybrid retrieval, rerank
eval/   gold_standard.json + 4 dimension evals + run_all.py
scripts/ingest.py       One-time index build
docs/ADR.md             Architecture Decision Record
```
