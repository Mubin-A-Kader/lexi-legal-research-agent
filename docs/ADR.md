# Architecture Decision Record — Legal Precedent Research Agent

**Status:** Implemented · **Context:** Lexi Backend take-home · **Corpus:** 56 Indian motor-accident / insurance judgments

---

## 1. Problem & forces

Build an agent that does **deep precedent research** (supporting precedents, adverse precedents,
strategy) *and* general corpus Q&A — and must **not** be a hard-coded pipeline, because evaluators
will probe it with arbitrary prompts. Constraints: hosted on a free tier (512 MB RAM), faithful to
the source documents, and measurable on four quality dimensions.

The central tension: **flexibility vs. structure.** A fixed retrieve→classify→synthesize pipeline
would score well on the case brief but fail the flexibility test. A fully open agent is flexible but
can under-retrieve, hallucinate holdings, or skip adverse analysis. The design resolves this by
keeping the *workflow* model-driven while making each *tool* deterministic and grounded.

## 2. Decision: tool-calling agent over a hybrid-RAG layer

A **tool-calling agent** (LangChain `AgentExecutor`) owns the control flow. It is given six
tools and decides which to call, how often, and in what order:

`semantic_search` · `keyword_search` · `get_document` · `find_adverse_precedents` · `analyze_precedent` · `list_documents`

- **General query** → typically one `semantic_search`, answer with cited `doc_id`s.
- **Deep research** → broad search → `analyze_precedent` on the strongest hits → synthesize a memo
  with supporting + adverse sections. The loop count is the model's choice, not the code's.

This is how "flexibility without if/else" is achieved: **routing is emergent**, expressed through the
system prompt and tool descriptions rather than branching logic. The same binary handles "which cases
involve commercial vehicles?" and "find adverse precedents on policy-void defences."

### Retrieval layer (the `*_search` tools)

```
Hybrid: BM25 (lexical) + Chroma dense (MiniLM)  →  RRF fusion  →  ONNX bi-encoder rerank  →  top-k
```

- **Hybrid + RRF**, because legal text needs both: BM25 nails exact tokens (case names, "Section 149",
  "pay and recover"); dense catches paraphrase ("unlicensed driver" ≈ "driver without an effective
  licence"). Reciprocal Rank Fusion needs no score normalisation between the two.
- **Rerank** re-scores the fused candidates by query–chunk cosine similarity, tightening order
  before the agent reads them.
- **Chunking:** 800-token, 100-overlap, paragraph-aware, with page spans retained for citation.

> **Reranker — minimal, torch-free, runs everywhere.** An earlier design used a cross-encoder
> (`ms-marco-MiniLM`), but that pulls **torch** and blew the 512 MB free tier. The shipped reranker is a
> **bi-encoder cosine rerank reusing the same ONNX all-MiniLM-L6-v2 embedder** (onnxruntime, no torch),
> so dense embeddings *and* rerank share one lightweight model and run **identically local and hosted**
> (`ENABLE_RERANKER=true` on both). Trade-off: a bi-encoder rerank is weaker than a cross-encoder
> (no query×chunk attention), so the precision ceiling is lower — but it costs ~0 extra memory and is
> faster (one batched embed pass vs. N pair-scorings). The cross-encoder is the upgrade path on a paid
> instance / hosted reranker endpoint — see §5/§8.

## 3. Key tradeoffs

| Decision | Chose | Gave up | Why acceptable |
|---|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (90 MB) | `bge-large` quality | Fits 512 MB; reranker recovers precision |
| LLM | OpenAI `gpt-4o-mini` | Sonnet/Opus reasoning ceiling | Cheapest with reliable tool-calling; provider is a one-line switch |
| LLM access | Provider-agnostic `agent/llm.py` | — | OpenAI ↔ Anthropic ↔ any OpenAI-compatible API (DeepSeek) via env only |
| Framework | LangChain `AgentExecutor` | Raw SDK control | Streaming callbacks → required live trace, less glue code |
| Index at boot | **Commit** Chroma (24 MB) | Repo size | Avoids 72 s rebuild *and* the embed-all RAM spike on free tier |
| Reranker | ONNX bi-encoder (shared MiniLM) | Cross-encoder precision | Torch-free → runs on 512 MB hosted too; cross-encoder is the paid-tier upgrade |

## 4. Query routing logic

No router model and no `if/else`. The system prompt tells the agent to *match output depth to the
question* and to **always surface adverse precedents during research**. Routing is the agent's tool
choice:

- short factual ask → one retrieval tool, concise answer;
- "find precedents that support/oppose X" → retrieve broadly, `analyze_precedent` on top hits,
  structured memo.

Robustness comes from tool *descriptions* (when to use keyword vs. semantic, when to read full text)
rather than orchestration code. This keeps the surface area for new query types at zero.

## 5. Scaling to 5,000 documents (≈90×)

The architecture holds; the components swap for scale:

| Layer | Now (56 docs) | At 5,000 |
|---|---|---|
| Dense index | Chroma, in-process | **Qdrant / Weaviate** (HNSW, server) — sub-100 ms at millions of vectors |
| Lexical | `rank_bm25` in memory | **Elasticsearch / OpenSearch** BM25 |
| Fusion/rerank | RRF + cross-encoder top-30 | Same, but **two-stage**: ANN recall-1000 → rerank top-100 → top-k |
| Ingestion | synchronous script | **async/batched** pipeline; embeddings cached by content hash |
| Latency/cost | per-call | **semantic cache** (Redis) on frequent queries; batch `analyze_precedent` |
| Metadata | doc_id/title | richer filters (court, year, statute) for **pre-filtered** retrieval |

Nothing in the agent layer changes — only the tool *implementations* behind the same interfaces. That
interface stability is the main payoff of putting deterministic tools under a model-driven loop.

## 6. Evaluation framework & results

Four automated evals (`eval/`) against a hand-curated gold standard (`eval/gold_standard.json`, **19
relevant docs incl. 5 confirmed-adverse**). Labels were built by mining the corpus for the case theme
(unlicensed-driver vs. insurer liability) and verifying each judgment's **operative order** from its
text (pay-and-recover ⇒ supporting; insurer exonerated / claimant left to recover from owner alone ⇒
adverse). An early version of the gold mislabelled six docs; re-reading each holding corrected them
(e.g. `doc_005` is *supporting* — insurer failed to prove the breach and was made to pay — not adverse;
`doc_028`/`doc_030` are *adverse* gratuitous-passenger escapes, not supporting). Each label now carries
an evidence note in `doc_notes`.

Results below are from `python eval/run_all.py` on `gpt-4o-mini` (full agent run ~9 min; reproducible,
saved to `eval/results.json`). Retrieval-layer numbers also reproduce offline with `--retrieval-only`.

| Dim | Metric | Method | Result |
|---|---|---|---|
| 1 Precision | gold-relevant ÷ presented | retrieval @k=30 / agent-cited, avg of 3 phrasings | **retrieval 0.81 · agent 0.94** |
| 2 Recall | gold found ÷ gold total | same | **retrieval 0.93 · agent 0.53** |
| 3 Reasoning | 1-5 × 4 rubrics, LLM judge given the cited source texts | `reasoning_eval.py` | **mean 3.5/5** (faithful 3, specific 4, logic 4, adverse-handling 3) |
| 4 Adverse ID | surfaced & labeled-adverse recall over 5 gold-adverse docs | `adverse_eval.py` | **0.40 surfaced / 0.40 labeled** |

**Reading the numbers.**
- **Retrieval recall went 0.56 → 0.93** after one fix: the reranker was fed many chunks of the *same*
  judgment and ranked depth-within-a-few-docs, burying other relevant judgments. Deduping to one chunk
  per doc *before* reranking (pool 120) made it rank breadth across judgments. Precision traded down
  (0.92 → 0.81) — the right call for precedent research, where missing a case costs more than an extra
  to vet, and `analyze_precedent` filters false positives downstream.
- **Agent recall 0.26 → 0.53.** The agent originally `analyze_precedent`-ed only the handful of hits
  from the adverse counter-search and ignored the main `semantic_search` results. Making it analyse the
  **union** of both searches roughly doubled coverage. It is still below retrieval recall because the
  model runs only two searches per query; a multi-query expansion tool is the next lever (§7/§8).
- **Adverse identification — the documented "headline failure" was largely two artifacts, not the
  agent's reasoning.** (1) *A broken gold key*: all three original adverse labels were wrong. (2)
  *Stance inversion*: the agent passed the **insurer's** framing ("insurer can deny…") as the analysis
  context, so `analyze_precedent` filed every insurer-pays (supporting) judgment as *adverse* and vice
  versa — that alone zeroed labelled-adverse recall. Anchoring stance to the **claimant** regardless of
  context phrasing fixed the inversion; with the corrected key, **labelled-adverse recall went 0.00 →
  0.40, and now equals surfaced recall** — i.e. the agent correctly labels 100% of the adverse docs it
  actually sees. The remaining ceiling is **retrieval**: `doc_002` (tractor-misuse) and `doc_041`
  (award against the owner only) are adverse on grounds whose vocabulary is generic, so BM25 — which
  cannot see holding *direction* — never surfaces them.

## 7. Failure analysis — where it breaks, what I'd fix first

1. **Adverse *retrieval* — now the binding constraint** *(highest priority)*. Stance *labelling* is
   fixed (0.00 → 0.40, and equal to surfaced recall — the agent labels everything it sees correctly).
   What's left is surfacing the adverse minority: `doc_002`/`doc_041` are adverse on grounds with
   generic vocabulary, and BM25 cannot see holding *direction*, so a lexical counter-search can't reach
   them. Fix: a semantic stance-classifier pass over a *wide* candidate set (embed each candidate's
   holding span, score insurer-escape vs insurer-pays) instead of relying on the lexical signal —
   this is real work, not a prompt tweak.
2. **Agent recall ceiling (0.53 vs retrieval 0.93)** — the agent runs only ~2 searches, so it sees a
   fraction of what retrieval can surface. Fix: a `multi_query` expansion tool (3-4 sub-queries → union
   → rerank) and a higher analyse-set size; measure answer-level (not just retrieval) recall as the
   corpus scales.
3. **Reasoning-quality variance** — the LLM judge is single-run and unpinned; mean swung 3.75 → 2.50 →
   3.50 across runs on essentially the same memos. Fix: `temperature=0` and average N judge runs before
   reporting a dim-3 number; faithfulness=3 still reflects mild holding-overreach, addressable with a
   citation-entailment guardrail.
4. **Gold-standard subjectivity** — labels are now read from each operative order and evidence-noted,
   but still curated by an engineer, not a lawyer; the gratuitous-passenger adverse docs (`doc_028`,
   `doc_030`) barely threaten a *third-party-death* client and arguably deserve lower weight. Fix:
   multi-annotator labels + confidence bands + threat-weighting per matter.
5. **Title heuristic / cold start** — captions occasionally parse as procedural noise
   (`"Reserved On: …"`); first hosted request pays model-download + index-load latency. Both cosmetic;
   fix with a tighter cause-title regex and a warm-up ping.

## 8. What I'd do with an extra week

- **Agent-layer recall harness** with query expansion + answer-level (not just retrieval) gold, and a
  recall-vs-cost curve.
- **Citation-faithfulness guardrail**: post-hoc check that every holding the agent states is entailed
  by the cited chunk (NLI or a cheap judge pass) before it reaches the user.
- **Structured outputs** for `analyze_precedent` (`strict` JSON schema) to kill parse fragility.
- **Conversation memory** so follow-ups ("now distinguish doc_005 for me") reuse prior context.
- **Hosted vector DB + metadata filters** (court/year/statute) to pilot the 5,000-doc path for real.
- **Human-in-the-loop labeling UI** to grow the gold set and track regressions per release.
