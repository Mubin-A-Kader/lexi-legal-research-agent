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

A **Claude tool-calling agent** (LangChain `AgentExecutor`) owns the control flow. It is given five
tools and decides which to call, how often, and in what order:

`semantic_search` · `keyword_search` · `get_document` · `analyze_precedent` · `list_documents`

- **General query** → typically one `semantic_search`, answer with cited `doc_id`s.
- **Deep research** → broad search → `analyze_precedent` on the strongest hits → synthesize a memo
  with supporting + adverse sections. The loop count is the model's choice, not the code's.

This is how "flexibility without if/else" is achieved: **routing is emergent**, expressed through the
system prompt and tool descriptions rather than branching logic. The same binary handles "which cases
involve commercial vehicles?" and "find adverse precedents on policy-void defences."

### Retrieval layer (the `*_search` tools)

```
Hybrid: BM25 (lexical) + Chroma dense (MiniLM)  →  RRF fusion  →  cross-encoder rerank  →  top-k
```

- **Hybrid + RRF**, because legal text needs both: BM25 nails exact tokens (case names, "Section 149",
  "pay and recover"); dense catches paraphrase ("unlicensed driver" ≈ "driver without an effective
  licence"). Reciprocal Rank Fusion needs no score normalisation between the two.
- **Cross-encoder rerank** restores the precision that the small embedding model gives up.
- **Chunking:** 800-token, 100-overlap, paragraph-aware, with page spans retained for citation.

## 3. Key tradeoffs

| Decision | Chose | Gave up | Why acceptable |
|---|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (90 MB) | `bge-large` quality | Fits 512 MB; reranker recovers precision |
| LLM | OpenAI `gpt-4o-mini` | Sonnet/Opus reasoning ceiling | Cheapest with reliable tool-calling; provider is a one-line switch |
| LLM access | Provider-agnostic `agent/llm.py` | — | OpenAI ↔ Anthropic ↔ any OpenAI-compatible API (DeepSeek) via env only |
| Framework | LangChain `AgentExecutor` | Raw SDK control | Streaming callbacks → required live trace, less glue code |
| Index at boot | **Commit** Chroma (24 MB) | Repo size | Avoids 72 s rebuild *and* the embed-all RAM spike on free tier |
| Reranker | MiniLM cross-encoder | Latency | Falls back TinyBERT → identity on OOM, so it degrades safely |

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

Four automated evals (`eval/`) against a hand-curated gold standard (`eval/gold_standard.json`, 18
relevant docs incl. 3 confirmed-adverse), built by mining the corpus for the case theme
(unlicensed-driver vs. insurer liability) and verifying each judgment's **outcome** from its text
(pay-and-recover ⇒ supporting; insurer exonerated ⇒ adverse).

Results below are from `python eval/run_all.py` on `gpt-4o-mini` (full agent run ~250 s; reproducible,
saved to `eval/results.json`). Retrieval-layer numbers also reproduce offline with `--retrieval-only`.

| Dim | Metric | Method | Result |
|---|---|---|---|
| 1 Precision | gold-relevant ÷ presented | retrieval @k=12 / agent-cited, avg of 3 phrasings | **retrieval 0.92 · agent 1.00** |
| 2 Recall | gold found ÷ gold total | same | **retrieval 0.37/query · agent 0.28** |
| 3 Reasoning | 1-5 × 4 rubrics, LLM judge given the cited source texts | `reasoning_eval.py` | **mean 3.75/5** (faithful 3, specific 4, logic 4, adverse-handling 4) |
| 4 Adverse ID | surfaced & labeled-adverse recall over 3 gold-adverse docs | `adverse_eval.py` | **0.33 surfaced / 0.00 labeled** |

**Reading the numbers.**
- **Precision is excellent** (retrieval 0.92, agent 1.00) — the agent essentially never presents an
  irrelevant judgment. It is conservative: it cites only what it actually analysed.
- **Recall is the weakness, and it compounds.** A single broad `semantic_search` returns ~12 hits;
  the agent deep-reads only ~3-4 of them, so **agent-cited recall (0.28) sits near single-query
  retrieval recall (0.37)**. The conservatism that buys precision costs coverage.
- **Adverse identification is the headline failure — and a worked eval→fix example.** Initial run:
  **0.00 surfaced**. The three gold-adverse judgments (`doc_005`, `doc_031`, `doc_041` — insurer
  *exonerated* / Swaran-Singh *limited*) were never even retrieved: the corpus is dominated by
  claimant-favourable "pay-and-recover" judgments, so one semantic query surfaces that majority cluster
  and the rarer exoneration cases rank too low to reach `analyze_precedent`. **Fix applied:** the system
  prompt now instructs the agent to run a second, *opposite-outcome* counter-search during research
  (generic, not case-specific). That lifted **surfaced recall 0.00 → 0.33** (it now finds `doc_031`,
  the key adverse SC authority). **But labeled-adverse recall is still 0.00**: the agent surfaced
  `doc_031` yet filed it under *Supporting*. `doc_031` (Laxmi Narain Dhut) contains both "insurer
  liable" and liability-*limiting* language, and the model latched onto the favourable half — a genuine
  **stance-classification** error, not a retrieval one. So the agent can look balanced while
  mis-labelling the precedent that actually threatens the client. **A confident-but-wrong adverse
  analysis is the most dangerous failure mode for this product** — hence a dedicated dimension.

## 7. Failure analysis — where it breaks, what I'd fix first

1. **Adverse stance-labelling** *(highest priority — correctness, not just coverage)*. The
   counter-search fix (above) already lifted adverse *surfacing* 0.00 → 0.33, but *labelling* is still
   0.00: the agent surfaces the adverse authority and then files it as Supporting because nuanced
   judgments (`doc_031`, Laxmi Narain Dhut) carry both liability and liability-*limiting* language.
   Fix: an explicit stance rubric in `analyze_precedent` ("a judgment that *limits* recovery or
   *exonerates* the insurer is ADVERSE even if it also recites general liability principles"), a
   second verification pass on each classification, and `strict` JSON output to kill parse drift.
2. **Recall ceiling generally (0.28)** — even for supporting cases the agent reads only its top few
   hits. Fix: raise the analyse-set size, add a `multi_query` expansion tool (3-4 sub-queries → union →
   rerank), and measure answer-level (not just retrieval) recall as the corpus scales.
3. **Faithfulness = 3** — the judge docks the memo for mild overreach (stating holdings slightly
   beyond what the cited chunk supports). Fix: a citation-entailment guardrail (cheap NLI / judge pass)
   that checks every asserted holding against its chunk before it reaches the user.
4. **Gold-standard subjectivity** — relevance/stance curated by an engineer, not a lawyer; defensible,
   not authoritative (note: the agent's call that `doc_006` is adverse is itself arguable). Fix:
   multi-annotator labels + confidence bands.
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
