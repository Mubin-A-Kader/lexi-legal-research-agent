"""Central config. Models + paths + retrieval params. Override via env."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = Path(
    os.getenv(
        "CORPUS_DIR",
        "/Users/mac-home/rag-cw/resume/Lexi/lexi_research_take_home_assessment_docs",
    )
)
EXTRACTED_JSON = DATA_DIR / "extracted_docs.json"  # pre-parsed PDF text, committed to repo
CHUNKS_JSON = DATA_DIR / "chunks.json"             # pre-chunked, committed to repo
CHROMA_DIR = DATA_DIR / "chroma_db"

# --- Models / LLM provider ---
# Provider-agnostic. Default = OpenAI (user supplies OPENAI_API_KEY).
# gpt-4o-mini is the cheapest OpenAI model with reliable tool-calling.
# OPENAI_BASE_URL lets you point the same code at any OpenAI-compatible API (e.g. DeepSeek).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" | "anthropic"
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # None = api.openai.com

# --- Embeddings / rerank ---
# Dense retrieval embeds via the OpenAI embeddings API (text-embedding-3-small). Keeps NO
# embedding model in the process — no onnxruntime, no torch — so the hosted runtime fits the
# 512 MB free tier with margin. (Local ONNX MiniLM OOM'd the box at inference peak.)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))  # text-embedding-3-small native dim
# Rerank reuses the same OpenAI embedder for a bi-encoder cosine re-score — no extra model load,
# costs only embedding-API calls. ON by default; switch off with ENABLE_RERANKER=false.
# ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() in ("1", "true", "yes")  # contextual retrieval benefit
ENABLE_RERANKER = True
# --- Chunking ---
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval ---
BM25_TOPK = int(os.getenv("BM25_TOPK", "80"))
DENSE_TOPK = int(os.getenv("DENSE_TOPK", "80"))
RERANK_TOPK = int(os.getenv("RERANK_TOPK", "30"))

COLLECTION_NAME = "legal_judgments"
