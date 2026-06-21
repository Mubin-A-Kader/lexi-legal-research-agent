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

# --- Embeddings ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Chunking ---
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# --- Retrieval ---
BM25_TOPK = int(os.getenv("BM25_TOPK", "20"))
DENSE_TOPK = int(os.getenv("DENSE_TOPK", "20"))
RERANK_TOPK = int(os.getenv("RERANK_TOPK", "8"))

COLLECTION_NAME = "legal_judgments"
