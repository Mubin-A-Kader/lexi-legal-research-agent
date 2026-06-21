"""PDF -> text -> token-aware chunks. Pre-extract to JSON so startup skips PDF parsing."""
import json
import re
from dataclasses import dataclass, asdict
from typing import List

import fitz  # PyMuPDF
import tiktoken

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class DocChunk:
    chunk_id: str       # e.g. "doc_001#3"
    doc_id: str         # "doc_001"
    text: str
    page_start: int
    page_end: int
    title: str          # first non-trivial line of judgment, heuristic
    n_tokens: int


def _ntok(text: str) -> int:
    return len(_enc.encode(text))


def extract_pdf(path: Path) -> dict:
    """Return {doc_id, title, pages:[{page, text}], full_text}."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        txt = page.get_text("text")
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        pages.append({"page": i + 1, "text": txt})
    doc.close()
    full = "\n\n".join(p["text"] for p in pages)
    title = _guess_title(full)
    return {"doc_id": path.stem, "title": title, "pages": pages, "full_text": full}


def _guess_title(full_text: str) -> str:
    """Heuristic: first line that looks like a case caption (has 'v.' / 'vs') else first real line."""
    lines = [l.strip() for l in full_text.splitlines() if len(l.strip()) > 5]
    for l in lines[:40]:
        if re.search(r"\b(v\.?s?\.?|versus)\b", l, re.I) and len(l) < 160:
            return l
    return lines[0][:160] if lines else "Untitled judgment"


def chunk_document(extracted: dict) -> List[DocChunk]:
    """Paragraph-aware, ~CHUNK_TOKENS with CHUNK_OVERLAP. Tracks page span."""
    doc_id = extracted["doc_id"]
    title = extracted["title"]

    # Build (text, page) paragraph units.
    units = []
    for p in extracted["pages"]:
        for para in re.split(r"\n\s*\n", p["text"]):
            para = para.strip()
            if para:
                units.append((para, p["page"]))

    chunks: List[DocChunk] = []
    cur, cur_pages, cur_tok, idx = [], [], 0, 0

    def flush():
        nonlocal cur, cur_pages, cur_tok, idx
        if not cur:
            return
        text = "\n\n".join(cur)
        chunks.append(
            DocChunk(
                chunk_id=f"{doc_id}#{idx}",
                doc_id=doc_id,
                text=text,
                page_start=min(cur_pages),
                page_end=max(cur_pages),
                title=title,
                n_tokens=_ntok(text),
            )
        )
        idx += 1

    for para, page in units:
        ptok = _ntok(para)
        if cur_tok + ptok > config.CHUNK_TOKENS and cur:
            flush()
            # overlap: carry trailing paragraphs up to CHUNK_OVERLAP tokens
            ov, ov_tok, ov_pages = [], 0, []
            for back_para, back_page in zip(reversed(cur), reversed(cur_pages)):
                t = _ntok(back_para)
                if ov_tok + t > config.CHUNK_OVERLAP:
                    break
                ov.insert(0, back_para)
                ov_pages.insert(0, back_page)
                ov_tok += t
            cur, cur_pages, cur_tok = ov[:], ov_pages[:], ov_tok
        cur.append(para)
        cur_pages.append(page)
        cur_tok += ptok
    flush()
    return chunks


def build_corpus(corpus_dir: Path = None) -> tuple[list[dict], list[DocChunk]]:
    corpus_dir = corpus_dir or config.CORPUS_DIR
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    docs, all_chunks = [], []
    for pdf in pdfs:
        ex = extract_pdf(pdf)
        docs.append(ex)
        all_chunks.extend(chunk_document(ex))
    return docs, all_chunks


def save_extracted(docs: list[dict], chunks: list[DocChunk]):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Store doc metadata + full text (drop per-page to keep file lean but keep full_text)
    slim_docs = [
        {"doc_id": d["doc_id"], "title": d["title"], "full_text": d["full_text"]}
        for d in docs
    ]
    config.EXTRACTED_JSON.write_text(json.dumps(slim_docs, ensure_ascii=False, indent=2))
    config.CHUNKS_JSON.write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2)
    )


def load_chunks() -> List[DocChunk]:
    raw = json.loads(config.CHUNKS_JSON.read_text())
    return [DocChunk(**c) for c in raw]


def load_docs() -> dict:
    """doc_id -> {title, full_text}"""
    raw = json.loads(config.EXTRACTED_JSON.read_text())
    return {d["doc_id"]: d for d in raw}
