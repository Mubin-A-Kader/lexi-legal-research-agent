"""Streamlit UI for the legal precedent research agent.

REQUIRED by the assessment: intermediate reasoning steps must be visible — tool calls,
documents retrieved, scores, and classifications — not just the final answer.

We use a custom LangChain callback that renders each tool call + result live into the page.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

import config

st.set_page_config(page_title="Lexi Legal Precedent Researcher", page_icon="⚖️", layout="wide")


class StreamlitTraceCallback(BaseCallbackHandler):
    """Render agent tool calls + results into an expander as they happen."""

    def __init__(self, container):
        self.container = container
        self.step = 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.step += 1
        name = serialized.get("name", "tool")
        with self.container:
            st.markdown(f"**Step {self.step} — 🔧 `{name}`**")
            st.code(_short(input_str), language="text")

    def on_tool_end(self, output, **kwargs):
        with self.container:
            st.markdown("↳ _retrieved:_")
            st.markdown(_render_tool_output(str(output)))
            st.divider()

    def on_llm_start(self, *a, **k):
        pass

    def on_agent_finish(self, finish, **kwargs):
        with self.container:
            st.markdown("**✅ Agent finished reasoning.**")


def _short(s: str, n: int = 300) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + " …"


def _render_tool_output(out: str) -> str:
    """Pretty-print retrieval JSON as a compact list; pass prose through."""
    try:
        data = json.loads(out)
    except Exception:
        return f"> {_short(out, 800)}"
    if isinstance(data, list):
        lines = []
        for d in data[:10]:
            if "doc_id" in d:
                score = d.get("rerank_score", "")
                title = d.get("title", "")[:60]
                snip = _short(d.get("snippet", ""), 160)
                lines.append(f"- **{d['doc_id']}** ({score}) — {title}\n  > {snip}")
        return "\n".join(lines) or f"> {_short(out, 800)}"
    return f"> {_short(out, 800)}"


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚖️ Lexi Legal Researcher")
    st.caption("Precedent research over 56 Indian motor-accident / insurance judgments.")
    st.markdown(
        "**How it works** — a Claude tool-calling agent decides its own workflow per query: "
        "hybrid search (BM25 + dense + cross-encoder rerank), reads judgments, and classifies "
        "precedents as supporting / adverse / neutral."
    )
    st.markdown("---")
    st.subheader("Try")
    examples = [
        "Which judgments involve commercial vehicles?",
        "Find precedents supporting a claim where the truck driver had no valid license and the "
        "insurer says the policy is void.",
        "Find adverse precedents the insurer could use to deny liability for an unlicensed driver.",
        "What compensation was awarded in death cases with minor children as dependents?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set. Add it in the environment / Render secrets.")


# ---------------- Main ----------------
st.title("Legal Precedent Research Agent")

if "history" not in st.session_state:
    st.session_state["history"] = []

# Replay prior turns
for turn in st.session_state["history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

query = st.chat_input("Ask a research question…")
if "pending" in st.session_state and not query:
    query = st.session_state.pop("pending")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state["history"].append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        steps_box = st.expander("🧠 Agent reasoning steps (tools, retrieval, ranking)", expanded=True)
        cb = StreamlitTraceCallback(steps_box)

        with st.spinner("Researching the corpus…"):
            from agent.agent import build_agent

            executor = build_agent(callbacks=[cb])
            try:
                result = executor.invoke({"input": query})
                answer = result["output"]
            except Exception as e:
                answer = f"⚠️ Error: {e}"

        if isinstance(answer, list):  # anthropic content blocks
            answer = "\n".join(b.get("text", "") for b in answer if isinstance(b, dict))
        st.markdown("### Answer")
        st.markdown(answer)
        st.session_state["history"].append({"role": "assistant", "content": answer})
