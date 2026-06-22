"""Streamlit UI for the legal precedent research agent — Material Glassmorphism theme.

REQUIRED by the assessment: intermediate reasoning steps must be visible — tool calls,
documents retrieved, scores, and classifications — not just the final answer. A custom
LangChain callback renders each tool call + result live into the page.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

import config

st.set_page_config(page_title="Legal Research Agent", layout="wide")


# ---------------- Material Glassmorphism CSS ----------------
GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Animated rich dark gradient backdrop (deep night palette) */
.stApp {
    background: linear-gradient(-45deg, #050414, #0d0b24, #141233, #1c1746, #0a0820);
    background-size: 400% 400%;
    animation: gradientShift 22s ease infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Global light typography with soft shadow for readability over the gradient */
html, body, .stApp, .stMarkdown, p, span, label, li,
h1, h2, h3, h4, h5, h6, .stChatInput textarea {
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #f3f4ff !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.35);
}
h1, h2, h3 { font-weight: 700 !important; letter-spacing: -0.01em; }

/* Keep Material icon glyphs (avatars, widget icons) on their ligature font — the global
   Inter rule above otherwise renders ligatures as raw text ("face", "smart_toy"). */
span[data-testid="stIconMaterial"],
[data-testid="stChatMessageAvatarUser"] span,
[data-testid="stChatMessageAvatarAssistant"] span,
.material-icons, .material-icons-outlined, .material-symbols-rounded,
[class*="material-symbols"], [class*="material-icons"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons' !important;
    text-shadow: none !important;
    font-feature-settings: 'liga' !important;
}

/* Glass sidebar (dark frosted) */
[data-testid="stSidebar"] {
    background: rgba(10, 8, 32, 0.55) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
[data-testid="stSidebar"] * { text-shadow: 0 1px 2px rgba(0,0,0,0.4); }

/* Frosted Material chat cards (dark glass) */
.stChatMessage {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem;
}

/* Glass chat input (dark) */
[data-testid="stChatInput"] {
    background: rgba(10, 8, 32, 0.6) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
}
[data-testid="stChatInput"] textarea { color: #f3f4ff !important; }

/* Expanders + code blocks as subtle glass panels */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px !important;
}
.stCode, code {
    background: rgba(0, 0, 0, 0.3) !important;
    border-radius: 10px !important;
}

/* Buttons + inputs as glass */
.stButton button, .stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 12px !important;
    color: #f3f4ff !important;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease;
}
.stButton button:hover {
    background: rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-1px);
}

/* --- Leak fix 1: white layout bars (top header + sticky bottom chat container) --- */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
.stApp > header,
.stApp [data-testid="stBottom"] > div {
    background: transparent !important;
    background-color: transparent !important;
}

/* --- Leak fix 2: dropdown white-on-white (BaseWeb popover menu) --- */
div[data-baseweb="popover"],
div[data-baseweb="popover"] div[data-baseweb="menu"],
ul[role="listbox"] {
    background: rgba(20, 18, 48, 0.95) !important;
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
}
ul[role="listbox"] li,
li[role="option"] {
    background: transparent !important;
    color: #f3f4ff !important;
}
ul[role="listbox"] li:hover,
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: rgba(255, 255, 255, 0.12) !important;
}
</style>
"""
st.markdown(GLASS_CSS, unsafe_allow_html=True)


# ---------------- Secret-key auth ----------------
def _check_auth() -> bool:
    """Gate the app behind a shared secret when APP_PASSWORD is set (env or st.secrets).
    If no password is configured (e.g. local dev), the app is open."""
    secret = os.getenv("APP_PASSWORD")
    if not secret:
        try:
            secret = st.secrets.get("APP_PASSWORD")
        except Exception:
            secret = None
    if not secret:
        return True
    if st.session_state.get("authed"):
        return True
    st.title("Legal Research Agent")
    st.markdown("This workspace is access-restricted.")
    entered = st.text_input("Access key", type="password")
    if entered:
        if entered == secret:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Incorrect access key.")
    st.stop()


_check_auth()


class StreamlitTraceCallback(BaseCallbackHandler):
    """Render agent tool calls + results into an expander as they happen."""

    def __init__(self, container):
        self.container = container
        self.step = 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.step += 1
        name = serialized.get("name", "tool")
        with self.container:
            st.markdown(f"**Step {self.step} — `{name}`**")
            st.code(_short(input_str), language="text")

    def on_tool_end(self, output, **kwargs):
        with self.container:
            st.markdown("Retrieved:")
            st.markdown(_render_tool_output(str(output)))
            st.divider()

    def on_llm_start(self, *a, **k):
        pass

    def on_agent_finish(self, finish, **kwargs):
        with self.container:
            st.markdown("**Agent finished reasoning.**")


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
    st.header("Legal Research Agent")
    st.caption("Precedent research over 56 Indian motor-accident / insurance judgments.")
    st.markdown(
        "A tool-calling agent decides its own workflow per query: hybrid search "
        "(BM25 + dense + bi-encoder rerank), reads judgments, and classifies precedents as "
        "supporting / adverse / neutral."
    )
    st.markdown("---")

    st.subheader("Settings")
    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
        help="Tool-calling model for the agent.",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1,
                            help="0 = deterministic. Higher = more exploratory.")
    if st.button("Clear chat", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

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
st.title("Legal Research Agent")

# Explicit emoji avatars — deterministic regardless of the Material icon font.
AVATARS = {"user": "🧑", "assistant": "⚖️"}

if "history" not in st.session_state:
    st.session_state["history"] = []

# Replay prior turns
for turn in st.session_state["history"]:
    with st.chat_message(turn["role"], avatar=AVATARS.get(turn["role"])):
        st.markdown(turn["content"])

query = st.chat_input("Ask a research question…")
if "pending" in st.session_state and not query:
    query = st.session_state.pop("pending")

if query:
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(query)
    st.session_state["history"].append({"role": "user", "content": query})

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        steps_box = st.expander("Agent reasoning steps (tools, retrieval, ranking)", expanded=True)
        cb = StreamlitTraceCallback(steps_box)

        with st.spinner("Researching the corpus…"):
            from agent.agent import build_agent

            executor = build_agent(callbacks=[cb], model=model, temperature=temperature)
            try:
                result = executor.invoke({"input": query})
                answer = result["output"]
            except Exception as e:
                answer = f"Error: {e}"

        if isinstance(answer, list):  # anthropic content blocks
            answer = "\n".join(b.get("text", "") for b in answer if isinstance(b, dict))
        st.markdown("### Answer")
        st.markdown(answer)
        st.session_state["history"].append({"role": "assistant", "content": answer})
