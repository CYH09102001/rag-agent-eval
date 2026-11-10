"""Streamlit demo UI.   Run:  streamlit run app.py

Modes (decided automatically):
  - live answers   : ANTHROPIC_API_KEY is set, and either no APP_ACCESS_CODE is
                     configured (local use) or the visitor entered the code.
  - search-only    : otherwise. Shows the passages retrieval finds, free, no key needed.
"""
import os

import streamlit as st

# --- Hosted deployments keep settings in Streamlit "secrets"; copy them into
# environment variables BEFORE importing the project modules (config.py reads
# the environment when it is imported). Locally this does nothing.
_SETTINGS = ("ANTHROPIC_API_KEY", "APP_ACCESS_CODE", "DAILY_QUERY_LIMIT", "CLAUDE_MODEL",
             "JUDGE_MODEL", "EMBEDDER", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K")
try:
    for _name in _SETTINGS:
        if _name in st.secrets and not os.environ.get(_name):
            os.environ[_name] = str(st.secrets[_name])
except Exception:  # no secrets file: normal when running locally
    pass

import access  # noqa: E402
import config  # noqa: E402
from agent import run_agent  # noqa: E402
from rag import rag_answer  # noqa: E402
from vectorstore import VectorStore  # noqa: E402

st.set_page_config(page_title="RAG Agent Eval", page_icon="📄")
st.title("📄 Company filings Q&A")
st.caption("Answers come from the documents in data/, with sources. The agent can also call a growth-prediction model.")

EXAMPLES = [
    "Who is the CEO of Prairie Health Systems?",
    "What share of Prairie Health's revenue came from Medicare and Medicaid?",
    "Which of the three companies grew revenue fastest in fiscal 2025?",
    "What dividend per share did Lakeshore Robotics pay? (not in the documents)",
    "Use the growth predictor on Summit Cloud's fiscal 2025 revenue and growth rate.",
]


@st.cache_resource
def load_store() -> VectorStore:
    store = VectorStore()
    store.build_index()
    return store


@st.cache_resource
def load_budget() -> access.DailyBudget:
    return access.DailyBudget()  # one shared counter for every visitor


def safe(text: str) -> str:
    """Streamlit renders $...$ as LaTeX, which garbles dollar amounts. Escape them."""
    return text.replace("$", r"\$")


store = load_store()
budget = load_budget()
HAS_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))
GATED = access.access_code_configured()
st.session_state.setdefault("unlocked", False)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    mode = st.radio("Mode", ["Agent (search + tools)", "Plain RAG"])

    if GATED and HAS_KEY and not st.session_state.unlocked:
        code = st.text_input("Access code for live Claude answers", type="password",
                             help="Reviewers: the code is in the email or resume that linked you here.")
        if code:
            if access.check_code(code):
                st.session_state.unlocked = True
                st.rerun()
            else:
                st.error("Wrong code.")
    LIVE = HAS_KEY and (not GATED or st.session_state.unlocked)
    st.write("**Live answers:** " + ("✅ on" if LIVE else "⛔ off (search-only)"))
    if LIVE:
        st.caption(f"{budget.remaining} live answers left today (shared budget).")

    st.markdown("**Try an example:**")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex.replace(" (not in the documents)", "")

    st.divider()
    st.caption(f"Model `{config.MODEL}` · embedder `{config.EMBEDDER}` · {store.count()} chunks indexed")

if not LIVE:
    if not HAS_KEY:
        st.info("No ANTHROPIC_API_KEY found, so the app is in **search-only mode**: it shows the passages "
                "retrieval finds, but Claude can't write an answer. Add your key to `.env` and restart to enable answers.")
    else:
        st.info("**Search-only mode.** Retrieval works for everyone; enter the access code in the sidebar "
                "to unlock live Claude answers and the agent.")

summary_path = config.RESULTS_DIR / "latest_summary.md"
with st.expander("About this project and its evaluation results"):
    st.markdown(
        "Retrieval-augmented generation over three **fictional** company reports: ChromaDB retrieval, Claude "
        "answers with citations, an agent that can call a scikit-learn model, and an evaluation suite. "
        "See the repository README for architecture, experiments and limitations."
    )
    if summary_path.exists():
        st.markdown(safe(summary_path.read_text()))
    else:
        st.caption("No evaluation results saved yet. Run `python eval.py --mode all` to generate them.")

# --------------------------------------------------------------------- chat
question = st.chat_input("Ask about the companies...") or st.session_state.pop("pending", None)
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        if not LIVE:
            hits = store.retrieve(question)
            st.write("**Top matching passages** (retrieval only, no generated answer):")
            for h in hits:
                st.markdown(f"**{h['source']}** (similarity {h['score']})")
                st.write(safe(h["text"]))
            st.stop()
        if not budget.try_use():
            st.warning("Today's live-answer budget for this demo is used up. Search-only still works; "
                       "live answers reset tomorrow.")
            st.stop()
        try:
            with st.spinner("Thinking..."):
                if mode.startswith("Agent"):
                    out = run_agent(question, store)
                else:
                    out = rag_answer(question, store)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        st.write(safe(out["answer"]))
        if out.get("trace"):
            with st.expander(f"Agent steps ({len(out['trace'])} tool calls)"):
                for t in out["trace"]:
                    st.markdown(f"**Step {t['step']}: `{t['tool']}`**")
                    st.json({"input": t["input"], "output": t["output"]})
        if out.get("chunks"):
            with st.expander("Retrieved passages"):
                for c in out["chunks"]:
                    st.markdown(f"**{c['source']}** (similarity {c['score']})")
                    st.write(safe(c["text"]))
        st.caption(f"Latency: {out['latency_s']}s")
