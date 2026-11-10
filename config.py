"""Central settings. Everything tunable lives here (or in your .env) so that
experiments like "what if chunks were smaller?" are a one-line change.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
RESULTS_DIR = ROOT / "eval_results"

# --- Models -----------------------------------------------------------------
# Haiku is the cheap/fast tier: use it while developing. Switch to a bigger
# model for a final demo by setting CLAUDE_MODEL in .env.
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
# The "judge" grades answers during evaluation. Using a stronger model than the
# one being graded is good practice; by default it's the same one.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", MODEL)

# --- Retrieval --------------------------------------------------------------
# "minilm"  -> sentence-transformers all-MiniLM-L6-v2 (real semantic embeddings,
#              downloads ~90MB the first time you run it)
# "hashing" -> offline bag-of-words embeddings (no download; weaker, but great
#              for tests and for seeing what a *bad* embedder does to retrieval)
EMBEDDER = os.getenv("EMBEDDER", "minilm")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))      # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))

# The exact phrase the model must use when the documents don't contain the
# answer. Evaluation looks for it, so keep prompt and eval in sync via this.
ABSTAIN_PHRASE = "I don't have enough information"
