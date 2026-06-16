"""config.py — Shared paths, model names, and pipeline knobs."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

SUBSET_PATH = DATA_DIR / "subset.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
DEVSET_PATH = DATA_DIR / "dev_set_verified.json"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_PATH = DATA_DIR / "bm25.pkl"
ABBR_PATH = DATA_DIR / "abbreviations.json"

# Models
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
COLLECTION = "annual_reports"
LLM_MODEL = "claude-haiku-4-5-20251001"   # cheap model for all nodes

# Retrieval / control knobs
N_CANDIDATES = 40        # hybrid candidates before reranking (recall headroom)
TOP_K = 6                # reranked chunks shown to the LLM
MAX_RETRIEVAL_RETRIES = 1
MAX_HALLUCINATION_RETRIES = 2

# Approximate static FX -> EUR for the comparison questions.
# NOTE: illustrative only. A real solution needs the report-period FX rate.
FX_TO_EUR = {"EUR": 1.0, "USD": 0.92, "GBP": 1.17, "AUD": 0.60,
             "CAD": 0.68, "JPY": 0.0060, "SEK": 0.088, "CHF": 1.04}
