"""nodes.py — The graph nodes (logic for each box in the README flowchart).

Run from the project root so the cross-package imports resolve:
    python run.py        /        python evaluate.py
"""

import json

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

import config
from pipeline.schemas import (Decomposition, Grade, Grounded, NumberAnswer,
                              default_for_kind, finalize_answer, schema_for_kind)
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import Reranker

load_dotenv(config.PROJECT_ROOT.parent / "agentic-rag" / ".env")  # reuse the agentic-rag key

# ---- shared resources (loaded once) ----
RETRIEVER = HybridRetriever()
RERANKER = Reranker()
LLM = ChatAnthropic(model=config.LLM_MODEL, temperature=0)

_subset = json.loads(config.SUBSET_PATH.read_text())
NAME2SHA = {r["company_name"]: r["sha1"] for r in _subset}
NAME2CUR = {r["company_name"]: r.get("cur", "USD") for r in _subset}
COMPANY_NAMES = list(NAME2SHA)


# ---- prompts ----
RULES = {
    "number": "Return ONLY the numeric value, scaled to the base unit. If the context presents "
              "the figure in a scaled unit (e.g. a header like '(Dollars in millions)' or "
              "'in thousands'), multiply it out to the FULL value: e.g. 406.1 under "
              "'(Dollars in millions)' -> 406100000. Percentages and ratios: return the number "
              "as stated. No units, commas, or words. If the figure is not in the context, return null.",
    "boolean": "Return true only if the context clearly supports it; otherwise false.",
    "name": "Return the single name/title. If not in the context, return null.",
    "names": "Return the list of names/titles. If none are present, return an empty list.",
}

GENERATE_TMPL = (
    "Answer the question using ONLY the context from the company's annual report.\n"
    "Do NOT use outside knowledge.\n\n"
    "Question: {question}\n\nContext:\n{context}\n\n{rules}"
)
GRADE_TMPL = ("Question: {question}\n\nRetrieved context:\n{context}\n\n"
              "Do these chunks contain enough information to answer the question?")
GROUNDED_TMPL = ("Context:\n{context}\n\nProposed answer: {answer}\n\n"
                 "Is the proposed answer fully supported by the context?")
FORMULATE_TMPL = (
    "Rewrite this annual-report question as a concise search query for retrieving the relevant "
    "passage. Keep the key topic and entities; add likely synonyms and financial-statement terms. "
    "DROP boilerplate ('in the annual report', 'according to', 'if not available return N/A', "
    "'Did the company...'). Return ONLY the query.\n\nQuestion: {question}")
EXPAND_TMPL = ("Rewrite this annual-report question into a single, more retrieval-friendly "
               "query: expand abbreviations, add likely financial-statement terms and synonyms. "
               "Return only the rewritten query.\n\nQuestion: {question}\nPrevious query: {prev}")
DECOMP_TMPL = ("This question compares several companies on one financial metric.\n"
               "Question: {question}\n\nIdentify the metric and whether we want the lowest or highest.")


def _context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(c["document"] for c in chunks)


def _resolve_companies(question: str) -> list[str]:
    """Company names mentioned in the question (drop ones that are substrings of a longer match)."""
    hits = [n for n in COMPANY_NAMES if n in question]
    return [n for n in hits if not any(n != m and n in m for m in hits)]


# ---- NODES ----
def analyze_query(state: dict) -> dict:
    companies = _resolve_companies(state["question"])
    shas = [NAME2SHA[c] for c in companies]
    route = "compare" if len(companies) > 1 else "single"
    return {"companies": companies, "shas": shas, "route": route,
            "query": state["question"], "retry_count": 0, "hallucination_retries": 0}


def formulate_query(state: dict) -> dict:
    """Turn the raw question into a focused retrieval query (run before first retrieve).

    The bare yes/no/factual question retrieves poorly — boilerplate buries the
    signal. A topic+synonym query surfaces scattered narrative evidence.
    """
    r = LLM.invoke(FORMULATE_TMPL.format(question=state["question"]))
    return {"query": r.content.strip()}


def retrieve(state: dict) -> dict:
    sha = state["shas"][0] if state["shas"] else ""
    cands = RETRIEVER.search(state["query"], sha, k=config.N_CANDIDATES) if sha else []
    return {"candidates": cands}


def rerank(state: dict) -> dict:
    top = RERANKER.rerank(state["query"], state["candidates"], top_k=config.TOP_K)
    return {"candidates": top, "context": _context(top)}


def grade(state: dict) -> dict:
    if not state["context"]:
        return {"retrieval_ok": False}
    g = LLM.with_structured_output(Grade).invoke(
        GRADE_TMPL.format(question=state["question"], context=state["context"]))
    return {"retrieval_ok": g.sufficient}


def expand_query(state: dict) -> dict:
    r = LLM.invoke(EXPAND_TMPL.format(question=state["question"], prev=state["query"]))
    return {"query": r.content.strip(), "retry_count": state["retry_count"] + 1}


def generate(state: dict) -> dict:
    schema = schema_for_kind(state["kind"])
    prompt = GENERATE_TMPL.format(question=state["question"], context=state["context"],
                                  rules=RULES[state["kind"]])
    return {"answer": LLM.with_structured_output(schema).invoke(prompt)}


def check_hallucination(state: dict) -> dict:
    retries = state.get("hallucination_retries", 0)
    answer = finalize_answer(state["kind"], state["answer"])  # natural value, not pydantic repr
    g = LLM.with_structured_output(Grounded).invoke(
        GROUNDED_TMPL.format(context=state["context"], answer=answer))
    return {"grounded": g.grounded, "hallucination_retries": retries + 1}


def finalize(state: dict) -> dict:
    if state.get("route") == "compare":
        return {"final": state.get("compare_final", "N/A")}
    # Abstain to the kind default if the answer wasn't grounded.
    if state.get("grounded") is False:
        return {"final": default_for_kind(state["kind"])}
    return {"final": finalize_answer(state["kind"], state["answer"])}


def compare(state: dict) -> dict:
    """Comparison questions: extract the metric per company, normalize to EUR, pick min/max."""
    d = LLM.with_structured_output(Decomposition).invoke(
        DECOMP_TMPL.format(question=state["question"]))

    per_eur: dict[str, float] = {}
    for name, sha in zip(state["companies"], state["shas"]):
        subq = f"{d.metric} for {name}"
        top = RERANKER.rerank(subq, RETRIEVER.search(subq, sha, k=config.N_CANDIDATES),
                              top_k=config.TOP_K)
        prompt = GENERATE_TMPL.format(
            question=f"What is the {d.metric} for {name} ({NAME2CUR.get(name, 'USD')})?",
            context=_context(top), rules=RULES["number"])
        ans = LLM.with_structured_output(NumberAnswer).invoke(prompt)
        if ans.value is not None:
            rate = config.FX_TO_EUR.get(NAME2CUR.get(name, "USD"), 1.0)
            per_eur[name] = ans.value * rate

    if not per_eur:
        return {"compare_final": "N/A", "metric": d.metric, "direction": d.direction}
    pick = (min if d.direction == "lowest" else max)(per_eur, key=per_eur.get)
    return {"compare_final": pick, "per_company": per_eur,
            "metric": d.metric, "direction": d.direction}


# ---- routers (for conditional edges) ----
def route_after_analyze(state: dict) -> str:
    return state["route"]


def route_after_generate(state: dict) -> str:
    """Only verify-against-context where fabrication is the risk.

    Booleans were decided FROM the context already; abstentions (N/A) are the
    safe answer. Re-grounding those only adds a flaky False/N-A default.
    """
    if state["kind"] == "boolean":
        return "skip"
    if finalize_answer(state["kind"], state["answer"]) == "N/A":
        return "skip"
    return "verify"


def route_grade(state: dict) -> str:
    if state["retrieval_ok"]:
        return "ok"
    return "retry" if state["retry_count"] < config.MAX_RETRIEVAL_RETRIES else "giveup"


def route_hallucination(state: dict) -> str:
    if state.get("grounded"):
        return "ok"
    return "retry" if state["hallucination_retries"] < config.MAX_HALLUCINATION_RETRIES else "giveup"
