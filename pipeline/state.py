"""state.py — The graph state passed between nodes."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str
    kind: str                  # "number" | "boolean" | "name" | "names"

    # --- resolved by analyze_query ---
    companies: list[str]       # company name(s) named in the question
    shas: list[str]            # matching sha1(s)
    route: str                 # "single" | "compare"

    # --- single-company retrieval loop ---
    query: str                 # current (possibly reformulated) retrieval query
    retry_count: int
    candidates: list[dict]     # reranked chunks
    context: str               # text shown to the LLM
    retrieval_ok: bool

    # --- generation / verification ---
    answer: Any                # typed pydantic answer object (pre-finalize)
    grounded: bool
    hallucination_retries: int

    # --- comparison path ---
    metric: str
    direction: str             # "lowest" | "highest"
    per_company: dict          # company -> value (in EUR)
    compare_final: Any

    # --- output ---
    final: Any                 # submission value (number | bool | str | list | "N/A")
