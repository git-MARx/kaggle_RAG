"""graph.py — Wire the nodes into a LangGraph state machine (see ../README.md)."""

from langgraph.graph import END, START, StateGraph

from pipeline.nodes import (analyze_query, check_hallucination, compare,
                           expand_query, finalize, generate, grade, rerank,
                           retrieve, route_after_analyze, route_after_generate,
                           route_grade, route_hallucination)
from pipeline.state import AgentState


def build_app():
    g = StateGraph(AgentState)

    for name, fn in [("analyze", analyze_query), ("retrieve", retrieve),
                     ("rerank", rerank), ("grade", grade), ("expand", expand_query),
                     ("generate", generate), ("hallucination", check_hallucination),
                     ("compare", compare), ("finalize", finalize)]:
        g.add_node(name, fn)

    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", route_after_analyze,
                            {"single": "retrieve", "compare": "compare"})

    # single-company retrieval loop
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "grade")
    g.add_conditional_edges("grade", route_grade,
                            {"ok": "generate", "retry": "expand", "giveup": "generate"})
    g.add_edge("expand", "retrieve")

    # generation -> verify only when fabrication is the risk; else straight to finalize
    g.add_conditional_edges("generate", route_after_generate,
                            {"verify": "hallucination", "skip": "finalize"})
    g.add_conditional_edges("hallucination", route_hallucination,
                            {"ok": "finalize", "retry": "generate", "giveup": "finalize"})

    # comparison path
    g.add_edge("compare", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


app = build_app()
