"""reranker.py — Cross-encoder reranking of candidate chunks.

A cross-encoder reads (query, chunk) TOGETHER and scores true relevance, unlike
the bi-encoder embeddings that scored query and chunk independently. This is the
step that should pull an answer-bearing financial table above chatty prose that
merely mentions the topic.
"""

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 6) -> list[dict]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c["document"]) for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
        return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_k]]
