"""hybrid_search.py — Hybrid retrieval (dense + BM25), hard-scoped to one company.

Consumes what ingest.py built:
  - Chroma collection 'annual_reports' (cosine over ENRICHED-text embeddings,
    clean text stored as the document, sha1/company/heading/is_table metadata)
  - data/bm25.pkl: [{id, document, metadata, tokens(enriched)}]

Pipeline per query:
  dense  : embed query -> Chroma .query(where sha1 == target) -> top-N
  sparse : BM25 over THAT company's chunks only -> top-N
  merge  : Reciprocal Rank Fusion -> top-k candidates for the reranker

The company filter is a HARD pre-filter on both sides, so a query can only ever
see the target company's chunks (no cross-document contamination).
"""

import pickle
import re
from collections import defaultdict
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1] / "data"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_PATH = DATA_DIR / "bm25.pkl"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "annual_reports"
# bge-v1.5: prefix the QUERY (not passages) with this instruction for retrieval.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RRF_K = 60  # Reciprocal Rank Fusion constant (standard default)

_TOKEN_RE = re.compile(r"[a-z0-9&]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.coll = client.get_collection(COLLECTION)

        records = pickle.loads(BM25_PATH.read_bytes())
        self._by_sha: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            self._by_sha[r["metadata"]["sha1"]].append(r)
        self._bm25_cache: dict[str, tuple] = {}   # sha1 -> (BM25Okapi|None, records)

    # ---- dense ----
    def _dense(self, query: str, sha1: str, n: int) -> list[tuple]:
        vec = self.model.encode(QUERY_INSTRUCTION + query,
                                normalize_embeddings=True).tolist()
        res = self.coll.query(query_embeddings=[vec], n_results=n,
                              where={"sha1": sha1})
        return list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0]))

    # ---- sparse (BM25 over this company's chunks only) ----
    def _bm25_for(self, sha1: str) -> tuple:
        if sha1 not in self._bm25_cache:
            recs = self._by_sha.get(sha1, [])
            bm = BM25Okapi([r["tokens"] for r in recs]) if recs else None
            self._bm25_cache[sha1] = (bm, recs)
        return self._bm25_cache[sha1]

    def _sparse(self, query: str, sha1: str, n: int) -> list[tuple]:
        bm, recs = self._bm25_for(sha1)
        if not bm:
            return []
        scores = bm.get_scores(tokenize(query))
        top = sorted(range(len(recs)), key=lambda i: scores[i], reverse=True)[:n]
        return [(recs[i]["id"], recs[i]["document"], recs[i]["metadata"]) for i in top]

    # ---- merge ----
    def search(self, query: str, sha1: str, k: int = 15,
               n_dense: int = 30, n_sparse: int = 30) -> list[dict]:
        dense = self._dense(query, sha1, n_dense)
        sparse = self._sparse(query, sha1, n_sparse)

        rrf: dict[str, float] = defaultdict(float)
        info: dict[str, tuple] = {}
        for ranked in (dense, sparse):
            for rank, (cid, doc, meta) in enumerate(ranked):
                rrf[cid] += 1.0 / (RRF_K + rank)
                info[cid] = (doc, meta)

        ordered = sorted(rrf, key=lambda c: rrf[c], reverse=True)[:k]
        return [{"id": c, "document": info[c][0], "metadata": info[c][1],
                 "score": rrf[c]} for c in ordered]


if __name__ == "__main__":
    # Smoke test (requires ingest.py to have been run first).
    # Usage: python retrieval/hybrid_search.py ["query"] [sha1]
    import sys

    r = HybridRetriever()
    # Default to a doc that was ACTUALLY ingested (handles --limit runs).
    sha1 = sys.argv[2] if len(sys.argv) > 2 else next(iter(r._by_sha))
    query = sys.argv[1] if len(sys.argv) > 1 else "cash flow from operations and total revenue"
    company = r._by_sha[sha1][0]["metadata"]["company"]

    print(f"query={query!r}\ncompany={company} ({sha1[:10]})\n")
    hits = r.search(query, sha1, k=5)
    if not hits:
        print("  (no hits — is this sha1 in the ingested set?)")
    for h in hits:
        first = h["document"].splitlines()[0][:70]
        print(f"  {h['score']:.4f} | table={h['metadata']['is_table']} | {first}")
