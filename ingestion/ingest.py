"""ingest.py — Embed chunks into Chroma + build a BM25 corpus.

EMBED-ONLY strategy:
  for each doc -> chunk_markdown (clean) -> enrich a COPY of each chunk ->
  embed the enriched copy, but STORE the clean chunk as the document.

Single Chroma collection; sha1/company/heading/is_table in metadata so the
retriever can hard-filter by company. A parallel BM25 corpus (tokenized over the
same enriched text) is pickled for hybrid retrieval.

Run:  python3 ingestion/ingest.py            # all 100 docs
      python3 ingestion/ingest.py --limit 2  # smoke test on 2 docs
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import chunk_markdown
from enrich import enrich_text, load_abbreviations

HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1] / "data"
MD_DIR = DATA_DIR / "EnterpriseRAG_2025_02_markdown"
SUBSET_PATH = DATA_DIR / "subset.json"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_PATH = DATA_DIR / "bm25.pkl"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "annual_reports"
ADD_BATCH = 256        # chunks per Chroma add()
ENCODE_BATCH = 64      # chunks per embedding forward pass

_TOKEN_RE = re.compile(r"[a-z0-9&]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def main(limit: int | None = None) -> None:
    subset = json.loads(SUBSET_PATH.read_text())
    if limit:
        subset = subset[:limit]
    print(f"[1/5] Loaded subset: {len(subset)} docs")

    abbr = load_abbreviations()
    print(f"[2/5] Abbreviations loaded")

    print(f"[3/5] Loading embedding model '{EMBED_MODEL}' ...")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"[3/5] Model ready")

    print(f"[4/5] Initialising Chroma at {CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)   # rebuild fresh each run
    except Exception:
        pass
    coll = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    print(f"[4/5] Chroma collection '{COLLECTION}' ready")

    bm25_records: list[dict] = []
    buf_ids, buf_docs, buf_embs, buf_metas = [], [], [], []
    n_chunks = 0

    def flush() -> None:
        if buf_ids:
            coll.add(ids=buf_ids, documents=buf_docs,
                     embeddings=buf_embs, metadatas=buf_metas)
            buf_ids.clear(); buf_docs.clear(); buf_embs.clear(); buf_metas.clear()

    print(f"[5/5] Ingesting {len(subset)} docs ...")
    for i, rec in enumerate(subset, 1):
        sha1 = rec["sha1"]
        company = rec.get("company_name", "")
        md_path = MD_DIR / sha1 / f"{sha1}.md"
        if not md_path.exists():
            print(f"  [{i}/{len(subset)}] SKIP missing markdown: {sha1}")
            continue

        print(f"  [{i}/{len(subset)}] {company[:40]} — chunking ...", flush=True)
        chunks = chunk_markdown(md_path.read_text(encoding="utf-8"), sha1, company)
        amap = abbr.get(sha1, {})

        print(f"  [{i}/{len(subset)}] {company[:40]} — embedding {len(chunks)} chunks ...", flush=True)
        enriched = [enrich_text(c["text"], amap) for c in chunks]
        vectors = model.encode(enriched, batch_size=ENCODE_BATCH,
                               normalize_embeddings=True, show_progress_bar=False)

        for c, enr, vec in zip(chunks, enriched, vectors):
            cid = f"{sha1}:{n_chunks}"
            n_chunks += 1
            meta = {"sha1": sha1, "company": company,
                    "heading": c["heading"], "is_table": c["is_table"]}
            buf_ids.append(cid)
            buf_docs.append(c["text"])           # CLEAN text stored / shown to LLM
            buf_embs.append(vec.tolist())        # vector of ENRICHED text
            buf_metas.append(meta)
            bm25_records.append({"id": cid, "document": c["text"],
                                 "metadata": meta, "tokens": tokenize(enr)})
            if len(buf_ids) >= ADD_BATCH:
                flush()
        print(f"  [{i}/{len(subset)}] {company[:40]} — done ({len(chunks)} chunks)", flush=True)

    flush()
    BM25_PATH.write_bytes(pickle.dumps(bm25_records))
    print(f"\nDone: {n_chunks} chunks across {len(subset)} docs")
    print(f"  Chroma collection '{COLLECTION}' -> {CHROMA_DIR}")
    print(f"  BM25 corpus ({len(bm25_records)} records) -> {BM25_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="ingest only the first N docs (smoke test)")
    main(ap.parse_args().limit)
