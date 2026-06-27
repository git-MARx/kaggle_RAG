"""Quick look at chunking output without needing Chroma or embeddings."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "ingestion"))
from chunking import chunk_markdown

DATA_DIR = Path(__file__).parent / "data"
MD_DIR = DATA_DIR / "EnterpriseRAG_2025_02_markdown"
SUBSET_PATH = DATA_DIR / "subset.json"

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5

subset = json.loads(SUBSET_PATH.read_text())[:limit]

total_chunks = 0
CHARS_PER_TOKEN = 4  # rough English estimate
TOKEN_LIMIT = 512

print(f"{'#':<4} {'Company':<35} {'chunks':>6} {'tables':>6} {'avg_ch':>7} {'max_ch':>7} {'max_tok~':>9} {'over?':>6}")
print("-" * 82)
all_chunks = []
for i, rec in enumerate(subset, 1):
    sha1 = rec["sha1"]
    company = rec.get("company_name", "Unknown")
    md_path = MD_DIR / sha1 / f"{sha1}.md"
    if not md_path.exists():
        print(f"{i:<4} {company:<35} MISSING")
        continue
    chunks = chunk_markdown(md_path.read_text(encoding="utf-8"), sha1, company)
    n_tables = sum(1 for c in chunks if c["is_table"])
    lengths = [len(c["text"]) for c in chunks]
    avg_len = int(sum(lengths) / len(lengths)) if lengths else 0
    max_len = max(lengths) if lengths else 0
    max_tok = max_len // CHARS_PER_TOKEN
    over = sum(1 for l in lengths if l // CHARS_PER_TOKEN > TOKEN_LIMIT)
    total_chunks += len(chunks)
    all_chunks.extend(chunks)
    print(f"{i:<4} {company[:35]:<35} {len(chunks):>6} {n_tables:>6} {avg_len:>7} {max_len:>7} {max_tok:>9} {over:>6}")

print("-" * 82)
print(f"Total chunks: {total_chunks}  |  Est. over 512 tokens: {sum(1 for c in all_chunks if len(c['text']) // CHARS_PER_TOKEN > TOKEN_LIMIT)}")

# Show the biggest chunks
print("\nTop 5 longest chunks (chars):")
top5 = sorted(all_chunks, key=lambda c: len(c["text"]), reverse=True)[:5]
for c in top5:
    print(f"  {len(c['text'])} chars (~{len(c['text'])//CHARS_PER_TOKEN} tok)  table={c['is_table']}  heading={c['heading'][:60]}")
