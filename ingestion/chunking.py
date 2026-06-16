"""chunking.py — Structure-aware chunking of the (clean) markdown.

Rules:
  - Split into sections by markdown headings (#, ##, ###, ####).
  - Tables (contiguous `|...|` rows) become their OWN chunks.
  - Oversized prose sections are split by paragraph, capped at MAX_CHARS.
  - Every chunk is stamped with its heading path so it stays self-explanatory
    (a table chunk keeps "what statement is this" via the heading; its units/
    year-labels already live inside the table rows).

Output: a list of chunk dicts with metadata. Stays CLEAN — enrichment for the
embedded copy happens later in ingest.py (embed-only strategy).
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1] / "data"
MD_DIR = DATA_DIR / "EnterpriseRAG_2025_02_markdown"
SUBSET_PATH = DATA_DIR / "subset.json"

MAX_CHARS = 1500  # prose chunk size cap (~375 tokens); tables are never split

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_IMAGE_RE = re.compile(r"^!\[.*\]\(.*\)$")
_SPAN_RE = re.compile(r"<[^>]+>")  # strip <span ...></span>, <br>, etc. from headings


_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")  # a |---|---| separator row


def _is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def _first_cell_blank(line: str) -> bool:
    cells = line.split("|")
    return len(cells) > 1 and cells[1].strip() == ""


def _compact_row(line: str) -> str:
    """Strip per-cell padding whitespace (markdown tables pad columns heavily).

    Keeps the pipe structure and content; drops noise that inflates size and
    wastes embedder tokens.  "|  A    |  B   |" -> "| A | B |"
    """
    cells = line.split("|")
    if len(cells) <= 2:
        return line
    out = []
    for c in cells[1:-1]:
        c = c.strip()
        if c and set(c) <= {"-", ":"}:   # a separator cell of dashes/colons
            c = "---"
        out.append(c)
    return "| " + " | ".join(out) + " |"


def _split_table(table_lines: list[str], max_chars: int) -> list[str]:
    """Split a big table into row-groups, repeating the header in each piece.

    Header = rows up to/including the |---| separator, plus following rows whose
    first cell is blank (the years / "(Dollars in millions)" unit rows).
    """
    if len("\n".join(table_lines)) <= max_chars:
        return ["\n".join(table_lines)]

    sep_idx = next((j for j, l in enumerate(table_lines) if _SEP_RE.match(l)), None)
    header_end = (sep_idx + 1) if sep_idx is not None else 1
    while header_end < len(table_lines) and _first_cell_blank(table_lines[header_end]):
        header_end += 1

    header_end = min(header_end, 5)        # guard: don't let header over-capture
    header = table_lines[:header_end]
    data = table_lines[header_end:]
    header_text = "\n".join(header)

    pieces: list[str] = []
    cur: list[str] = []
    cur_len = len(header_text)
    for row in data:
        if cur and cur_len + len(row) + 1 > max_chars:
            pieces.append("\n".join(header + cur))
            cur, cur_len = [], len(header_text)
        cur.append(row)
        cur_len += len(row) + 1
    if cur:
        pieces.append("\n".join(header + cur))
    return pieces


def _clean_heading(text: str) -> str:
    text = _SPAN_RE.sub("", text)          # drop inline html/anchors
    text = text.replace("**", "").replace("__", "")
    return text.strip(" _#*").strip()


def _pack_paragraphs(text: str, max_chars: int) -> list[str]:
    """Greedily pack blank-line-separated paragraphs into <= max_chars chunks."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    cur = ""
    for p in paras:
        if len(p) > max_chars:                       # a single huge paragraph
            if cur:
                out.append(cur)
                cur = ""
            out.extend(p[i:i + max_chars] for i in range(0, len(p), max_chars))
        elif cur and len(cur) + len(p) + 2 > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def chunk_markdown(content: str, sha1: str, company: str, max_chars: int = MAX_CHARS) -> list[dict]:
    lines = content.splitlines()
    heading_stack: dict[int, str] = {}     # level -> cleaned heading text
    chunks: list[dict] = []
    buffer: list[str] = []

    def heading_path() -> str:
        return " > ".join(heading_stack[lvl] for lvl in sorted(heading_stack))

    def make_chunk(text: str, is_table: bool) -> dict:
        hp = heading_path()
        body = f"{hp}\n\n{text}" if hp else text
        return {"text": body, "sha1": sha1, "company": company,
                "heading": hp, "is_table": is_table}

    def flush_prose() -> None:
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            for sub in _pack_paragraphs(text, max_chars):
                chunks.append(make_chunk(sub, is_table=False))

    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEADING_RE.match(line)
        if m:
            flush_prose()
            level = len(m.group(1))
            heading_stack[level] = _clean_heading(m.group(2))
            for deeper in [lvl for lvl in heading_stack if lvl > level]:
                del heading_stack[deeper]
            i += 1
        elif _is_table_row(line):
            flush_prose()
            table = []
            while i < len(lines) and _is_table_row(lines[i]):
                table.append(_compact_row(lines[i]))
                i += 1
            for piece in _split_table(table, max_chars):
                chunks.append(make_chunk(piece, is_table=True))
        else:
            if not _IMAGE_RE.match(line.strip()):   # drop standalone image refs
                buffer.append(line)
            i += 1

    flush_prose()
    return chunks


def chunk_document(sha1: str, company: str = "") -> list[dict]:
    md = (MD_DIR / sha1 / f"{sha1}.md").read_text(encoding="utf-8")
    return chunk_markdown(md, sha1, company)


if __name__ == "__main__":
    subset = {r["sha1"]: r["company_name"] for r in json.loads(SUBSET_PATH.read_text())}
    sha1 = "682de8e45fd9688f3452bc0e18257132a8f3cff6"  # Sonic Automotive
    chunks = chunk_document(sha1, subset.get(sha1, ""))

    tables = [c for c in chunks if c["is_table"]]
    sizes = [len(c["text"]) for c in chunks]
    print(f"doc {sha1[:10]} ({subset.get(sha1)}): {len(chunks)} chunks "
          f"({len(tables)} tables) | size min/avg/max = "
          f"{min(sizes)}/{sum(sizes)//len(sizes)}/{max(sizes)}")

    # Show the cash-flow table chunk (the one holding 406.1) to confirm it's intact.
    for c in tables:
        if "Net cash provided by operating activities" in c["text"]:
            print(f"\n--- table chunk under heading: {c['heading'][:80]} ---")
            print("\n".join(c["text"].splitlines()[:6]), "...")
            break
