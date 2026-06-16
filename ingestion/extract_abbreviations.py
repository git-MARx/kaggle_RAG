"""extract_abbreviations.py — Build a per-document abbreviation map.

Regex-first: find `Full Term ("ABBR")` definitions in each .md.
Output: single JSON keyed by sha1, values are {abbr: full_term} dicts.

Precision comes from the letter-match filter: a parenthetical is only kept as
an abbreviation if the acronym's letters line up with the initials of the
preceding significant words (so "(see Note 6)" / "(2022)" get dropped).
"""

import json
import re
from pathlib import Path

# ---- Paths (edit these if your data lives elsewhere) ----
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]                       # kaggle_rag/
DATA_DIR = PROJECT_ROOT / "data"                     # kaggle_rag/data
SUBSET_PATH = DATA_DIR / "subset.json"
MD_DIR = DATA_DIR / "EnterpriseRAG_2025_02_markdown"
OUTPUT_PATH = DATA_DIR / "abbreviations.json"

# Words that don't contribute an initial to an acronym (EBITDA skips "and", etc.)
STOPWORDS = {"and", "of", "the", "for", "to", "in", "a", "an",
             "on", "at", "with", "or", "by", "&"}

# A parenthetical whose content looks like an acronym: starts with a letter,
# 2-10 chars of letters/digits/&/./-, optionally wrapped in quotes.
ACRONYM_RE = re.compile(r'\(\s*["“\'`]?([A-Za-z][A-Za-z0-9&./\-]{1,9})["”\'`]?\s*\)')


def _first_letter(word: str) -> str:
    for c in word:
        if c.isalpha():
            return c.upper()
    return ""


def _is_significant(word: str) -> bool:
    core = "".join(c for c in word if c.isalnum()).lower()
    return bool(core) and core not in STOPWORDS and any(c.isalpha() for c in word)


def find_full_term(preceding_words: list[str], acronym: str) -> str | None:
    """Return the full term if the preceding words' initials match the acronym."""
    acr = [c.upper() for c in acronym if c.isalpha()]
    if len(acr) < 2:
        return None

    sig_positions = [i for i, w in enumerate(preceding_words) if _is_significant(w)]
    if len(sig_positions) < len(acr):
        return None

    # Take the last len(acr) significant words and compare their initials.
    chosen = sig_positions[-len(acr):]
    initials = [_first_letter(preceding_words[i]) for i in chosen]
    if initials != acr:
        return None

    full = " ".join(preceding_words[chosen[0]:]).strip()
    return full.strip(" ,;:\"'`()*_#") or None


def extract_from_text(content: str) -> dict[str, str]:
    """Find all `Full Term ("ABBR")` definitions in one document."""
    result: dict[str, str] = {}
    for m in ACRONYM_RE.finditer(content):
        acronym = m.group(1)
        # Real acronyms carry uppercase; this drops "(see)", "(the)", "(net)".
        if sum(c.isupper() for c in acronym) < 2:
            continue
        preceding = content[max(0, m.start() - 200):m.start()]
        words = preceding.split()[-15:]  # small window before the parenthesis
        full = find_full_term(words, acronym)
        if full and acronym not in result:
            result[acronym] = full
    return result


def main() -> None:
    sha1_list = [rec["sha1"] for rec in json.loads(SUBSET_PATH.read_text())]

    abbreviations: dict[str, dict[str, str]] = {}
    for sha1 in sha1_list:
        md_path = MD_DIR / sha1 / f"{sha1}.md"
        if not md_path.exists():
            print(f"  [skip] missing markdown for {sha1}")
            continue
        abbreviations[sha1] = extract_from_text(md_path.read_text(encoding="utf-8"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(abbreviations, indent=2, ensure_ascii=False))

    total = sum(len(v) for v in abbreviations.values())
    docs_with = sum(1 for v in abbreviations.values() if v)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  docs processed: {len(abbreviations)}  |  with >=1 abbr: {docs_with}")
    print(f"  total abbreviations: {total}  |  avg/doc: {total / max(1, len(abbreviations)):.1f}")


if __name__ == "__main__":
    main()
