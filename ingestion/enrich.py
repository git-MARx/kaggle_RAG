"""enrich.py — Inline-enrich text with abbreviation expansions.

STRATEGY: "embedded only".
The enriched text is what gets EMBEDDED / INDEXED (so BM25 + the embedding model
match either "CTC" or "cost to company"). The clean original is kept as the chunk
content shown to the LLM. So this module only provides the enrichment function —
ingest.py does:  embed(enrich_text(chunk))  but stores the original chunk text.

Turns:  "CTC: 1,234"  ->  "CTC (cost to company): 1,234"
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1] / "data"
ABBR_PATH = DATA_DIR / "abbreviations.json"

# Chars that count as "part of a token". Used in look-arounds so an abbreviation
# only matches when standalone — not inside a larger token, and not as a fragment
# split off an acronym that contains . & / - (e.g. "SG" inside "SG&A").
_TOKEN = r"A-Za-z0-9&./\-"


def load_abbreviations(path: Path = ABBR_PATH) -> dict[str, dict[str, str]]:
    """Load {sha1: {abbr: full_term}} produced by extract_abbreviations.py."""
    return json.loads(Path(path).read_text())


def _pattern(abbr: str) -> re.Pattern:
    # Case-sensitive (no IGNORECASE) so the pronoun "it" never matches "IT".
    # Look-arounds = "not glued to another token char" on either side; handles
    # acronyms ending in punctuation (U.S.) that plain \b can't.
    return re.compile(rf"(?<![{_TOKEN}]){re.escape(abbr)}(?![{_TOKEN}])")


def enrich_text(text: str, abbr_map: dict[str, str]) -> str:
    """Insert 'ABBR (full term)' at every standalone occurrence of each abbr.

    This is the text to EMBED — not to show the LLM.
    """
    # Longest abbreviations first, so "SG&A" is enriched before a bare "SG".
    for abbr in sorted(abbr_map, key=len, reverse=True):
        full = abbr_map[abbr]

        def repl(m: re.Match, full: str = full) -> str:
            # Skip if the full term already sits next to the abbr on EITHER side:
            #   "CTC (cost to company)"  -> full after
            #   "Cost to Company (CTC)"  -> full before (the definition site)
            span = len(full) + 4
            before = m.string[max(0, m.start() - span):m.start()].lower()
            after = m.string[m.end():m.end() + span].lower()
            if full.lower() in before or full.lower() in after:
                return m.group(0)
            return f"{m.group(0)} ({full})"

        text = _pattern(abbr).sub(repl, text)
    return text


if __name__ == "__main__":
    abbrs = load_abbreviations()
    sha1 = next(iter(abbrs))
    md = (DATA_DIR / "EnterpriseRAG_2025_02_markdown" / sha1 / f"{sha1}.md").read_text()
    enriched = enrich_text(md, abbrs[sha1])

    print(f"doc {sha1[:10]}: {len(abbrs[sha1])} abbrs | "
          f"{len(md):,} -> {len(enriched):,} chars (+{len(enriched) - len(md):,})")
    for abbr, full in abbrs[sha1].items():
        i = enriched.find(f"{abbr} ({full})")
        if i != -1:
            print("  example:", repr(enriched[i:i + len(abbr) + len(full) + 4]))
            break
