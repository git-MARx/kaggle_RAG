"""validate_abbr.py — Pull out abbreviation entries that contain a symbol.

Reads data/abbreviations.json ({sha1: {abbr: full}}) and writes a flat
{abbr: full} JSON of every entry whose abbreviation OR its full name contains a
symbol (any non-alphanumeric, non-space character). Handy for eyeballing
potentially noisy extractions (e.g. stray punctuation, markdown leftovers).
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1] / "data"
IN_PATH = DATA_DIR / "abbreviations.json"
OUT_PATH = DATA_DIR / "abbr_with_symbols.json"

# A "symbol" = anything that is NOT a letter, digit, whitespace, or comma.
# (Comma is allowed — it's normal in full names like "Diversity, Equity, and Inclusion".)
SYMBOL_RE = re.compile(r"[^A-Za-z0-9\s,.&'-]")


def has_symbol(text: str) -> bool:
    return bool(SYMBOL_RE.search(text))


def main() -> None:
    data = json.loads(IN_PATH.read_text())

    flagged: dict[str, str] = {}
    for entries in data.values():               # drop sha1, flatten
        for abbr, full in entries.items():
            if has_symbol(abbr) or has_symbol(full):
                flagged[abbr] = full             # duplicate abbr -> last wins

    OUT_PATH.write_text(json.dumps(flagged, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_PATH}")
    print(f"  flagged {len(flagged)} unique abbreviations containing a symbol")


if __name__ == "__main__":
    main()
