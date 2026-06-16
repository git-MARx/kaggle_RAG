"""evaluate.py — Score the pipeline against the hand-verified dev set.

Dev set: data/dev_set_verified.json (q_index references questions.json).
Comparison is lenient + honest: boolean/N-A auto-judged; numbers matched by the
digits they contain; names by token overlap. Anything unclear is flagged REVIEW.

Usage: python evaluate.py
"""

import json
import re

import config
from pipeline.graph import app


def _nums(s) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?[\d,]*\.?\d+", str(s))]


def judge(kind: str, predicted, expected) -> str:
    exp = str(expected)
    # abstention
    if exp.strip().upper().startswith("N/A") or exp.strip() == "N/A":
        return "PASS" if predicted == "N/A" else "FAIL"
    if kind == "boolean":
        want = "true" in exp.lower()
        return "PASS" if bool(predicted) is want else "FAIL"
    if kind == "number":
        if predicted == "N/A":
            return "FAIL"
        try:
            p = float(predicted)
        except (TypeError, ValueError):
            return "REVIEW"
        cands = _nums(expected)
        if not cands:
            return "REVIEW"
        target = max(cands, key=abs)        # the full/canonical value in expected
        if target == 0:
            return "PASS" if p == 0 else "FAIL"
        return "PASS" if abs(p - target) / abs(target) < 0.01 else "FAIL"
    # name / names -> token overlap, flag for eyeballing
    pred_txt = " ".join(predicted) if isinstance(predicted, list) else str(predicted)
    overlap = set(re.findall(r"[a-z]+", pred_txt.lower())) & set(re.findall(r"[a-z]+", exp.lower()))
    return "REVIEW" if overlap else "FAIL"


def main() -> None:
    questions = json.loads(config.QUESTIONS_PATH.read_text())
    dev = json.loads(config.DEVSET_PATH.read_text())["items"]

    rows, counts = [], {"PASS": 0, "FAIL": 0, "REVIEW": 0}
    for item in dev:
        q = questions[item["q_index"]]
        predicted = app.invoke({"question": q["text"], "kind": q["kind"]}).get("final", "N/A")
        verdict = judge(q["kind"], predicted, item["answer"])
        counts[verdict] += 1
        rows.append((item["q_index"], q["kind"], verdict, predicted, item["answer"]))

    print(f"{'idx':>4} {'kind':<8} {'verdict':<7} {'predicted':<22} expected")
    print("-" * 90)
    for idx, kind, verdict, pred, exp in rows:
        print(f"{idx:>4} {kind:<8} {verdict:<7} {str(pred)[:22]:<22} {str(exp)[:40]}")
    print("-" * 90)
    print(f"PASS={counts['PASS']}  FAIL={counts['FAIL']}  REVIEW={counts['REVIEW']}  (of {len(dev)})")


if __name__ == "__main__":
    main()
