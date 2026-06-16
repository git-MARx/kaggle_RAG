"""run.py — Run the pipeline over questions.json and write typed answers.

Usage:
    python run.py                 # all questions
    python run.py --limit 5       # first 5 (smoke test)
    python run.py --out answers.json
"""

import argparse
import json

import config
from pipeline.graph import app


def answer_one(q: dict) -> dict:
    state = {"question": q["text"], "kind": q["kind"]}
    result = app.invoke(state)
    return {"text": q["text"], "kind": q["kind"], "answer": result.get("final", "N/A")}


def main(limit: int | None, out: str) -> None:
    questions = json.loads(config.QUESTIONS_PATH.read_text())
    if limit:
        questions = questions[:limit]

    answers = []
    for i, q in enumerate(questions):
        a = answer_one(q)
        answers.append(a)
        print(f"[{i + 1}/{len(questions)}] ({q['kind']}) {q['text'][:60]}...  -> {a['answer']}")

    (config.PROJECT_ROOT / out).write_text(json.dumps(answers, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(answers)} answers -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="answers.json")
    args = ap.parse_args()
    main(args.limit, args.out)
