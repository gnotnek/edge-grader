"""Re-score finished grading runs from their raw predictions, the way SciEntsBank papers report:
accuracy and macro-F1 with a 95% range (bootstrap over questions), the 2-way score, and the
majority-class floor (SemEval-2013's simplest baseline). No model is run.

    uv run edge-grader-baseline-ci results/board/e*-jetson-test_u*-20261021-*/baseline-*.json
"""

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from . import metrics
from .data import question_id


def two_way(labels: list[str | None]) -> list[str | None]:
    return [None if x is None else ("correct" if x == "correct" else "incorrect") for x in labels]


def rescore(summary_path: Path) -> dict:
    run = json.loads(summary_path.read_text())
    raw = summary_path.parent / "raw" / f"{summary_path.stem}.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    gold, pred = [r["gold"] for r in rows], [r["pred"] for r in rows]
    groups = [question_id(r["id"]) for r in rows]
    three = metrics.score(gold, pred)
    # The raw file must be the run the summary describes.
    assert three["accuracy"] == run["metrics"]["accuracy"], summary_path
    # shortcut: the floor uses this split's most frequent label; SemEval took the training
    # set's. Both are "incorrect" on every SciEntsBank split (checked 2026-11-11).
    top = Counter(gold).most_common(1)[0][0]
    g2, p2 = two_way(gold), two_way(pred)
    return {
        "run": str(summary_path), "label": run["label"], "split": run["split"],
        "scheme": run["scheme"],
        "three_way": {**three, "ci95": metrics.ci95(gold, pred, groups)},
        "two_way": {**metrics.score(g2, p2), "ci95": metrics.ci95(g2, p2, groups)},
        "majority": {"label": top, **metrics.score(gold, [top] * len(gold)),
                     "two_way": metrics.score(g2, [two_way([top])[0]] * len(g2))},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run summary JSONs (raw/ beside them)")
    ap.add_argument("--out", default="results/ci")
    args = ap.parse_args(argv)
    runs = [rescore(p) for p in args.runs]
    now = datetime.now(UTC)
    out = Path(args.out) / f"baseline-ci-{now.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "time": now.isoformat(timespec="seconds"),
        "method": "95% percentile bootstrap, whole questions resampled (data.question_id), "
                  "2000 resamples, seed 0; unparsed replies count as wrong",
        "runs": runs}, indent=2))
    for r in runs:
        t, w, m = r["three_way"], r["two_way"], r["majority"]
        print(f"{r['label']:24} {r['split']:8} 3-way {t['accuracy']:.1%} {t['ci95']['accuracy']} "
              f"F1 {t['macro_f1']:.3f} {t['ci95']['macro_f1']} | 2-way {w['accuracy']:.1%} "
              f"{w['ci95']['accuracy']} | majority {m['accuracy']:.1%} F1 {m['macro_f1']:.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
