"""Cascade and defer-to-teacher, simulated from Jetson runs that already exist (no new board time).

    uv run python -m edge_grader.cascade

Every version here was run on the whole split on the same board in the same clock state (GPU max
612 MHz, before the 10-13 → 10-19 change), so each answer's label and cost under every version are
known. A policy only chooses which version's label to keep:

- cascade: grade with E2B and E4B; where they agree keep that label, where they don't take the
  thinking-on label (E2B's or E4B's). Cost = both baselines on every answer + thinking on the
  answers that disagreed.
- defer: the same agreement check, but disagreements go back to the teacher. Reported as accuracy
  on the answers kept against coverage (the share kept), beside plain accuracy, never in place of it.

Agreement has no threshold, so nothing is fitted on the splits it is reported on. A run's energy is
measured over its whole grading window; an answer's share is apportioned by its latency
(shortcut: assumes power is flat within a run; per-answer power needs per-request telemetry).
"""

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from .board_check import mcnemar, read_answers, read_run
from .compare import holm

BOARD = Path("results/board")
# The 612 MHz set: baseline (phase 0 / stage 1) and thinking on (B1), same flags otherwise.
# test_ua's baselines are from 2026-09-30 (stage 1): same server flags, no board-state.txt yet.
RUNS = {
    "test_ua": {"e2b": "e2b-jetson-test_ua-20260930-224702",
                "e4b": "e4b-jetson-test_ua-20260930-225337",
                "think-e2b": "think-e2b-jetson-test_ua-20261011-232209",
                "think-e4b": "think-e4b-jetson-test_ua-20261012-012827"},
    "test_uq": {"e2b": "e2b-jetson-test_uq-20261009-010217",
                "e4b": "e4b-jetson-test_uq-20261009-070836",
                "think-e2b": "think-e2b-jetson-test_uq-20261012-052845",
                "think-e4b": "think-e4b-jetson-test_uq-20261012-081640"},
    "test_ud": {"e2b": "e2b-jetson-test_ud-20261009-011158",
                "e4b": "e4b-jetson-test_ud-20261009-072141",
                "think-e2b": "think-e2b-jetson-test_ud-20261012-132931",
                "think-e4b": "think-e4b-jetson-test_ud-20261013-160350"},
}


def load(run_dir: Path) -> tuple[dict[str, dict], float]:
    """Answers by id, each with its apportioned energy "j", and the run's measured J per answer."""
    run = read_run(run_dir)
    rows = read_answers(run["_summaries"][0])
    total = run["j_per_answer"] * len(rows)
    lat = sum(r["latency_s"] for r in rows.values())
    for r in rows.values():
        r["j"] = total * r["latency_s"] / lat
    return rows, run["j_per_answer"]


def point(name: str, chosen: dict[str, str], runs: dict[str, dict[str, dict]],
          used: dict[str, list[str]]) -> dict:
    """chosen: id -> version whose label is kept (or "teacher"); used: id -> versions run on it."""
    ids = sorted(chosen)
    kept = [i for i in ids if chosen[i] != "teacher"]
    right = {i: runs[chosen[i]][i]["pred"] == runs[chosen[i]][i]["gold"] for i in kept}
    joules = [sum(runs[v][i]["j"] for v in used[i]) for i in ids]
    secs = [sum(runs[v][i]["latency_s"] for v in used[i]) for i in ids]
    return {
        "policy": name,
        "n": len(ids),
        "coverage": round(len(kept) / len(ids), 4),
        "accuracy": round(sum(right.values()) / len(ids), 4) if len(kept) == len(ids) else None,
        "accuracy_on_kept": round(sum(right.values()) / len(kept), 4) if kept else None,
        "escalated": round(sum(len(used[i]) > 2 for i in ids) / len(ids), 4),
        "j_per_answer": round(statistics.fmean(joules), 2),
        "mean_latency_s": round(statistics.fmean(secs), 2),
        "median_latency_s": round(statistics.median(secs), 2),
        "_right": right,
    }


def policies(runs: dict[str, dict[str, dict]]) -> list[dict]:
    ids = sorted(runs["e2b"])
    for v in runs.values():
        assert sorted(v) == ids, "every version must have graded the same answers"
    agree = {i: runs["e2b"][i]["pred"] == runs["e4b"][i]["pred"] for i in ids}
    out = [point(v, dict.fromkeys(ids, v), runs, {i: [v] for i in ids}) for v in runs]
    for think in ("think-e2b", "think-e4b"):
        out.append(point(f"cascade e2b+e4b → {think}",
                         {i: "e4b" if agree[i] else think for i in ids}, runs,
                         {i: ["e2b", "e4b"] + ([] if agree[i] else [think]) for i in ids}))
    out.append(point("defer e2b+e4b → teacher", {i: "e4b" if agree[i] else "teacher" for i in ids},
                     runs, {i: ["e2b", "e4b"] for i in ids}))
    return out


def front(points: list[dict]) -> set[str]:
    """Policies with full coverage that no other beats on both accuracy and J per answer."""
    full = [p for p in points if p["accuracy"] is not None]
    return {p["policy"] for p in full
            if not any(q["accuracy"] >= p["accuracy"] and q["j_per_answer"] <= p["j_per_answer"]
                       and (q["accuracy"], q["j_per_answer"]) != (p["accuracy"], p["j_per_answer"])
                       for q in full)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="results/cascade")
    args = ap.parse_args(argv)

    result, tests = {}, []
    for split, dirs in RUNS.items():
        loaded = {v: load(BOARD / d) for v, d in dirs.items()}
        runs = {v: rows for v, (rows, _) in loaded.items()}
        pts = policies(runs)
        on_front = front(pts)
        base = next(p for p in pts if p["policy"] == "e4b")["_right"]
        for p in pts:
            p["on_front"] = p["policy"] in on_front
            if p["policy"].startswith("cascade"):  # paired against the plain E4B baseline
                ids = sorted(base)
                p["vs_e4b"] = mcnemar([p["_right"][i] for i in ids], [base[i] for i in ids])
                tests.append(p["vs_e4b"])
        result[split] = {"runs": dirs, "measured_j_per_answer": {v: j for v, (_, j) in loaded.items()},
                         "points": pts}
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = round(ph, 3)

    now = datetime.now(UTC)
    for s in result.values():
        for p in s["points"]:
            del p["_right"]
    run = {"kind": "cascade-simulation", "board": "jetson", "clock_state": "gpu max 612 MHz",
           "energy_per_answer": "run's measured energy apportioned by latency",
           "time": now.isoformat(timespec="seconds"), "splits": result}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cascade-jetson-{now:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")

    for split, s in result.items():
        print(f"\n{split}")
        for p in s["points"]:
            acc = p["accuracy"] if p["accuracy"] is not None else p["accuracy_on_kept"]
            extra = f"  p_holm {p['vs_e4b']['p_holm']}" if "vs_e4b" in p else ""
            print(f"  {p['policy']:28} acc {acc:.3f}  cover {p['coverage']:.2f}  esc {p['escalated']:.2f}"
                  f"  {p['j_per_answer']:7.2f} J  {p['mean_latency_s']:6.2f} s"
                  f"{'  front' if p['on_front'] else ''}{extra}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
