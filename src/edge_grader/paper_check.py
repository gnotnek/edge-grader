"""Check the manuscript's tables against the committed runs.

    uv run python -m edge_grader.paper_check            # every table
    uv run python -m edge_grader.paper_check --table tests

Each table is recomputed from results/board/ (each run's summary and per-answer replies) and the
board-check files in results/estimates/ (energy and latency from the board's telemetry), with the
same code that wrote the comparison files (compare_jetson, compare_thinking), and nothing is
written. Every cell of manuscript/main.tex's tables is printed beside the recomputed value. Exit
status 1 if any differs. No GPU, no network, no models.
"""

import argparse
import re
import sys
from pathlib import Path

from . import compare_jetson, compare_thinking

TEX = Path("manuscript/main.tex")
SPLITS = ("test_uq", "test_ud")


def table_rows(tex: str, label: str) -> list[list[str]]:
    """The body rows of the table labelled `label`, each a list of cleaned cells."""
    start = tex.index(f"\\label{{tab:{label}}}")
    body = tex[start:tex.index("\\end{tabular}", start)]
    body = body.split("\\midrule", 1)[1] if "\\midrule" in body else body
    rows = []
    for line in body.split("\\\\"):
        line = re.sub(r"\\(midrule|bottomrule|toprule)", "", line).strip()
        if line:
            rows.append([clean(c) for c in line.split("&")])
    return rows


def clean(cell: str) -> str:
    """LaTeX cell -> plain text: no \\textbf/\\textit wrappers, footnote marks or thousands braces."""
    cell = re.sub(r"\\text(bf|it)\{([^{}]*)\}", r"\2", cell)
    cell = cell.replace("$^\\ast$", "").replace("{,}", ",").replace("\\_", "_")
    return " ".join(cell.split())


def is_bold(tex_cell: str) -> bool:
    return "\\textbf{" in tex_cell


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def span(values: list[float], digits: int | None = None) -> str:
    """A range as the paper writes it: "lo--hi", or one number when both ends round the same.
    Energy above 100 J is given in whole joules."""
    if digits is None:
        digits = 0 if min(values) >= 100 else 1
    lo, hi = f"{min(values):.{digits}f}", f"{max(values):.{digits}f}"
    return lo if lo == hi else f"{lo}--{hi}"


def acc_ci(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{pct(r['accuracy'])} ({pct(lo)}--{pct(hi)})"


THREE = {"E2B untouched": "e2b", "E2B tweaked": "ft-e2b", "E4B untouched": "e4b",
         "E4B tweaked (reference)": "ft-e4b"}
TESTS = {"E2B tweaked vs E2B untouched": ("e2b", "ft-e2b"), "E2B tweaked vs E4B untouched": ("e4b", "ft-e2b"),
         "E4B untouched vs E2B untouched": ("e2b", "e4b"), "E4B tweaked vs E4B untouched": ("e4b", "ft-e4b")}
VARIANTS = {"untouched": ["e2b"], "adapter (E2B tweaked)": ["ft-e2b"],
            "seeds 0--4 (range)": [f"ft-s{n}-e2b" for n in range(5)], "soup of seeds 0--4": ["ft-soup-e2b"],
            "5-way labels": ["ft-5w-s0-e2b"], "label-first reason": ["ft-lf-e2b"], "reason-first": ["ft-rf-e2b"],
            "adapter + nearest examples": ["ex4-ft-e2b"], "untouched + nearest examples": ["ex4-e2b"],
            "untouched + random examples": ["rand4-e2b"]}
THINKING = {("untouched", "off"): "e2b", ("untouched", "on"): "think-e2b",
            ("tweaked", "off"): "ft-e2b", ("tweaked", "on"): "ft-think-e2b"}


def expected(tex: str, jetson: dict, thinking: dict) -> dict[str, list[tuple[str, str, str]]]:
    """Per table, (what, paper, here) for every checked cell."""
    by = {(r["version"], r["split"]): r for r in jetson["rows"]}
    tby = {(r["version"], r["split"]): r for r in thinking["rows"]}
    out: dict[str, list[tuple[str, str, str]]] = {}

    rows = out["three"] = []
    for cells in table_rows(tex, "three"):
        v = THREE[cells[0]]
        uq, ud = by[v, "test_uq"], by[v, "test_ud"]
        here = [acc_ci(uq), acc_ci(ud), f"{uq['macro_f1']:.3f} / {ud['macro_f1']:.3f}",
                span([uq["median_latency_s"], ud["median_latency_s"]], 2),
                span([uq["j_per_answer"], ud["j_per_answer"]])]
        heads = ["test_uq acc (95%)", "test_ud acc (95%)", "macro-F1 uq / ud", "median latency s", "J/answer"]
        rows += [(f"{cells[0]}: {h}", p, e) for h, p, e in zip(heads, cells[1:], here, strict=True)]

    rows = out["tests"] = []
    tests = {(t["first"], t["second"], t["split"]): t for t in jetson["tests"]}
    for cells in table_rows(tex, "tests"):
        first, second = TESTS[cells[0]]
        for split, paper in zip(SPLITS, cells[1:], strict=True):
            t = tests[first, second, split]
            rows.append((f"{cells[0]}: {split}", paper,
                         f"{100 * t['diff']:+.1f} ({t['only_second_right']}/{t['only_first_right']})"))
    stated = re.search(r"All eight tests: corrected \$p\\le([\d.]+)\$", tex).group(1)
    worst = max(t["p_holm"] for t in jetson["tests"])
    rows.append(("largest corrected p of the eight, at most", f"p <= {stated}",
                 f"p <= {stated}" if worst <= float(stated) else f"p = {worst:.2g}"))

    rows = out["variants"] = []
    for cells in table_rows(tex, "variants"):
        vs = VARIANTS[cells[0]]
        here = [span([by[v, s]["accuracy"] * 100 for v in vs]) for s in SPLITS]
        here.append(span([by[v, s]["j_per_answer"] for v in vs for s in SPLITS]))
        rows += [(f"{cells[0]}: {h}", p, e) for h, p, e in zip((*SPLITS, "J/answer"), cells[1:], here, strict=True)]

    rows = out["thinking"] = []
    ttests = {(t["first"], t["second"], t["split"]): t for t in thinking["tests"]}
    raw = [line.split("&") for line in tex[tex.index("\\label{tab:thinking}"):].split("\\end{tabular}")[0]
           .split("\\midrule", 1)[1].split("\\\\") if "&" in line]
    for cells, raw_cells in zip(table_rows(tex, "thinking"), raw, strict=True):
        v = THINKING[cells[0], cells[1]]
        here = [pct(tby[v, s]["accuracy"]) for s in SPLITS]
        here.append(span([tby[v, s]["j_per_answer"] for s in SPLITS]))
        name = f"{cells[0]}, thinking {cells[1]}"
        rows += [(f"{name}: {h}", p, e) for h, p, e in zip((*SPLITS, "J/answer"), cells[2:], here, strict=True)]
        if cells[1] == "on":  # bold = significantly different from thinking off (Holm, 22 tests)
            off = THINKING[cells[0], "off"]
            for s, raw_cell in zip(SPLITS, raw_cells[2:4], strict=True):
                sig = ttests[off, v, s]["p_holm"] < 0.05
                rows.append((f"{name}: {s} bold (significant)", str(is_bold(raw_cell)), str(sig)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--table", choices=["three", "tests", "variants", "thinking"])
    ap.add_argument("--tex", type=Path, default=TEX)
    args = ap.parse_args(argv)
    tables = expected(args.tex.read_text(), compare_jetson.compute(), compare_thinking.compute())
    titles = {"three": "Table II, the three graders", "tests": "Table III, paired comparisons",
              "variants": "Table IV, every fine-tuned E2B version", "thinking": "Table V, thinking against fine-tuning"}
    bad = 0
    for name, rows in tables.items():
        if args.table and name != args.table:
            continue
        print(f"\n{titles[name]}")
        for what, paper, here in rows:
            ok = paper == here
            bad += not ok
            print(f"  {what:<52} paper {paper:<22} here {here:<22} {'ok' if ok else 'DIFFERS'}")
    print(f"\n{'every checked number matches the manuscript' if not bad else f'{bad} cells differ'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
