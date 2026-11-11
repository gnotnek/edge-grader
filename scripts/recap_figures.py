"""Tables and figures for docs/results/recap.md: every grading run, 3-way and 2-way.

    uv run --with matplotlib python scripts/recap_figures.py > /tmp/recap-tables.md

2-way merges "contradictory" into "incorrect" (correct vs not-correct), from the same
per-answer predictions in raw/*.jsonl; nothing is re-run. Figures go to docs/results/figures/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edge_grader.board_check import read_answers
from edge_grader.compare import two
from edge_grader.metrics import score

R = Path("results")
FIG = Path("docs/results/figures")
SPLITS = ("test_ua", "test_uq", "test_ud")
C3, C2 = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (dataviz reference palette)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def pick(pattern: str) -> Path | None:
    """Newest whole-split run matching the glob (smoke runs of 100 skipped)."""
    files = [f for f in sorted(R.glob(pattern), key=lambda p: p.name[-20:])
             if "/raw/" not in str(f) and json.loads(f.read_text()).get("n_requested", 0) == 0]
    return files[-1] if files else None


def scores(path: Path | None) -> dict | None:
    if path is None:
        return None
    a = read_answers(path).values()
    g, p = [r["gold"] for r in a], [r["pred"] for r in a]
    s3, s2 = score(g, p), score([two(x) for x in g], [two(x) for x in p])
    summary = json.loads(path.read_text())["metrics"]["accuracy"]
    assert abs(s3["accuracy"] - summary) < 1e-3, path  # 3-way must match the run's own summary
    return {"n": s3["n"], "acc3": s3["accuracy"], "f1_3": s3["macro_f1"],
            "acc2": s2["accuracy"], "f1_2": s2["macro_f1"], "file": str(path)}


# (section, version, platform, glob with {m} model and {s} split)
SCIENTSBANK = [
    ("Untouched", "Untouched", "Kaggle T4", "kaggle/baseline-{m}-kaggle-t4-{s}-3way-*.json"),
    ("Untouched", "Untouched", "Jetson", "board/{m}-jetson-{s}-*/baseline-*.json"),
    ("Untouched", "Untouched", "Pi 5", "board/{m}-pi5-{s}-*/baseline-*.json"),
    ("Untouched", "little-gemma engine", "Jetson", "board/lg-{m}-jetson-{s}-*/baseline-*.json"),
    ("Untouched", "`--swa-full` prefix reuse", "Jetson", "board/swafull-{m}-jetson-{s}-*/baseline-*.json"),
    ("RAG", "RAG-en (top 3)", "Kaggle T4", "kaggle/rag-{m}-kaggle-t4-en-k3-{s}-3way-*.json"),
    ("RAG", "RAG-en (top 3)", "Jetson", "board/rag-{m}-jetson-{s}-*/rag-*.json"),
    ("RAG", "RAG-id (top 3)", "Kaggle T4", "kaggle/rag-{m}-kaggle-t4-id-k3-{s}-3way-*.json"),
    ("Thinking", "Thinking on (budget 320)", "Jetson", "board/think-{m}-jetson-{s}-*/think-*.json"),
    ("Thinking", "Thinking on (budget 320)", "Pi 5", "board/think-{m}-pi5-{s}-*/think-*.json"),
    ("Fine-tuned", "SciEntsBank adapter (Oct 28)", "Kaggle T4", "kaggle/baseline-{m}-kaggle-t4-ft-{s}-3way-*.json"),
    ("Fine-tuned", "SciEntsBank label-first", "Kaggle T4", "kaggle/baseline-{m}-kaggle-t4-ft-lf-{s}-3way-*.json"),
    ("Fine-tuned", "SciEntsBank reason-first", "Kaggle T4", "kaggle/baseline-{m}-kaggle-t4-ft-rf-{s}-3way-*.json"),
    ("Fine-tuned", "Beetle + SAF adapter", "Kaggle T4", "kaggle/baseline-{m}-kaggle-t4-ft-beetle-saf-{s}-3way-*.json"),
    ("Fine-tuned", "Beetle + SAF label-first", "Kaggle T4", "kaggle/baseline-{m}-kaggle-t4-ft-lf-beetle-saf-{s}-3way-*.json"),
    ("Fine-tuned", "Beetle + SAF reason-first", "Kaggle T4", "kaggle/baseline-{m}-kaggle-t4-ft-rf-beetle-saf-{s}-3way-*.json"),
]
HELD_OUT = [
    ("Untouched", "kaggle/{d}-baseline-{m}-kaggle-t4-{s}-3way-*.json"),
    ("SciEntsBank adapter (Oct 28)", "kaggle/{d}-baseline-{m}-kaggle-t4-ft-{s}-3way-*.json"),
    ("SciEntsBank label-first", "kaggle/{d}-baseline-{m}-kaggle-t4-ft-lf-{s}-3way-*.json"),
    ("SciEntsBank reason-first", "kaggle/{d}-baseline-{m}-kaggle-t4-ft-rf-{s}-3way-*.json"),
]


def cell(x: dict | None, k: str) -> str:
    if x is None:
        return "—"
    return f"{x['acc' + k]:.1%} ({x['f1_' + k]:.2f})".replace("%", "")


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def main() -> None:
    sb = {(m, v, p, s): scores(pick(g.format(m=m, s=s)))
          for m in ("e2b", "e4b") for _, v, p, g in SCIENTSBANK for s in SPLITS}
    ho = {(m, v, d, s): scores(pick(g.format(m=m, d=d, s=s)))
          for m in ("e2b", "e4b") for v, g in HELD_OUT for d in ("beetle", "saf")
          for s in ("test_ua", "test_uq")}

    hdr = ["Section", "Version", "Platform"] + [f"{s} {k}" for s in SPLITS for k in ("3-way", "2-way")]
    for m in ("e2b", "e4b"):
        rows = [[sec, v, p] + [cell(sb[m, v, p, s], k) for s in SPLITS for k in ("3", "2")]
                for sec, v, p, _ in SCIENTSBANK if any(sb[m, v, p, s] for s in SPLITS)]
        print(f"\n### SciEntsBank, {m.upper()}\n\n{table(hdr, rows)}")
    hdr = ["Version"] + [f"{d.title() if d == 'beetle' else 'SAF'} {s} {k}"
                         for d in ("beetle", "saf") for s in ("test_ua", "test_uq") for k in ("3-way", "2-way")]
    for m in ("e2b", "e4b"):
        rows = [[v] + [cell(ho[m, v, d, s], k) for d in ("beetle", "saf") for s in ("test_ua", "test_uq")
                       for k in ("3", "2")] for v, _ in HELD_OUT if any(ho[m, v, d, s] for d in ("beetle", "saf")
                                                                      for s in ("test_ua", "test_uq"))]
        print(f"\n### Beetle and SAF, {m.upper()}\n\n{table(hdr, rows)}")
    print("\n<!-- source files -->")
    for x in [*sb.values(), *ho.values()]:
        if x:
            print(f"<!-- {x['file']} n={x['n']} -->")

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK, "font.family": "sans-serif"})

    # Figure 1: every method on the held-out SciEntsBank splits.
    pick_rows = [("Untouched", "Untouched", "Kaggle T4"), ("RAG-en", "RAG-en (top 3)", "Kaggle T4"),
                 ("RAG-id", "RAG-id (top 3)", "Kaggle T4"),
                 ("Thinking on (Jetson)", "Thinking on (budget 320)", "Jetson"),
                 ("FT SciEntsBank (Oct 28)", "SciEntsBank adapter (Oct 28)", "Kaggle T4"),
                 ("FT SciEntsBank label-first", "SciEntsBank label-first", "Kaggle T4"),
                 ("FT SciEntsBank reason-first", "SciEntsBank reason-first", "Kaggle T4"),
                 ("FT Beetle+SAF", "Beetle + SAF adapter", "Kaggle T4"),
                 ("FT Beetle+SAF label-first", "Beetle + SAF label-first", "Kaggle T4"),
                 ("FT Beetle+SAF reason-first", "Beetle + SAF reason-first", "Kaggle T4")]
    panels = [(m, s) for m in ("e2b", "e4b") for s in ("test_uq", "test_ud")]
    titles = {"test_uq": "unseen questions (test_uq, n=733)", "test_ud": "unseen domains (test_ud, n=4,562)"}
    bars(FIG / "scientsbank-methods.png", panels, pick_rows,
         lambda m, s, v, p: sb[m, v, p, s], lambda m, s: f"{m.upper()}, {titles[s]}",
         "SciEntsBank accuracy by method: 3-way vs 2-way")

    # Figure 2: held-out datasets, adapters trained on SciEntsBank only.
    ho_rows = [(v, v) for v, _ in HELD_OUT]
    panels = [(m, d, s) for m in ("e2b", "e4b") for d, s in
              (("beetle", "test_ua"), ("beetle", "test_uq"), ("saf", "test_ua"), ("saf", "test_uq"))]
    bars(FIG / "beetle-saf-transfer.png", panels, [(lab, v, None) for lab, v in ho_rows],
         lambda m, d, s, v, p: ho[m, v, d, s],
         lambda m, d, s: f"{m.upper()}, {'Beetle' if d == 'beetle' else 'SAF'} {s}",
         "Held-out datasets (never trained on): 3-way vs 2-way", cols=4)

    # Figure 3: accuracy against energy on the Jetson, test_ua, 612 MHz runs (energies from summary.md).
    jet = [("E2B little-gemma", 1.96, "board/lg-e2b-jetson-test_ua-*/baseline-*.json"),
           ("E2B --swa-full", 2.30, "board/swafull-e2b-jetson-test_ua-*/baseline-*.json"),
           ("E2B llama.cpp", 3.95, "board/e2b-jetson-test_ua-20260930-*/baseline-*.json"),
           ("E4B --swa-full", 4.00, "board/swafull-e4b-jetson-test_ua-*/baseline-*.json"),
           ("E4B llama.cpp", 7.12, "board/e4b-jetson-test_ua-20260930-*/baseline-*.json"),
           ("E2B thinking", 139, "board/think-e2b-jetson-test_ua-*/think-*.json"),
           ("E4B thinking", 280, "board/think-e4b-jetson-test_ua-*/think-*.json")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (name, joules, g) in enumerate(jet):
        x = scores(pick(g))
        a3, a2 = x["acc3"] * 100, x["acc2"] * 100
        ax.plot([a3, a2], [i, i], color=GRID, lw=3, zorder=1)
        ax.scatter([a3], [i], s=70, color=C3, edgecolor="white", lw=2, zorder=3)
        ax.scatter([a2], [i], s=70, color=C2, edgecolor="white", lw=2, zorder=3)
        ax.text(a3 - 0.6, i, f"{a3:.1f}", ha="right", va="center", fontsize=8, color=INK)
        ax.text(a2 + 0.6, i, f"{a2:.1f}", ha="left", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(jet)), [f"{n}  ·  {j:g} J" for n, j, _ in jet])
    ax.invert_yaxis()
    ax.set_xlim(50, 82)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Jetson, SciEntsBank test_ua: versions ordered by energy per answer", loc="left", color=INK)
    style(ax, "x")
    ax.legend(handles=legend(), frameon=False, loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "jetson-accuracy-energy.png", dpi=160)


def legend():
    from matplotlib.patches import Patch
    return [Patch(color=C3, label="3-way (correct / contradictory / incorrect)"),
            Patch(color=C2, label="2-way (correct / not correct)")]


def style(ax, grid_axis: str) -> None:
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def bars(out, panels, rows, get, title, suptitle, cols=2):
    n = len(panels)
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(5.2 * cols if cols == 2 else 4.2 * cols,
                             0.42 * len(rows) * ((n + cols - 1) // cols) + 1.6), sharex=True, squeeze=False)
    h = 0.38
    for ax, panel in zip(axes.flat, panels):
        for i, (label, v, p) in enumerate(rows):
            x = get(*panel, v, p)
            if x is None:
                ax.text(1, i, "not run", va="center", fontsize=8, color=MUTED)
                continue
            for off, k, c in ((-h / 2, "acc3", C3), (h / 2, "acc2", C2)):
                ax.barh(i + off, x[k] * 100, height=h - 0.04, color=c)
                ax.text(x[k] * 100 + 0.8, i + off, f"{x[k] * 100:.1f}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(title(*panel), loc="left", fontsize=10, color=INK)
        style(ax, "x")
        if ax not in axes[:, 0]:
            ax.tick_params(labelleft=False)
    for ax in axes[-1]:
        ax.set_xlabel("Accuracy (%)")
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, color=INK)
    fig.legend(handles=legend(), frameon=False, loc="upper right", ncol=2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    main()
