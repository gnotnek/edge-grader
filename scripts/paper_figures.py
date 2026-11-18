"""Figures for the manuscript (manuscript/main.tex), drawn from committed comparison JSON only.

    uv run --with matplotlib python scripts/paper_figures.py

Fig. accuracy vs energy per answer on the Jetson, test_uq and test_ud, every grader in
results/estimates/compare-jetson-scientsbank-20261116-052730.json (docs/results/jetson-graders.md).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

SRC = Path("results/estimates/compare-jetson-scientsbank-20261116-052730.json")
OUT = Path("docs/results/figures/jetson-three-graders.pdf")
# The three graders take categorical slots 1-3 (dataviz reference palette; 3 is the all-pairs cap for a scatter).
HERO = {"E2B tweaked (Oct 28 adapter)": ("E2B tweaked", "#2a78d6"),
        "E4B untouched": ("E4B untouched", "#eb6834"),
        "E2B untouched": ("E2B untouched", "#1baf7a")}
REF = "E4B Oct 28 adapter"
# Other E2B versions far enough from the adapter cluster to name; the rest stay unlabelled grey.
SHORT = {"E2B reason-first": "reason-first", "E2B adapter + kNN examples": "adapter + examples",
         "E2B + kNN examples": "untouched + examples"}
INK, MUTED, GREY, GRID = "#0b0b0b", "#52514e", "#a8a7a2", "#e4e3df"


def main() -> None:
    rows = json.loads(SRC.read_text())["rows"]
    for split in ("test_uq", "test_ud"):  # self-check: every hero and the reference ran on both splits
        names = {r["name"] for r in rows if r["split"] == split}
        assert set(HERO) | {REF} <= names, (split, set(HERO) | {REF} - names)

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True)
    for ax, split, title in zip(axes, ("test_uq", "test_ud"), ("Unseen questions (test_uq)", "Unseen domains (test_ud)")):
        for r in (r for r in rows if r["split"] == split):
            x, y, name = r["j_per_answer"], r["accuracy"] * 100, r["name"]
            if name in HERO:
                label, color = HERO[name]
                ax.scatter(x, y, s=60, color=color, edgecolor="white", lw=1.5, zorder=4)
                left = label != "E2B untouched"  # left of the point: clear of the grey versions
                ax.annotate(label, (x, y), xytext=(-6 if left else 6, -3), ha="right" if left else "left",
                            textcoords="offset points", fontsize=8, color=INK, weight="bold")
            elif name == REF:
                ax.scatter(x, y, s=40, marker="D", facecolor="white", edgecolor=MUTED, lw=1.2, zorder=3)
                ax.annotate("E4B tweaked", (x, y), xytext=(6, 2), textcoords="offset points", fontsize=7, color=MUTED)
            else:
                ax.scatter(x, y, s=18, color=GREY, edgecolor="white", lw=0.8, zorder=2)
                if name in SHORT:
                    above = name == "E2B + kNN examples"
                    ax.annotate(SHORT[name], (x, y), xytext=(0, 5 if above else -9), textcoords="offset points",
                                fontsize=6.5, color=MUTED, ha="center")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([3, 4, 6, 10, 20, 40]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["3", "4", "6", "10", "20", "40"])
        ax.set_xlim(2.4, 55)
        ax.set_ylim(55, 83)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Energy per answer (J, log scale)")
    axes[0].set_ylabel("3-way accuracy (%)")
    handles = [Line2D([], [], ls="", marker="o", ms=6, color=c, label=l) for l, c in HERO.values()]
    handles += [Line2D([], [], ls="", marker="D", ms=5, mfc="white", mec=MUTED, label="E4B tweaked (reference)"),
                Line2D([], [], ls="", marker="o", ms=4, color=GREY, label="other E2B versions")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    print(OUT)


if __name__ == "__main__":
    main()
