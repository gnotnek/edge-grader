"""Figures for the manuscript (manuscript/main.tex), drawn from committed comparison JSON only.

    uv run --with matplotlib python scripts/paper_figures.py

Fig. accuracy vs energy per answer on the Jetson, test_uq and test_ud, every grader in
results/estimates/compare-jetson-scientsbank-20261116-052730.json (docs/results/jetson-graders.md).
Fig. energy per answer and energy per extra correct answer, from the same file plus the thinking-on
rows of results/estimates/compare-thinking-scientsbank-20261120-142118.json (higher clocks, marked).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

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


THINK = Path("results/estimates/compare-thinking-scientsbank-20261120-142118.json")
OUT_E = Path("docs/results/figures/jetson-energy.pdf")
# Row label -> (name in its JSON, from the thinking file?, colour). Order = how the bars are drawn.
ENERGY_ROWS = [
    ("E2B untouched", "E2B untouched", False, "#1baf7a"),
    ("E2B tweaked (adapter)", "E2B tweaked (Oct 28 adapter)", False, "#2a78d6"),
    ("E2B soup of 5 seeds", "E2B soup (s0-s4)", False, GREY),
    ("E2B label-first reason", "E2B label-first", False, GREY),
    ("E2B 5-way labels", "E2B 5-way s0", False, GREY),
    ("E4B untouched", "E4B untouched", False, "#eb6834"),
    ("E2B + nearest examples", "E2B + kNN examples", False, GREY),
    ("E4B tweaked (adapter)", "E4B Oct 28 adapter", False, MUTED),
    ("E2B + random examples", "E2B + random examples", False, GREY),
    ("E2B adapter + examples", "E2B adapter + kNN examples", False, GREY),
    ("E2B reason-first", "E2B reason-first", False, GREY),
    ("E2B untouched, thinking on*", "E2B untouched", True, GREY),
    ("E2B tweaked, thinking on*", "Oct 28 adapter", True, GREY),
]


def energy() -> None:
    """Two panels: J per answer, and extra J per extra correct answer against untouched E2B."""
    off = {(r["name"], r["split"]): r for r in json.loads(SRC.read_text())["rows"]}
    on = {(r["name"], r["split"]): r for r in json.loads(THINK.read_text())["rows"] if r["thinking"]}
    base = {sp: off[("E2B untouched", sp)] for sp in ("test_uq", "test_ud")}
    rows = []
    for label, name, think, colour in ENERGY_ROWS:
        rs = [(on if think else off)[(name, sp)] for sp in ("test_uq", "test_ud")]
        j = [r["j_per_answer"] for r in rs]
        per = [(r["j_per_answer"] - base[r["split"]]["j_per_answer"]) / (r["accuracy"] - base[r["split"]]["accuracy"])
               for r in rs] if name != "E2B untouched" or think else None
        rows.append((label, j, per, think, colour))
    # self-check against the paper: adapter 9-11 J and E4B 31-41 J per extra correct answer
    per = {r[0]: r[2] for r in rows}
    assert [round(x) for x in per["E2B tweaked (adapter)"]] == [9, 11], per["E2B tweaked (adapter)"]
    assert sorted(round(x) for x in per["E4B untouched"]) == [31, 41], per["E4B untouched"]

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.3), sharey=True)
    ys = range(len(rows))[::-1]
    for y, (label, j, p, think, colour) in zip(ys, rows):
        mean = sum(j) / 2
        a.barh(y, mean, height=0.62, color=colour, hatch="////" if think else None,
               edgecolor="white" if not think else MUTED, lw=0.6)
        a.text(mean * 1.08, y, f"{min(j):.1f}\u2013{max(j):.1f} J" if max(j) - min(j) >= 0.1 else f"{mean:.1f} J",
               va="center", fontsize=6.5, color=INK)
        if p:
            b.plot(p, [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
            b.scatter(p[0], y, s=22, marker="o", color=colour if colour != GREY else MUTED, zorder=3)
            b.scatter(p[1], y, s=22, marker="s", color=colour if colour != GREY else MUTED, zorder=3)
            b.text(max(p) * 1.4, y, f"{min(p):,.0f}\u2013{max(p):,.0f} J", va="center", fontsize=6.5, color=INK)
        else:
            b.text(1.2, y, "baseline", va="center", fontsize=6.5, color=MUTED)
    a.set_yticks(list(ys), [r[0] for r in rows])
    for ax, title, xlabel, lim in ((a, "(a) Energy per graded answer", "J per answer (log scale)", (2.5, 600)),
                                   (b, "(b) Price of each accuracy gain",
                                    "Extra J per extra correct answer (log scale)", (1, 12000))):
        ax.set_xscale("log")
        ax.set_xlim(*lim)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    b.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, color=MUTED, label="test_uq"),
                      Line2D([], [], ls="", marker="s", ms=4, color=MUTED, label="test_ud")],
             loc="upper right", frameon=False, fontsize=7)
    fig.text(0.01, 0.01, "*Thinking-on runs at higher GPU clocks (1,020 / 918 MHz) than the others (612 MHz); their cost is approximate.",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_E)
    fig.savefig(OUT_E.with_suffix(".png"), dpi=200)
    print(OUT_E)


if __name__ == "__main__":
    main()
    energy()
