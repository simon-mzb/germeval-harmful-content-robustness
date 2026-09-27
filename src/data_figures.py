"""
data_figures.py -- the figures for the Data chapter (Section 3.1).

Separated from `data_profile.py` on purpose: that module measures and writes
`results/data_profile.json`, this one only reads that file and draws. A number
in the chapter therefore never depends on a plotting decision, and a figure can
be redrawn without recomputing anything.

Every figure here answers a question the chapter actually asks; a plot
that merely displays a column is left out. Style comes from `plotstyle.py`,
which fixes the chart-type policy for the whole thesis. An earlier exploratory
version of these figures predates that policy and broke three of its rules: it drew
label distributions as unsorted vertical bars, it gave the two editions
separate y-axes so their bars were not comparable, and it put the 2025 binary
and 2026 six-way VIO label spaces on one axis as if they were one scheme.

Usage
-----
python -m src.data_figures            # all figures
python -m src.data_figures --only labels provenance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.plotstyle import (COLOURS, FULL_WIDTH, GRID_COLOUR,
                           apply_thesis_style, save_figure)

_ROOT = Path(__file__).parent.parent
PROFILE_PATH = _ROOT / "results" / "data_profile.json"

SUBTASK_TITLE = {"c2a": "C2A (calls to action)", "dbo": "DBO (attacks on democracy)",
                 "vio": "VIO (violence)"}
EDITION_COLOUR = {"2025": COLOURS[0], "2026": COLOURS[1]}

# The injected cohort keeps one colour wherever it appears.
INJECTED_COLOUR = COLOURS[1]
GRID_GUIDE = GRID_COLOUR
CORPUS_COLOUR = COLOURS[0]


def _load() -> dict[str, Any]:
    if not PROFILE_PATH.exists():
        raise SystemExit("results/data_profile.json is missing; run src.data_profile first")
    return json.loads(PROFILE_PATH.read_text())


def _thousands(x: float) -> str:
    return f"{int(x):,}"


# ---------------------------------------------------------------------------

def figure_labels(prof: dict[str, Any]) -> Path:
    """
    Class distribution per subtask and edition.

    The classes this thesis is about span three orders of magnitude, so the
    axis has to be logarithmic: on a linear axis `glorification` at 27 items is
    a line of zero width beside `nothing` at 15,901. That rules out bars, whose
    length is read against a zero baseline that a log axis does not have, so
    the value is carried by the position of a dot and the guide line is only an
    aid to the eye. This is a deliberate departure from the chart-type policy's
    "horizontal bars" rule for label distributions, for that reason.

    VIO gets its own row rather than a shared axis because its label space is
    redesigned between the editions; drawing the binary and the six-way scheme
    against each other would invent a correspondence between them.
    """
    dist = prof["label_distributions"]
    panels = [("c2a", "2025"), ("c2a", "2026"),
              ("dbo", "2025"), ("dbo", "2026"),
              ("vio", "2025"), ("vio", "2026")]

    fig, axes = plt.subplots(3, 2, figsize=(FULL_WIDTH, 5.0))
    for ax, (st, ed) in zip(axes.ravel(), panels):
        d = dist[st][ed]
        items = sorted(d["counts"].items(), key=lambda kv: kv[1])
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        ypos = np.arange(len(names))
        ax.hlines(ypos, 1, vals, color=GRID_GUIDE, linewidth=0.7, zorder=1)
        ax.plot(vals, ypos, "o", color=EDITION_COLOUR[ed], markersize=4.5, zorder=2)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names)
        ax.set_xscale("log")
        ax.set_xlim(1, max(vals) * 12)
        ax.set_xticks([1, 1e2, 1e4])  # same ticks in every panel, so the
        ax.minorticks_off()           # six panels read as one figure
        ax.set_ylim(-0.6, len(names) - 0.4)
        for y, v in zip(ypos, vals):
            ax.text(v * 1.5, y, f"{_thousands(v)} ({100 * v / d['n']:.2f}%)",
                    va="center", ha="left", fontsize=6)
        ax.set_title(f"{SUBTASK_TITLE[st]} — {ed} (n = {_thousands(d['n'])})", fontsize=7.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    for ax in axes[-1]:
        ax.set_xlabel("items (log scale)")
    fig.tight_layout(h_pad=1.0, w_pad=1.6)
    return save_figure(fig, "eda", "data_label_distribution")


def figure_provenance(prof: dict[str, Any]) -> Path:
    """
    Share of each class made up of marked items (the added posts that the id and
    anonymisation markers identify; the JSON keeps its older key name "injected").

    This is the chapter's central data finding, so the form is the plainest one
    available: one bar per class, the class size beside its name, a common
    0-100 axis across all four panels. C2A is absent because it was not enriched.
    The class size reads "n = ..." and every bar value carries "%", so the two
    numbers cannot be taken for the same unit.
    """
    per = prof["provenance"]["per_class"]
    groups = [("dbo", "2025"), ("dbo", "2026"), ("vio", "2025"), ("vio", "2026")]

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 3.4), sharex=True)
    for ax, (st, ed) in zip(axes.ravel(), groups):
        sel = sorted(per[st][ed].items(), key=lambda kv: kv[1]["n"])
        names = [f"{c} (n = {_thousands(v['n'])})" for c, v in sel]
        vals = [v["injected_pct_of_class"] for _, v in sel]
        ypos = np.arange(len(sel))
        ax.barh(ypos, vals, color=INJECTED_COLOUR, height=0.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(f"{st.upper()} {ed}", fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        for y, v in zip(ypos, vals):
            ax.text(min(v + 3, 99), y, f"{v:.0f}%", va="center", ha="left", fontsize=6)
    fig.supxlabel("share of the class made up of marked items (%)", fontsize=8)
    fig.tight_layout(w_pad=1.6, h_pad=1.0)
    return save_figure(fig, "eda", "data_provenance_by_class")


def figure_length(prof: dict[str, Any]) -> Path:
    """
    Text length per subtask, both editions overlaid as step outlines.

    A log axis, and for the same reason as the previous figure: the median
    tweet is 88 characters and the longest is 10,000, so on a linear axis the
    distribution is one spike against a flat tail. The predecessor drew these
    as box plots, where the box collapsed to a line and nothing was readable.
    Counts are normalised to shares so the editions can be compared despite
    2026 being roughly twice the size.
    """
    lp = prof["length_profile"]
    bins = np.array(lp["histogram_bins"])
    centres = np.sqrt(bins[:-1] * bins[1:])

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.1), sharey=True)
    for ax, st in zip(axes, ("c2a", "dbo", "vio")):
        for ed in ("2025", "2026"):
            counts = np.array(lp["histogram_counts"][st][ed], dtype=float)
            ax.step(centres, counts / counts.sum(), where="mid",
                    color=EDITION_COLOUR[ed], linewidth=1.0, label=ed)
            med = lp["overall"][st][ed]["char"]["median"]
            ax.axvline(med, color=EDITION_COLOUR[ed], linestyle=":", linewidth=0.7, alpha=0.8)
        ax.set_xscale("log")
        ax.set_title(SUBTASK_TITLE[st], fontsize=8)
        ax.set_xlabel("characters (log scale)")
    axes[0].set_ylabel("share of items")
    # One legend above the panels: inside the C2A panel it covered the curves.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=EDITION_COLOUR[ed], linewidth=1.0, label=ed)
               for ed in ("2025", "2026")]
    handles.append(Line2D([], [], color="#555555", linestyle=":", linewidth=0.7, label="median"))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.08))
    fig.tight_layout(w_pad=1.2)
    return save_figure(fig, "eda", "data_length_distribution")


def figure_reannotation(prof: dict[str, Any]) -> Path:
    """
    Where the 2026 edition changed a 2025 label, for the texts present in both.

    Only ST2 appears: ST1 changed no label at all (0 of 6,756), and ST3's label
    space was redesigned, so a change there cannot be separated from the
    redesign. Cells carry raw counts, per the chart-type policy for matrices.
    """
    ra = prof["cross_edition"]["reannotation"]["dbo"]
    trans = ra["transitions"]
    labels = sorted({k for k in trans} | {c for v in trans.values() for c in v})
    m = np.array([[trans.get(r, {}).get(c, 0) for c in labels] for r in labels], dtype=float)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.62, 2.5))
    masked = np.ma.masked_where(m == 0, m)
    im = ax.imshow(masked, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("label in 2026")
    ax.set_ylabel("label in 2025")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j] > 0:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > m.max() * 0.6 else "black")
    ax.set_title(f"DBO: {ra['label_changed_n']} of {_thousands(ra['shared_texts'])} shared texts "
                 f"relabelled ({ra['label_changed_pct']:.1f}%)", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="texts", shrink=0.85)
    fig.tight_layout()
    return save_figure(fig, "eda", "data_reannotation")


def figure_overlap(prof: dict[str, Any]) -> Path:
    """
    How far the three subtasks annotate the same texts.

    Relevant because Dimension 2 evaluates one fixed configuration across the
    three subtasks. If they largely share their texts, that dimension varies
    the annotation question while holding the text distribution roughly fixed,
    which is a stronger design than three unrelated datasets would give. The
    two panels share an axis so the growth between editions is visible rather
    than normalised away.
    """
    cs = prof["cross_subtask"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.2), sharex=True, sharey=True)
    xmax = max(cs[ed]["union_distinct_texts"] for ed in ("2025", "2026"))
    for ax, ed in zip(axes, ("2025", "2026")):
        d = cs[ed]
        pairs = list(d["pairwise"].items())
        names = [k.replace("|", " ∩ ").upper() for k, _ in pairs] + ["all three"]
        vals = [v["overlap_n"] for _, v in pairs] + [d["in_all_three"]]
        ypos = np.arange(len(names))
        ax.barh(ypos, vals, height=0.6,
                color=[CORPUS_COLOUR] * (len(names) - 1) + [INJECTED_COLOUR])
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        for y, v in zip(ypos, vals):
            ax.text(v + xmax * 0.015, y, _thousands(v), va="center", fontsize=6)
        ax.set_xlim(0, xmax * 1.18)
        ax.set_title(f"{ed}: {_thousands(d['union_distinct_texts'])} distinct texts",
                     fontsize=8)
        ax.set_xlabel("texts shared between subtasks")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=1.0)
    return save_figure(fig, "eda", "data_subtask_overlap")


FIGURES = {
    "labels": figure_labels,
    "provenance": figure_provenance,
    "length": figure_length,
    "reannotation": figure_reannotation,
    "overlap": figure_overlap,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw the Section 3.1 figures.")
    ap.add_argument("--only", nargs="+", choices=list(FIGURES))
    args = ap.parse_args()

    apply_thesis_style()
    prof = _load()
    for name in (args.only or list(FIGURES)):
        path = FIGURES[name](prof)
        print(f"  {name:14s} -> {path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
