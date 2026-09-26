"""
plotstyle.py -- one visual convention for every figure that enters the thesis.

Figures were being built ad hoc, each with its own sizes and colours. That is
tolerable for exploratory work and wrong for a document, because a reader
reads inconsistent styling as carelessness before reading the content. This
module fixes the convention in one place so that later figures inherit it
instead of a promise to unify them at the end.

Sizes are chosen for the actual page: the thesis body is typeset with 3.5 cm
margins, giving a text width of roughly 14 cm (5.5 in). A figure produced at
FULL_WIDTH therefore lands in LaTeX at scale 1 and its font sizes are the
sizes the reader sees, which is why they are set slightly below the body size
rather than shrunk on inclusion.

Usage
-----
from src.plotstyle import apply_thesis_style, save_figure, COLOURS
apply_thesis_style()
...
save_figure(fig, "e2", "reliability_tml")
"""

from __future__ import annotations

from pathlib import Path

FULL_WIDTH = 5.5      # inches; matches the ~14 cm text block
HALF_WIDTH = 2.7

# Colour-blind safe (Okabe-Ito), used in a fixed order so that the same
# component keeps the same colour across every figure in the document.
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

GRID_COLOUR = "#CCCCCC"
REFERENCE_COLOUR = "#777777"


def apply_thesis_style() -> None:
    """Set the rcParams every thesis figure shares. Idempotent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID_COLOUR,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.7,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.0,
        "axes.prop_cycle": cycler(color=COLOURS),
    })


def save_figure(fig, subdir: str, name: str) -> Path:
    """
    Write a figure as PDF (for LaTeX) and PNG (for reading in chat or a browser).

    Both come from the same call, so the version discussed and the version
    typeset can never diverge. Returns the PDF path.
    """
    root = Path(__file__).parent.parent / "figures" / subdir
    root.mkdir(parents=True, exist_ok=True)
    pdf = root / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(root / f"{name}.png")
    return pdf


# ---------------------------------------------------------------------------
# Chart-type policy
# ---------------------------------------------------------------------------
#
# Which chart for which job, fixed once so the document reads as one document.
# The rule of thumb behind it: the mark must match the data type. Bars for
# unordered categories, lines only over a genuinely ordered axis, heatmaps for
# matrices. A line drawn across categories invents a trend that does not exist,
# and that is the single most common way a results chapter misleads.
#
#   comparison across categories   grouped bars, horizontal when the category
#   (component x subtask,          labels are words; error bars are the
#    strategy x subtask)           bootstrap CI, never the standard deviation
#
#   per-class F1                   grouped bars, classes ordered by support
#                                  descending, support printed on the axis so
#                                  the reporting tier is visible next to the value
#
#   anything over an ordered       line with markers (confidence, thresholds,
#   numeric axis                   training epochs, escalation rate vs cutoff)
#
#   calibration                    reliability curve plus a bin-count panel
#                                  underneath; see e2_summary.reliability_figure
#
#   distributions                  histogram; overlaid step outlines when two
#                                  editions are compared, never filled bars
#                                  stacked on each other
#
#   label distributions            horizontal bars sorted by frequency
#
#   confusion matrices             heatmap, row-normalised, raw counts printed
#                                  in the cells
#
# Never: pie charts, 3-D anything, a second y-axis, or a line joining
# unordered categories. Never drop or thin measured points to make a curve
# behave -- add the panel that explains them instead.

# A component keeps one colour everywhere in the thesis. Fixed here rather than
# left to plot order, because later runs add components and a colour that shifts
# between two figures is read as a different thing.
COMPONENT_COLOUR = {
    "tml_svm": COLOURS[0],
    "tml_xgboost": COLOURS[1],
    "tml_lightgbm": COLOURS[2],
    "encoder_1b": COLOURS[3],
    "encoder_134m": COLOURS[4],
    "llm_llammlein": COLOURS[5],
    "llm_qwen_fewshot": "#000000",
}
