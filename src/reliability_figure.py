"""
reliability_figure.py -- the reliability figure of the appendix "Reliability Diagrams".

Read-only over results/e2/e2_<subtask>_2025_<component>.json, key primary/calibration/reliability_bins, and nothing else:
the figure is generated from the artefacts and no number in it is typed. One panel per subtask, six components each, equal-width
bins exactly as the reported expected calibration error (the figure and the metric are one measurement). Under each curve a
count panel on a logarithmic axis says how many items each bin holds, which is where the high-confidence bins show.

Colour-blind safety: Okabe-Ito colours from plotstyle (fixed per component) AND a distinct marker and line style per component,
so the figure survives greyscale. The classical components are drawn at the PRIMARY seed (the bins are stored for that seed only),
so no ECE is printed in the figure: Table "calibration" reports the three-seed mean for them and the two would differ.

Usage:  python -m src.reliability_figure
Writes figures/e2/reliability_all.{pdf,png} and prints the derived figures the appendix quotes.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.plotstyle import (COMPONENT_COLOUR, FULL_WIDTH, REFERENCE_COLOUR, apply_thesis_style, save_figure)

SUBTASKS = ("c2a", "dbo", "vio")
TITLE = {"c2a": "C2A", "dbo": "DBO", "vio": "VIO"}
COMPONENTS = ("tml_svm", "tml_xgboost", "tml_lightgbm", "encoder_1b", "llm_llammlein", "llm_qwen_fewshot")
LABEL = {"tml_svm": "TF-IDF + SVM", "tml_xgboost": "TF-IDF + XGBoost", "tml_lightgbm": "TF-IDF + LightGBM",
         "encoder_1b": "ModernGBERT 1B", "llm_llammlein": "LLäMmlein 7B", "llm_qwen_fewshot": "Qwen 2.5-14B (few-shot)"}
MARKER = dict(zip(COMPONENTS, ("o", "s", "^", "D", "v", "X")))
STYLE = dict(zip(COMPONENTS, ("-", "--", "-.", ":", "-", "--")))
MIN_CURVE_ITEMS = 10   # a bin with fewer items is counted in the lower panel but not drawn as a point of the curve (single-item bins dominated the DBO panel)
FLOOR = 0.25   # a top-label confidence cannot fall below 1/K; DBO has four classes


def load() -> dict:
    out = {}
    for s in SUBTASKS:
        for c in COMPONENTS:
            d = json.loads((ROOT / "results" / "e2" / f"e2_{s}_2025_{c}.json").read_text())
            out[(s, c)] = d["primary"]["calibration"]["reliability_bins"]
    return out


def top_bin_figures(bins: dict) -> dict:
    """Per cell: the top bin's share of the pool and the gap between its stated confidence and observed accuracy (points)."""
    out = {}
    for (s, c), b in bins.items():
        n = sum(x["n"] for x in b)
        t = b[-1]
        out[(s, c)] = dict(share=100 * t["n"] / n, gap=100 * abs(t["mean_confidence"] - t["accuracy"]), n=t["n"], lower=t["bin_lower"])
    return out


def main() -> None:
    import matplotlib.pyplot as plt
    apply_thesis_style()
    bins = load()
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 3.3), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12, "wspace": 0.3})
    handles = []
    for col, s in enumerate(SUBTASKS):
        ax, axn = axes[0][col], axes[1][col]
        ax.plot([FLOOR, 1], [FLOOR, 1], color=REFERENCE_COLOUR, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
        for c in COMPONENTS:
            b = bins[(s, c)]
            xs, ys, ns = [x["mean_confidence"] for x in b], [x["accuracy"] for x in b], [x["n"] for x in b]
            col_ = COMPONENT_COLOUR[c]
            keep = [i for i, n in enumerate(ns) if n >= MIN_CURVE_ITEMS]
            h, = ax.plot([xs[i] for i in keep], [ys[i] for i in keep], marker=MARKER[c], linestyle=STYLE[c], color=col_, markersize=2.6, linewidth=0.9, zorder=2, label=LABEL[c])
            axn.plot(xs, ns, marker=MARKER[c], linestyle=STYLE[c], color=col_, markersize=2.0, linewidth=0.7)
            if s == SUBTASKS[0]:
                handles.append(h)
        ax.set_title(TITLE[s])
        ax.set_xlim(FLOOR, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        axn.set_yscale("log")
        axn.set_ylim(0.6, 2e4)
        axn.set_yticks([1, 100, 10000])
    axes[0][0].set_ylabel("observed accuracy")
    axes[1][0].set_ylabel("items")
    fig.supxlabel("mean confidence", fontsize=8)
    fig.legend(handles, [LABEL[c] for c in COMPONENTS], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), columnspacing=1.6)
    print(save_figure(fig, "e2", "reliability_all"))
    # provenance: the digests of the 18 inputs, so a figure older than its data can be detected
    import hashlib
    src = {f"e2_{s}_2025_{c}.json": hashlib.sha256((ROOT / "results" / "e2" / f"e2_{s}_2025_{c}.json").read_bytes()).hexdigest()
           for s in SUBTASKS for c in COMPONENTS}
    src["_min_curve_items"] = MIN_CURVE_ITEMS
    (ROOT / "figures" / "e2" / "reliability_all.sources.json").write_text(json.dumps(src, indent=1, sort_keys=True) + "\n")
    for (s, c), v in top_bin_figures(bins).items():
        print(s, c, f"top bin from {v['lower']:.3f}: {v['n']} items, {v['share']:.1f}% of the pool, gap {v['gap']:.2f} pp")


if __name__ == "__main__":
    main()
