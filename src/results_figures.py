"""
results_figures.py -- the two data figures of the Results chapter (Chapter 4).

Like `data_figures.py`, this module only reads stored artefacts and draws. It
computes no new quantity: every value it plots is a figure the chapter already
prints (the combination table of Section 4.2 and Section 4.3.3), taken from the same records the tables are
derived from. As a guard against a figure drifting from its table, every plotted
difference and score is formatted the way the chapter prints it and looked up in
the chapter source when that source is present; a figure that shows a number the
text does not print is refused rather than written.

Figures
-------
combination_vs_best   strategy minus best single component, both editions,
                      paired 95% interval, filled marker = survives Holm
cross_edition         macro-F1 of every component on 2025 and on 2026

Usage
-----
python -m src.results_figures            # both figures
python -m src.results_figures --only cross_edition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotstyle import (COLOURS, FULL_WIDTH, REFERENCE_COLOUR,
                           apply_thesis_style, save_figure)

_ROOT = Path(__file__).parent.parent
RESULTS = _ROOT / "results"
CHAPTER = _ROOT.parent / "chapters" / "04_results.tex"   # absent in the submission repo

SUBTASKS = ("c2a", "dbo", "vio")
SUBTASK_TITLE = {"c2a": "C2A (calls to action)", "dbo": "DBO (attacks on democracy)",
                 "vio": "VIO (violence)"}
EDITION_COLOUR = {"2025": COLOURS[0], "2026": COLOURS[1]}
STRATEGIES = (("soft_voting", "Soft voting"), ("cascade", "Cascade"),
              ("llm_as_judge", "LLM-as-judge"), ("soft_voting_all_six", "Soft voting, all six"))
COMPONENTS = (("tml_svm", "TF-IDF + SVM"), ("tml_xgboost", "TF-IDF + XGBoost"),
              ("tml_lightgbm", "TF-IDF + LightGBM"), ("encoder_1b", "ModernGBERT 1B"),
              ("llm_llammlein", "LLäMmlein 7B (fine-tuned)"),
              ("llm_qwen_fewshot", "Qwen 2.5-14B (few-shot)"))


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"results/{name} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_printed(values: list[str]) -> None:
    """Refuse a figure whose numbers the chapter does not print (thesis tree only)."""
    if not CHAPTER.exists():
        return
    text = CHAPTER.read_text(encoding="utf-8")
    missing = [v for v in values if v not in text]
    if missing:
        raise SystemExit(f"figure values not printed in {CHAPTER.name}: {missing}")


def _strategy_rows(record: dict, edition: str) -> list[dict]:
    rows = []
    for st in SUBTASKS:
        for key, label in STRATEGIES:
            s = record["cells"][st]["strategies"].get(key)
            if s is None:
                continue
            sig = s["significance"]
            # significance compares a = best single against b = strategy; plot b - a.
            diff = -100 * sig["bootstrap"]["difference"]
            lo, hi = sorted(-100 * x for x in sig["bootstrap"]["ci_difference"])
            if f"{diff:.2f}" != f"{s['delta_vs_best_pp']:.2f}" and abs(diff - s["delta_vs_best_pp"]) > 0.006:
                raise SystemExit(f"{edition} {st} {key}: bootstrap difference {diff:.2f} "
                                 f"disagrees with delta_vs_best_pp {s['delta_vs_best_pp']}")
            rows.append({"subtask": st, "strategy": key, "label": label, "edition": edition,
                         "diff": s["delta_vs_best_pp"], "lo": lo, "hi": hi,
                         "survives": sig["multiplicity"]["survives_holm_bootstrap"]})
    return rows


def figure_combination_vs_best() -> Path:
    rows = (_strategy_rows(_load("e3_combination.json"), "2025")
            + _strategy_rows(_load("e3_combination_2026.json"), "2026"))
    _require_printed([f"{r['diff']:+.2f}" for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.6), sharey=True)
    ypos = {key: i for i, (key, _) in enumerate(reversed(STRATEGIES))}
    offset = {"2025": 0.14, "2026": -0.14}
    for ax, st in zip(axes, SUBTASKS):
        ax.axvline(0, color=REFERENCE_COLOUR, linewidth=0.8, zorder=1)
        for r in (r for r in rows if r["subtask"] == st):
            y = ypos[r["strategy"]] + offset[r["edition"]]
            c = EDITION_COLOUR[r["edition"]]
            ax.plot([r["lo"], r["hi"]], [y, y], color=c, linewidth=1.1, zorder=2)
            ax.plot(r["diff"], y, "o", markersize=4.2, zorder=3, markeredgecolor=c,
                    markerfacecolor=c if r["survives"] else "white", markeredgewidth=1.0)
        if not any(r["subtask"] == st and r["edition"] == "2026" and r["strategy"] == "llm_as_judge"
                   for r in rows):
            ax.text(-0.6, ypos["llm_as_judge"] + offset["2026"], "not run on 2026",
                    fontsize=5.5, color=REFERENCE_COLOUR, va="center", ha="right")
        ax.set_title(SUBTASK_TITLE[st].replace(" (", "\n("), fontsize=7.5)
        ax.set_xlim(min(r["lo"] for r in rows) - 0.8, max(r["hi"] for r in rows) + 0.8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(ypos.values()), [dict(STRATEGIES)[k] for k in ypos])
    fig.supxlabel("difference to the best single component (percentage points of macro-F$_1$)",
                  fontsize=7.5, y=0.05)
    handles = [Line2D([], [], color=EDITION_COLOUR["2025"], marker="o", label="2025"),
               Line2D([], [], color=EDITION_COLOUR["2026"], marker="o", label="2026"),
               Line2D([], [], color="#444444", marker="o", linestyle="none", label="survives Holm"),
               Line2D([], [], color="#444444", marker="o", markerfacecolor="white",
                      linestyle="none", label="does not survive")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.55, 1.07),
               fontsize=6.5, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save_figure(fig, "e3", "combination_vs_best")


def figure_cross_edition() -> Path:
    r25 = _load("e3_combination.json")["cells"]
    r26 = _load("e3_combination_2026.json")["cells"]
    printed = []
    for st in SUBTASKS:
        a, b = r25[st]["per_component_macro_f1"], r26[st]["per_component_macro_f1"]
        printed += [f"{100 * a['llm_llammlein']:.2f}", f"{100 * b['llm_llammlein']:.2f}"]
        for comp in ("llm_llammlein", "encoder_1b"):
            printed.append(f"{100 * (a[comp] - b[comp]):.2f}")
    _require_printed(printed)

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.5), sharey=True)
    ypos = {key: i for i, (key, _) in enumerate(reversed(COMPONENTS))}
    for ax, st in zip(axes, SUBTASKS):
        a, b = r25[st]["per_component_macro_f1"], r26[st]["per_component_macro_f1"]
        for key, _ in COMPONENTS:
            y, v25, v26 = ypos[key], 100 * a[key], 100 * b[key]
            ax.plot([v25, v26], [y, y], color="#999999", linewidth=1.6, zorder=2,
                    solid_capstyle="butt")
            ax.plot(v25, y, "o", color=EDITION_COLOUR["2025"], markersize=4, zorder=3)
            ax.plot(v26, y, "o", color=EDITION_COLOUR["2026"], markersize=4, zorder=3)
        ax.set_title(SUBTASK_TITLE[st].replace(" (", "\n("), fontsize=7.5)
        ax.set_xlim(30, 90)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(ypos.values()), [dict(COMPONENTS)[k] for k in ypos])
    fig.supxlabel("macro-F$_1$ (%)", fontsize=7.5, y=0.05)
    handles = [Line2D([], [], color=EDITION_COLOUR[e], marker="o", linestyle="none",
                      label=f"{e}") for e in ("2025", "2026")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.07),
               fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save_figure(fig, "e3", "cross_edition")


FIGURES = {"combination_vs_best": figure_combination_vs_best,
           "cross_edition": figure_cross_edition}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES))
    args = parser.parse_args()
    apply_thesis_style()
    for name in args.only or FIGURES:
        print("wrote", FIGURES[name]())


if __name__ == "__main__":
    main()
