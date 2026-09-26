"""test_selection_diagnosis.py -- does the grid-edge alarm actually fire?

`diagnose_fit` answers "was this fine-tune given enough budget". This answers
the other half, and until 2026-08-28 nothing did: **did the SEARCH resolve, or
did the grid decide?** #31a found the SVM's C on the upper edge in 14 of 15
folds by hand; #84b found the encoder's lr on its upper edge by hand. Both are
the same defect, both were caught by someone remembering to look, and the
campaign records everything needed to catch them automatically.

The same discipline as the fit alarm applies here: a healthy arm must stay
QUIET, because an alarm that always fires is not an alarm. Half the cases below
exist for that.

Run: .venv/bin/python -m tests.test_selection_diagnosis
"""
from __future__ import annotations

import sys

from src.e2_runner import diagnose_selection, selection_diagnosis_for

PASS = FAIL = 0

LR = [2.0e-4, 3.0e-4, 5.0e-4]
R = [8, 16]
GRID = [{"lora_r": r, "lr": lr} for lr in LR for r in R]


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  \033[32mPASS\033[0m  " + name)
    else:
        FAIL += 1
        print("  \033[31mFAIL\033[0m  " + name)


def sel(*pairs):
    return [{"lora_r": r, "lr": lr} for r, lr in pairs]


def grids(*margins):
    """One selection_grid per fit, with a given winner-to-runner-up margin."""
    return [[{"selection_macro_f1": 0.60}, {"selection_macro_f1": 0.60 - m}]
            for m in margins]


# --- the failure the check exists for --------------------------------------

d = diagnose_selection(sel((8, 5e-4), (16, 5e-4), (8, 5e-4), (16, 5e-4), (8, 5e-4)), GRID)
check("all fits on the upper lr edge -> grid_edge:lr", "grid_edge:lr" in d["flags"])
check("  ... and it says which end and what to do",
      "upper end" in d["notes"][0] and "re-run the WHOLE arm" in d["notes"][0])
check("  ... and lora_r, which varied, is NOT flagged",
      "grid_edge:lora_r" not in d["flags"])
check("  ... the counts travel with the flag",
      d["axes"]["lr"]["selected_counts"]["0.0005"] == 5
      and d["axes"]["lr"]["share_at_upper_edge"] == 1.0)

d = diagnose_selection(sel((8, 2e-4), (8, 2e-4), (8, 2e-4)), GRID)
check("all fits on the LOWER edge of both axes -> both flagged",
      set(d["flags"]) == {"grid_edge:lr", "grid_edge:lora_r"})

# --- the healthy case, which must stay quiet -------------------------------

d = diagnose_selection(sel((8, 3e-4), (16, 3e-4), (8, 2e-4), (16, 5e-4), (8, 3e-4)),
                       GRID, selection_grids=grids(0.05, 0.06, 0.04, 0.05, 0.05),
                       fold_scores=[0.80, 0.81, 0.79, 0.82, 0.80])
check("an interior optimum raises nothing", d["ok"] and not d["flags"])
check("  ... and still reports the winner spread", d["n_distinct_winners"] == 4)

# One fit on an edge is not a finding -- this is the case that would make the
# check useless if it were stricter.
d = diagnose_selection(sel((8, 5e-4), (16, 3e-4), (8, 3e-4), (16, 2e-4), (8, 3e-4)), GRID)
check("a single fit on an edge is not flagged", d["ok"])

# --- selection on noise ----------------------------------------------------

d = diagnose_selection(sel((8, 3e-4), (16, 2e-4), (8, 5e-4), (16, 3e-4), (8, 2e-4)),
                       GRID, selection_grids=grids(0.001, 0.002, 0.001, 0.0005, 0.001),
                       fold_scores=[0.70, 0.78, 0.66, 0.81, 0.72])
check("unstable winners + margins below the fold spread -> selection_noise",
      "selection_noise" in d["flags"])
check("  ... and the margin statistics are reported",
      d["margin"]["n_fits"] == 5 and 0.0 < d["margin"]["median"] < 0.01
      and d["margin"]["fold_sd"] > 0)

# The condition that keeps this from firing on nine cells out of ten: if every
# fold picked the SAME configuration, nothing arbitrary happened, whatever the
# margin. Measured on the committed B3 artefacts, this is exactly the case that
# separates c2a/tml_svm from its eight siblings.
d = diagnose_selection(sel((8, 3e-4), (8, 3e-4), (8, 3e-4), (8, 3e-4), (8, 3e-4)),
                       GRID, selection_grids=grids(0.001, 0.001, 0.001, 0.001, 0.001),
                       fold_scores=[0.70, 0.78, 0.66, 0.81, 0.72])
check("one winner everywhere is never called noise, however thin the margin",
      "selection_noise" not in d["flags"] and d["n_distinct_winners"] == 1)

d = diagnose_selection(sel((8, 3e-4), (16, 2e-4), (8, 5e-4)),
                       GRID, selection_grids=grids(0.09, 0.10, 0.11),
                       fold_scores=[0.80, 0.805, 0.799])
check("margins above the fold spread stay quiet", d["ok"])

# Without fold_scores the noise rule cannot be evaluated and must not guess.
d = diagnose_selection(sel((8, 3e-4), (16, 2e-4)), GRID,
                       selection_grids=grids(0.0001, 0.0001))
check("no fold scores -> no noise verdict, and no crash",
      "selection_noise" not in d["flags"] and d["margin"] is not None)

# --- degenerate inputs -----------------------------------------------------

d = diagnose_selection(sel((8, 3e-4), (8, 3e-4)), [{"lora_r": 8, "lr": 3e-4}])
check("a one-cell grid is labelled, not silently 'ok'",
      d["flags"] == ["single_cell_grid"] and d["ok"])
check("  ... and no edge is claimed on an axis with one value", d["axes"] == {})

check("an empty artefact returns a shape rather than raising",
      diagnose_selection([], [])["n_fits"] == 0)
check("selection_diagnosis_for(None-ish) is None",
      selection_diagnosis_for({"repeats": []}) is None)

# --- the artefact path, which is what the runner and the notebook use ------

artefact = {
    "meta": {"grid": GRID},
    "repeats": [{
        "seed": 42,
        "per_fold_macro_f1": [0.80, 0.81, 0.79, 0.82, 0.80],
        "selected_params_per_fold": [{"lora_r": 8, "lr": 5e-4}] * 5,
        "folds_detail": [{"selection_grid": [{"selection_macro_f1": 0.6},
                                             {"selection_macro_f1": 0.5}]}] * 5,
    }],
    "seed_replicates": [{"selected_params": {"lora_r": 16, "lr": 5e-4}},
                        {"selected_params": {"lora_r": 8, "lr": 5e-4}}],
}
d = selection_diagnosis_for(artefact)
check("the artefact path finds the edge across folds AND replicates",
      "grid_edge:lr" in d["flags"] and d["n_fits"] == 7)
check("  ... using the primary repeat's fold scores",
      d["margin"]["n_fits"] == 5)

# A repeat list without the primary seed must still be readable -- an artefact
# produced with --seeds 43 is legal.
d = selection_diagnosis_for({"meta": {"grid": GRID},
                             "repeats": [{"seed": 43,
                                          "selected_params_per_fold": [{"lora_r": 8, "lr": 3e-4}],
                                          "folds_detail": []}]})
check("a non-primary-seed artefact still yields a verdict", d is not None and d["n_fits"] == 1)

# ---------------------------------------------------------------------------
# stopping_report -- the EVIDENCE for the epoch rule, not an alarm
# ---------------------------------------------------------------------------
# Does the stop mechanics make sense -- best
# result without missing it, without paying for dead runs? `patience = 2` was
# derived from 14 traces, which is a defensible derivation and a thin sample.
# 3.3 has to answer the obvious objection ("you stopped too early") with a
# measurement rather than with the derivation, and these are the two shapes
# that decide it.

from src.e2_runner import stopping_report


def cell(scores, ceiling=8, patience=2):
    """One grid cell's record, exactly as _fit_one writes it."""
    best = max(range(len(scores)), key=lambda i: scores[i])
    stopped_early = len(scores) < ceiling
    return {
        "params": {"lora_r": 8, "lr": 3e-4},
        "selection_macro_f1": scores[best],
        "best_epoch": best + 1,
        "epoch_trace": [{"epoch": i + 1, "selection_macro_f1": s}
                        for i, s in enumerate(scores)],
        "stopping": {"epochs_run": len(scores), "stopped_early": stopped_early,
                     "patience": patience, "ceiling": ceiling,
                     "ceiling_bound": (not stopped_early) and best + 1 == ceiling},
    }


def artefact(*cells):
    return {"meta": {"grid": GRID},
            "repeats": [{"seed": 42, "per_fold_macro_f1": [0.8] * len(cells),
                         "folds_detail": [{"selection_grid": [c]} for c in cells]}]}


# The healthy regime: a peak, then two clear declines, then the stop.
r = stopping_report(artefact(cell([0.40, 0.55, 0.62, 0.58, 0.54]),
                             cell([0.30, 0.61, 0.59, 0.55])))
check("healthy traces: every fit counted, none at the ceiling",
      r["n_fits"] == 2 and r["n_stopped_early"] == 2 and r["n_ceiling_bound"] == 0)
check("  ... none stopped on a rising trace", r["rising_at_stop"] == 0)
check("  ... and the verdict says patience was never binding, with the counts",
      "never the binding constraint" in r["verdict"] and "2 of 2" in r["verdict"])
check("  ... the tail gap is measured, not asserted", r["mean_tail_gap"] > 0)
check("  ... and the overrun over an oracle stop is reported",
      r["overrun_ratio"] > 1.0)

# The failure the rule could actually have: it stopped while climbing back.
r = stopping_report(artefact(cell([0.40, 0.62, 0.55, 0.58]),
                             cell([0.30, 0.61, 0.50, 0.57]),
                             cell([0.30, 0.62, 0.58, 0.55])))
check("two of three early stops on a rising trace -> patience is too tight",
      r["rising_at_stop"] == 2 and "too tight" in r["verdict"])
check("  ... and the share is reported for 3.3",
      abs(r["rising_at_stop_share"] - 2 / 3) < 1e-9)

# The opposite failure, which v1.7 exists to remove.
r = stopping_report(artefact(cell([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
                             cell([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])))
check("every fit rides the ceiling -> the budget decided, not the data",
      r["n_ceiling_bound"] == 2 and r["n_stopped_early"] == 0
      and "BUDGET, not the data" in r["verdict"])
check("  ... and the verdict quotes the counts rather than saying 'every fit'",
      "2 of 2" in r["verdict"])
check("  ... best_epoch_counts shows where the peaks sat",
      r["best_epoch_counts"] == {"8": 2})

# A near miss: the last epoch came within 1% of the best. Worth counting
# separately, because it is the case where the stop cost almost nothing.
r = stopping_report(artefact(cell([0.40, 0.6200, 0.6180, 0.6190])))
check("a stop within 1% of the best is counted as a near miss",
      r["near_miss_within_1pct"] == 1)

# Graceful on the pre-v1.7 artefacts, which have no `stopping` block at all.
check("an artefact without stopping blocks returns None, not a crash",
      stopping_report({"meta": {"grid": GRID}, "repeats": [
          {"seed": 42, "folds_detail": [
              {"selection_grid": [{"best_epoch": 5, "epoch_trace": []}]}]}]}) is None)
check("an empty artefact returns None", stopping_report({"repeats": []}) is None)

print("\n{} passed, {} failed".format(PASS, FAIL))
sys.exit(1 if FAIL else 0)
