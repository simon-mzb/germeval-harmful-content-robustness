"""
e3_combination_2026.py -- E3 stage 1 on 2026: the combination layer, fitted on 2025, applied once to E4c.

WHAT IT ANSWERS. Dimension 3 for SQ2: does a combination strategy fixed on 2025
keep its standing against the best single component when the same 2025-trained
components score the 2026 evaluation pool? Pre-registered before G3 was known:
stage 1 = soft voting + confidence cascade over the
E4c members, weights and thresholds fitted on 2025 and applied once to 2026. It
is the primary 2026 combination result in every G3 branch. The judge on 2026
(stage 2) is not here -- it runs only if G3(2025) = go, on self-hosted weights.

WHY A SEPARATE MODULE. `e3_combination.py` carries the judge campaign, whose
protocol digest (`f3916558e400`) is re-checked on every campaign day. This file
imports it read-only and never touches `PROTOCOL`; its own rule is `STAGE1` below,
with its own digest, and the E3 digest it inherits is recorded beside it.

THE RULE, in the order a reader would question it:
1. **Members**: the E3 primary set (best per family by E2 mean rank), asserted
   to equal the pre-registered `tml_svm`, `encoder_1b`, `llm_llammlein`. Their 2026 inputs
   are the `__e4c__` store entries: one final fit on 2025, scored on the frozen
   2026 pool (`e4c_runner`).
2. **Soft voting**: equal-weight mean, argmax -- nothing to fit.
3. **Cascade**: ONE threshold pair, fitted with E3's own grid, objective and
   tie-break on ALL 2025 out-of-fold items, applied unchanged to 2026. E3
   cross-fits on 2025 so that no item is scored under a threshold its own label
   helped choose; here no 2026 label enters the fit at all, so a single fit on
   all of 2025 is the analog of E4c's final fit. Counterfactual: fitting on 2026
   would tune on the evaluation items Dimension 3 exists to keep clean;
   averaging the five E3 fold thresholds is a rule nobody pre-registered.
4. **Reference and test**: the best single of all six E4c components on 2026,
   chosen by 2026 macro-F1 exactly as E3 chooses on 2025 (the strictest
   reference). `significance.compare` per strategy, Holm over every
   strategy-vs-best comparison in this artefact. Soft voting over all six is
   reported as the secondary set, as in E3.

WHAT IT DOES NOT CLAIM. One seed per component. The 2025 thresholds were
fitted on fold-model OOF probabilities while the 2026 probabilities come from
one final fit, so the confidence scale may differ between the two -- how well a
threshold transfers is part of what Dimension 3 measures, not a defect to tune
away. VIO 2026 is scored in the 2025 binary label space through
`e4_protocol.e4c.vio_mapping`. TML ran on Darwin/arm64, the neural members on
Linux/x86_64 (`mixes_machines`).

Usage
-----
python -m src.e3_combination_2026            # this text; runs nothing
python -m src.e3_combination_2026 --plan     # members, items, fitted thresholds; no 2026 score, no write
python -m src.e3_combination_2026 --go       # scores 2026 once; writes results/e3_combination_2026.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src import e3_combination as e3

RESULTS = e3.RESULTS
OUT = RESULTS / "e3_combination_2026.json"
PRED_OUT = RESULTS / "e3_combination_2026_predictions.npz"
EDITION = "2026"
VARIANT = "e4c"
MEMBERS = {"tml": "tml_svm", "encoder": "encoder_1b", "llm": "llm_llammlein"}

STAGE1: dict[str, Any] = {
    "package": "B7/E3 stage 1 on 2026 (Dimension 3)",
    "anchor": "THESIS_LOG NS 123 (h22), pre-registered 2026-09-15 before G3",
    "input_fit": "E2 component store, edition 2025, seed 42, calibrated pooled OOF probabilities",
    "input_apply": "E4c component store entries (__e4c__), edition 2026, seed 42: one final fit on 2025, "
                   "scored once on the frozen 2026 evaluation pool",
    "members": "the E3 primary set, asserted equal to tml_svm / encoder_1b / llm_llammlein",
    "soft_voting": "as E3: equal-weight mean of calibrated distributions, argmax; no fitted weights",
    "cascade": "as E3 (order, confidence, grid, objective, tie-break), but ONE threshold pair fitted on all "
               "2025 OOF items and applied unchanged to 2026; no 2026 label enters any fit",
    "reference": "best single of all six E4c components by 2026 macro-F1",
    "secondary_set": "all six E4c components, soft voting only",
    "test": "significance.compare per strategy vs the reference on identical 2026 items; Holm on the "
            "bootstrap p-value over every strategy-vs-best comparison in the artefact",
    "helps_if": "primary strategy macro-F1 above the reference AND survives Holm (bootstrap)",
    "applied": "once; the artefact is refused if it exists",
}


def stage1_digest() -> str:
    return hashlib.sha256(json.dumps(STAGE1, sort_keys=True).encode()).hexdigest()[:12]


def e4c_key(subtask: str, component: str) -> str:
    return "{}_{}/{}__{}__seed{}".format(subtask, EDITION, component, VARIANT, e3.SEED)


def load_2026(subtask: str, components: Sequence[str], store=None) -> dict[str, dict]:
    """The E4c entries, aligned by id and refused if they are not the same items."""
    from src.component_store import ComponentStore
    store = store or ComponentStore()
    recs = store.load_many([e4c_key(subtask, c) for c in components])
    out: dict[str, dict] = {}
    base = None
    for c in components:
        r = recs[e4c_key(subtask, c)]
        if r["meta"].get("uncalibrated"):
            raise ValueError("{} / {} 2026 is flagged uncalibrated".format(subtask, c))
        if r["meta"].get("trained_on") != "2025":
            raise ValueError("{} / {} 2026 was not trained on 2025".format(subtask, c))
        order = np.argsort(np.asarray(r["ids"]), kind="stable")
        r = dict(r, proba=np.asarray(r["proba"])[order], ids=np.asarray(r["ids"])[order],
                 fold=np.asarray(r["fold"])[order],
                 y_true=[str(v) for v in np.asarray(r["y_true"])[order]])
        if base is None:
            base = r
        else:
            if not np.array_equal(base["ids"], r["ids"]):
                raise ValueError("{} 2026: ids differ between {} and {}".format(subtask, components[0], c))
            if base["y_true"] != r["y_true"] or list(base["classes"]) != list(r["classes"]):
                raise ValueError("{} 2026: gold labels or class order differ for {}".format(subtask, c))
            if base["meta"].get("data_rules_id") != r["meta"].get("data_rules_id"):
                raise ValueError("{} 2026: data-rule generations differ for {}".format(subtask, c))
        out[c] = r
    return out


def fit_on_2025(recs25: dict[str, dict], members: Sequence[str]) -> dict[str, Any]:
    """The cascade thresholds from all 2025 OOF items; nothing else is fitted."""
    any_rec = recs25[members[0]]
    classes = [str(c) for c in any_rec["classes"]]
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[v] for v in any_rec["y_true"]])
    stages = [(recs25[c]["proba"].argmax(1), recs25[c]["proba"].max(1)) for c in members]
    th = e3.fit_cascade_thresholds(stages, y, len(classes))
    return {"classes": classes, "t1": th["t1"], "t2": th["t2"],
            "fit_macro_f1_2025_in_sample": th["fit_macro_f1"], "n_fit_items": int(len(y))}


def apply_to_2026(fit: dict[str, Any], recs26: dict[str, dict], members: Sequence[str]) -> dict[str, Any]:
    """Predictions on 2026. Takes no 2026 label: the test asserts it cannot depend on one."""
    classes = [str(c) for c in recs26[members[0]]["classes"]]
    if classes != fit["classes"]:
        raise ValueError("class order differs between 2025 ({}) and 2026 ({})".format(fit["classes"], classes))
    P = {c: np.asarray(recs26[c]["proba"]) for c in recs26}
    sv = e3.soft_vote([P[c] for c in members])
    sv_all = e3.soft_vote([P[c] for c in sorted(P)])
    stages = [(P[c].argmax(1), P[c].max(1)) for c in members]
    cpred, stage = e3.cascade_predict(stages, (fit["t1"], fit["t2"]))
    return {"soft_voting": {"proba": sv, "pred": sv.argmax(1)},
            "soft_voting_all_six": {"proba": sv_all, "pred": sv_all.argmax(1), "members": sorted(P)},
            "cascade": {"pred": cpred, "stage": stage}}


def _primary_set_from_e2(store) -> dict[str, str]:
    scores = {}
    for st in e3.SUBTASKS:
        scores[st] = {}
        for c in e3.FAMILY_OF:
            meta = json.loads(store._json(e3.e2_key(st, c)).read_text(encoding="utf-8"))
            scores[st][c] = float(meta["pooled_macro_f1"])
    primary = e3.select_primary_set(scores)
    if primary != MEMBERS:
        raise SystemExit("REFUSING: the E3 primary set is {}, the pre-registration names {}".format(primary, MEMBERS))
    return primary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--go", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if not (args.plan or args.go):
        print(__doc__)
        return 0
    if args.go and OUT.exists() and not args.force:
        print("SKIP: {} exists; stage 1 is applied once. Replace it with --force only "
              "if you mean to.".format(OUT))
        return 0
    return run(args)


def run(args) -> int:
    from src.component_store import ComponentStore
    store = ComponentStore()
    digest_before = e3.protocol_digest()
    primary = _primary_set_from_e2(store)
    members = [primary[f] for f in e3.CASCADE_ORDER]
    print("E3 stage 1 on 2026: stage-1 digest {}, inherits E3 protocol {}; members {}".format(
        stage1_digest(), digest_before, members))
    per = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # machine mix: recorded in the artefact
        for st in e3.SUBTASKS:
            recs25 = e3.load_subtask(st, members, store)
            recs26 = load_2026(st, sorted(e3.FAMILY_OF), store)
            fit = fit_on_2025(recs25, members)
            per[st] = (recs25, recs26, fit)
            print("  {}: fit on {} items (2025) -> thresholds t1={} t2={}; applied to {} items (2026)".format(
                st, fit["n_fit_items"], fit["t1"], fit["t2"], len(recs26[members[0]]["ids"])))
    if args.plan:
        return 0
    rc = write_artefact(per, members, primary)
    if e3.protocol_digest() != digest_before:
        raise SystemExit("e3_combination.PROTOCOL changed during this run")
    return rc


def write_artefact(per, members, primary) -> int:
    from src import significance as sig
    from src.calibration import calibration_report
    cells, comparisons, pred_arrays = {}, {}, {}
    for st in e3.SUBTASKS:
        recs25, recs26, fit = per[st]
        ap = apply_to_2026(fit, recs26, members)
        classes = fit["classes"]
        k = len(classes)
        idx = {c: i for i, c in enumerate(classes)}
        base_any = recs26[members[0]]
        y = np.array([idx[v] for v in base_any["y_true"]])
        single = {c: float(e3.macro_f1_idx(y, recs26[c]["proba"].argmax(1), k)) for c in recs26}
        best = max(single, key=lambda c: single[c])
        strategies = {"soft_voting": ap["soft_voting"]["pred"], "cascade": ap["cascade"]["pred"],
                      "soft_voting_all_six": ap["soft_voting_all_six"]["pred"]}
        off25 = e3.offline_strategies(recs25, primary)
        y25 = off25["y"]
        cell = {
            "n_items": int(len(y)), "classes": classes, "members": members,
            "mixes_machines": len({recs26[c]["meta"].get("machine") for c in members}) > 1,
            "member_machines": {c: recs26[c]["meta"].get("machine") for c in members},
            "member_config_ids_2026": {c: recs26[c]["meta"].get("config_id") for c in recs26},
            "per_component_macro_f1": single,
            "best_single": best, "best_single_macro_f1": single[best],
            "cascade_fit_2025": fit,
            "reference_2025_offline": {
                "note": "E3's cross-fitted offline figures on 2025 over the same members, for the edition "
                        "contrast; the judge is not part of stage 1",
                "best_single_macro_f1": max(v["macro_f1"] for v in off25["single"].values()),
                "soft_voting_macro_f1": e3.macro_f1_idx(y25, off25["soft_voting"]["pred"], k),
                "cascade_macro_f1": e3.macro_f1_idx(y25, off25["cascade"]["pred"], k),
            },
            "strategies": {},
        }
        for name, pred in strategies.items():
            f1 = e3.macro_f1_idx(y, pred, k)
            cell["strategies"][name] = {
                "macro_f1": f1,
                "per_class_f1": dict(zip(classes, e3.per_class_f1_idx(y, pred, k))),
                "delta_vs_best_pp": round(100 * (f1 - single[best]), 2),
            }
            pred_arrays["{}__{}".format(st, name)] = pred.astype(np.int8)
        cell["strategies"]["soft_voting"]["calibration"] = calibration_report(
            ap["soft_voting"]["proba"], [classes[i] for i in y], classes)
        stage = ap["cascade"]["stage"]
        cell["strategies"]["cascade"].update({
            "thresholds": {"t1": fit["t1"], "t2": fit["t2"], "fitted_on": "all 2025 OOF items"},
            "share_decided_by": {"tml": float((stage == 0).mean()), "encoder": float((stage == 1).mean()),
                                 "llm": float((stage == 2).mean())},
            "escalation_rate": {"to_encoder": float((stage >= 1).mean()), "to_llm": float((stage == 2).mean())},
        })
        base = recs26[best]
        comparisons[st] = []
        for name, pred in strategies.items():
            r = sig.compare(base, e3.as_record(base, pred, k), key_a=best, key_b=name)
            r["strategy"], r["primary"] = name, name != "soft_voting_all_six"
            comparisons[st].append(r)
        cells[st] = cell
        pred_arrays["{}__ids".format(st)] = np.asarray(base["ids"])
        pred_arrays["{}__y".format(st)] = y.astype(np.int8)

    mult = sig.apply_multiplicity(comparisons)
    mult["family"] = "every strategy-vs-best-single comparison in this artefact, every subtask"
    verdict = {}
    for st in e3.SUBTASKS:
        helps = []
        for r in comparisons[st]:
            name = r["strategy"]
            r["multiplicity"]["family"] = mult["family"]
            cells[st]["strategies"][name]["significance"] = r
            if r["primary"] and cells[st]["strategies"][name]["delta_vs_best_pp"] > 0 \
                    and r["multiplicity"]["survives_holm_bootstrap"]:
                helps.append(name)
        verdict[st] = {"combination_helps": bool(helps), "strategies_that_help": helps}
    artefact = {
        "experiment": "B7/E3 stage 1 on 2026 (Dimension 3)",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": e3._git_commit(),
        "stage1": STAGE1, "stage1_digest": stage1_digest(),
        "inherits_e3_protocol_digest": e3.protocol_digest(),
        "primary_set": primary,
        "cells": cells,
        "multiplicity": mult,
        "verdict": verdict,
        "claims_not_made": [
            "one seed per component: a corrected p is about these trained models on these items, not the method",
            "2025 thresholds come from fold-model OOF probabilities, 2026 probabilities from one final fit; "
            "threshold transfer is part of what is measured",
            "VIO 2026 is scored in the 2025 binary label space through e4_protocol.e4c.vio_mapping",
            "the TML member ran on Darwin/arm64, the neural members on Linux/x86_64",
            "no judge: stage 2 runs only if G3(2025) = go (NS 123 h22)",
        ],
    }
    OUT.write_text(json.dumps(artefact, indent=1, default=float) + "\n", encoding="utf-8")
    np.savez_compressed(PRED_OUT, **pred_arrays)
    print("wrote {}".format(OUT))
    for st in e3.SUBTASKS:
        c = cells[st]
        print("  {}: best single {} {:.4f} | soft vote {:+.2f} pp | cascade {:+.2f} pp | helps: {}".format(
            st, c["best_single"], c["best_single_macro_f1"], c["strategies"]["soft_voting"]["delta_vs_best_pp"],
            c["strategies"]["cascade"]["delta_vs_best_pp"], verdict[st]["strategies_that_help"] or "no"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
