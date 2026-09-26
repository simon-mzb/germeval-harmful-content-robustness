"""
b12_ablation.py -- the ablation study (B12): drop one member from each offline combination
strategy, on both editions.

WHAT IT ANSWERS. How much each component contributes to a combination: the
full three-member strategy against the same strategy with one member removed,
on identical items. A positive contribution means the combination is worse
without that member.

THE RULE (fixed before any ablation number existed):
1. **No "best combination" is chosen.** Ablating only the strategy that scored
   highest would select on the evaluation items, so BOTH offline primary
   strategies -- soft voting and the confidence cascade -- are ablated, on 2025
   and on 2026.
2. **The judge is not ablated.** Every leave-one-out judge would need new GWDG
   calls against a fixed quota, and G3 was still open when the rule was set.
3. **Soft voting without member m** = equal-weight mean of the other two.
   **Cascade without member m** = the other two stages in E3's order with one
   threshold, fitted with E3's grid, objective and tie-break. It is computed by
   `e3.fit_cascade_thresholds` with the last stage duplicated: the duplicate can
   never change a prediction, so the fit reduces to the one threshold, and the
   tie-break (least escalation, then lowest threshold) is unchanged.
4. **Fitting mirrors the full strategy.** 2025: cross-fitted over the E2 folds,
   as in E3. 2026: one threshold on all 2025 OOF items, applied once, as in
   `e3_combination_2026` -- no 2026 label enters any fit.
⚠️ The single family of 36 in rule 5 is SUPERSEDED AS PRIMARY: the ablation
covers two parent experiments, so the primary analysis is one family per parent
(2025, 2026; m = 18 each) in `multiplicity_by_experiment`. This artefact stays as computed.

5. **Test.** Each ablated variant against the full strategy of its subtask and
   edition, `significance.compare`, paired interval decides; one Holm family
   over all 36 comparisons.

Usage
-----
python -m src.b12_ablation --help
python -m src.b12_ablation            # writes results/b12_ablation.json (refuses to overwrite)
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src import e3_combination as e3
from src import e3_combination_2026 as s1
from src.harness import measuring_main_guard

OUT = e3.RESULTS / "b12_ablation.json"
STRATEGIES = ("soft_voting", "cascade")


def ablation_variants(members: Sequence[str]) -> dict[str, list[str]]:
    out = {"full": list(members)}
    for m in members:
        out["without_" + m] = [c for c in members if c != m]
    return out


def cascade_stages(P: dict[str, np.ndarray], members: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """E3's three-stage form; a two-member cascade duplicates its last stage (inert)."""
    stages = [(P[c].argmax(1), P[c].max(1)) for c in members]
    if len(stages) == 2:
        stages.append(stages[-1])
    if len(stages) != 3:
        raise ValueError("a cascade needs two or three members, got {}".format(list(members)))
    return stages


def _labels(rec: dict, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[str(v)] for v in rec["y_true"]])


def run_2025(recs25: dict[str, dict], members: Sequence[str]) -> dict[str, dict]:
    base = recs25[members[0]]
    classes = [str(c) for c in base["classes"]]
    k, y, fold = len(classes), _labels(base, classes), np.asarray(base["fold"])
    P = {c: np.asarray(recs25[c]["proba"]) for c in members}
    out = {}
    for name, ms in ablation_variants(members).items():
        cf = e3.cross_fit_cascade(cascade_stages(P, ms), y, fold, k)
        out[name] = {"members": ms,
                     "soft_voting": e3.soft_vote([P[c] for c in ms]).argmax(1),
                     "cascade": {"pred": cf["pred"], "stage": cf["stage"],
                                 "thresholds": cf["thresholds"]}}
    return out


def run_2026(recs25: dict[str, dict], P26: dict[str, np.ndarray], classes26: Sequence[str],
             members: Sequence[str]) -> dict[str, dict]:
    """2026 predictions. Takes 2026 probabilities only -- no 2026 label can enter."""
    base = recs25[members[0]]
    classes = [str(c) for c in base["classes"]]
    if [str(c) for c in classes26] != classes:
        raise ValueError("class order differs between 2025 ({}) and 2026 ({})".format(classes, list(classes26)))
    k, y25 = len(classes), _labels(base, classes)
    P25 = {c: np.asarray(recs25[c]["proba"]) for c in members}
    out = {}
    for name, ms in ablation_variants(members).items():
        th = e3.fit_cascade_thresholds(cascade_stages(P25, ms), y25, k)
        pred, stage = e3.cascade_predict(cascade_stages(P26, ms), (th["t1"], th["t2"]))
        out[name] = {"members": ms,
                     "soft_voting": e3.soft_vote([P26[c] for c in ms]).argmax(1),
                     "cascade": {"pred": pred, "stage": stage,
                                 "thresholds": {"t1": th["t1"], "t2": th["t2"], "fitted_on": "all 2025 OOF items"}}}
    return out


def main(argv=None) -> int:
    args = measuring_main_guard(__doc__, OUT, argv)
    if args is None:
        return 0
    from src import significance as sig
    from src.component_store import ComponentStore
    store = ComponentStore()
    digest = e3.protocol_digest()
    primary = s1._primary_set_from_e2(store)
    members = [primary[f] for f in e3.CASCADE_ORDER]
    cells, comparisons = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # machine mix: recorded below
        for st in e3.SUBTASKS:
            recs25 = e3.load_subtask(st, members, store)
            recs26 = s1.load_2026(st, members, store)
            classes = [str(c) for c in recs25[members[0]]["classes"]]
            k = len(classes)
            runs = {
                "2025": (recs25[members[0]], run_2025(recs25, members)),
                "2026": (recs26[members[0]], run_2026(recs25, {c: np.asarray(recs26[c]["proba"]) for c in members},
                                                      recs26[members[0]]["classes"], members)),
            }
            for ed, (template, res) in runs.items():
                y = _labels(template, classes)
                cell_key = "{}_{}".format(st, ed)
                cell = {"n_items": int(len(y)), "classes": classes, "members": members,
                        "mixes_machines": len({r["meta"].get("machine") for r in
                                               (recs25 if ed == "2025" else recs26).values()}) > 1,
                        "variants": {}}
                for name, v in res.items():
                    stage = v["cascade"]["stage"]
                    cell["variants"][name] = {
                        "members": v["members"],
                        "soft_voting_macro_f1": e3.macro_f1_idx(y, v["soft_voting"], k),
                        "cascade_macro_f1": e3.macro_f1_idx(y, v["cascade"]["pred"], k),
                        "cascade_thresholds": v["cascade"]["thresholds"],
                        "cascade_escalation_beyond_first_stage": float((stage >= 1).mean()),
                    }
                comparisons[cell_key] = []
                for strat in STRATEGIES:
                    full = res["full"][strat] if strat == "soft_voting" else res["full"][strat]["pred"]
                    for m in members:
                        abl = res["without_" + m]
                        abl = abl[strat] if strat == "soft_voting" else abl[strat]["pred"]
                        r = sig.compare(e3.as_record(template, full, k), e3.as_record(template, abl, k),
                                        key_a="{}/full".format(strat), key_b="{}/without_{}".format(strat, m),
                                        n_bootstrap=sig.SIG_BOOTSTRAP_N)
                        bs = r["bootstrap"]
                        r.update({
                            "strategy": strat, "dropped": m, "edition": ed, "subtask": st,
                            "contribution_pp": round(100 * bs["difference"], 2),
                            "contribution_ci_pp": [round(100 * x, 2) for x in bs["ci_difference"]],
                            "decided_by_paired_ci": ("distinguishable" if bs["difference_ci_excludes_zero"]
                                                     else "indistinguishable"),
                        })
                        comparisons[cell_key].append(r)
                        print("  {} {:<12s} without {:<14s} contribution {:+.2f} pp [{:+.2f}, {:+.2f}] {}".format(
                            cell_key, strat, m, r["contribution_pp"], *r["contribution_ci_pp"],
                            r["decided_by_paired_ci"]))
                cells[cell_key] = cell
    mult = sig.apply_multiplicity(comparisons)
    family = "every ablation comparison in this artefact (2 strategies x 3 drops x 3 subtasks x 2 editions)"
    mult["family"] = family
    for rows in comparisons.values():
        for r in rows:
            r["multiplicity"]["family"] = family
            key = "{}_{}".format(r["subtask"], r["edition"])
            cells[key]["variants"]["without_" + r["dropped"]].setdefault("significance", {})[r["strategy"]] = r
    if e3.protocol_digest() != digest:
        raise SystemExit("e3_combination.PROTOCOL changed during this run")
    payload = {
        "experiment": "B12 ablations over the offline combination strategies (THESIS_LOG NS 124 e)",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": e3._git_commit(),
        "e3_protocol_digest": digest, "stage1_digest": s1.stage1_digest(),
        "members": members,
        "sign": "contribution = full minus ablated; positive means the strategy is worse without the member",
        "not_ablated": "the LLM-as-judge: every variant needs new GWDG calls (NS 123 h21) and G3 is open",
        "multiplicity": mult,
        "cells": cells,
    }
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    print("wrote {} ({} comparisons, one family)".format(args.out, mult["m"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
