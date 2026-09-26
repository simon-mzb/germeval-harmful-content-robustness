"""
e5_irreducible.py -- the items no component recovers: size per class, why no strategy can
repair them, how confidently they are wrong, and whether the injected cohort explains them.

WHY A SIBLING OF `e5_error_analysis` AND NOT AN EDIT. That artefact is committed and
its numbers stay reproducible as they are. Section 4.5 needs six further quantities, all computed from the same stored predictions at seed 42:

1. **Size per class** of the irreducible set, with Wilson intervals: macro-F1 weighs every
   class equally, so the set is sized on that scale, not as a share of the pool.
2. **The construction check.** On an item every component gets wrong, the cascade is wrong
   by construction (it passes on one component's argmax); on a binary subtask every
   component's probability for the true class is below 0.5, so any mean is too; and where
   the primary set is unanimous the judge is never called. Recorded as counts so the text
   can say which part is a logical bound and which part is measured (DBO soft voting; the
   judge on the DBO items where the primary set disagrees).
3. **The consensus control.** Mean confidence of the WRONG components only, by the number k
   of components wrong on the item; the right components' confidence beside it. The mean
   over all six mixes the two and produces a spurious U-shape.
4. **The injected cohort** (short or 19-digit 2021 id, `data_profile._is_injected`): irreducible
   rate inside vs outside the cohort per class, with Wilson intervals. It marks provenance,
   not label quality.
5. **AUROC of confidence for error detection** per component: the probability that a
   correct prediction carries higher confidence than a wrong one. Threshold-free, which
   avoids an arbitrary threshold sweep. It is DEFINED on the stored calibrated
   probabilities, because those are what the cascade routes on -- and no invariance to
   temperature scaling is claimed, in any label space: temperatures are fitted PER FOLD
   (e.g. c2a `llm_llammlein` T = 2.500 / 3.048 / 1.397 / 1.490 / 1.031), so the pooled
   out-of-fold ranking mixes items through different monotone maps and can change even
   on two classes.
6. **The cascade the thesis fitted** on 2025 (E3's cross-fitted offline cascade over the
   primary set): its stage order, the thresholds chosen per fold, and the share of items
   passed beyond the first stage and on to the last. Section 4.5 reports these instead of a
   0.9 cutoff on another component; read from `e3.offline_strategies`, the same
   call the construction check already makes, so `e3_combination.py` is only imported.

Usage
-----
python -m src.e5_irreducible --help
python -m src.e5_irreducible          # writes results/e5_irreducible.json (refuses to overwrite)
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src import e3_combination as e3
from src.harness import measuring_main_guard

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "e5_irreducible.json"
COMPONENTS = sorted(e3.FAMILY_OF)
PRIMARY = {"tml": "tml_svm", "encoder": "encoder_1b", "llm": "llm_llammlein"}
INJECTED_BELOW = 10 ** 12   # short ids: no platform assigns them
INJECTED_FROM = 10 ** 18    # 19-digit Twitter ids, all dated 2021; counting only the short ids
                            # would miss 574 added DBO items


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for k successes in n trials (Brown, Cai & DasGupta 2001)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def auroc_confidence(conf: np.ndarray, correct: np.ndarray) -> float | None:
    """P(conf of a correct prediction > conf of a wrong one), ties counted half."""
    pos, neg = conf[correct], conf[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def analyse_subtask(subtask: str, recs: dict[str, dict]) -> dict[str, Any]:
    base = recs[COMPONENTS[0]]
    classes = [str(c) for c in base["classes"]]
    k_cls = len(classes)
    y = np.array([classes.index(str(v)) for v in base["y_true"]])
    ids = np.asarray(base["ids"])
    P = np.stack([np.asarray(recs[c]["proba"]) for c in COMPONENTS])
    pred, conf = P.argmax(2), P.max(2)
    wrong = pred != y
    k_wrong = wrong.sum(0)
    nobody = k_wrong == len(COMPONENTS)

    size = {}
    for i, c in enumerate(classes):
        m = y == i
        n_set = int((nobody & m).sum())
        size[c] = {"support": int(m.sum()), "n_irreducible": n_set, "share_of_class": n_set / int(m.sum()),
                   "ci_95": wilson(n_set, int(m.sum()))}

    members = [PRIMARY[f] for f in e3.CASCADE_ORDER]
    off = e3.offline_strategies(recs, PRIMARY)
    mi = [COMPONENTS.index(c) for c in members]
    primary_unanimous = (pred[mi] == pred[mi][0]).all(0)
    construction = {
        "n_irreducible": int(nobody.sum()),
        "binary_subtask": k_cls == 2,
        "cascade_recovers": int((off["cascade"]["pred"][nobody] == y[nobody]).sum()),
        "soft_voting_primary_recovers": int((off["soft_voting"]["pred"][nobody] == y[nobody]).sum()),
        "soft_voting_all_six_recovers": int((off["soft_voting_all"]["pred"][nobody] == y[nobody]).sum()),
        "all_six_vote_picks_a_class_no_component_predicted": int(
            (off["soft_voting_all"]["pred"][nobody][None, :] != pred[:, nobody]).all(0).sum()),
        "primary_set_unanimous_on_set": int((primary_unanimous & nobody).sum()),
        "judge_invoked_on_set": int((~primary_unanimous & nobody).sum()),
        "bound_note": ("cascade: wrong by construction on every item all components get wrong; binary subtasks: "
                       "soft voting wrong by construction (every true-class probability < 0.5) and the judge "
                       "never invoked (primary set unanimous); measured: soft voting on multi-class subtasks, "
                       "the judge on `judge_invoked_on_set` items"),
    }

    # The judge ON the set, READ from the 2025 cache -- no calls are made here.
    # G3 came back no, so the judge is reported on 2025 only.
    # `assemble_judge` is e3's own rule, including its fallback to the soft vote
    # when a cached row carries no label: rebuilding that join here would be a
    # second definition of the same strategy.
    rows = e3.JudgeCache(e3.PRIMARY_JUDGE, subtask).rows
    judged = e3.assemble_judge(off, recs, rows)["pred"]
    on_set = ~primary_unanimous & nobody
    construction["judge_model"] = e3.PRIMARY_JUDGE
    construction["judge_recovers"] = int((judged[nobody] == y[nobody]).sum())
    construction["judge_fallback_on_set"] = int(
        sum(1 for j in np.flatnonzero(on_set) if rows.get(int(ids[j]), {}).get("label") is None))
    construction["judge_rows_missing_on_set"] = int(
        sum(1 for j in np.flatnonzero(on_set) if int(ids[j]) not in rows))
    # Per class, because the set is read per class everywhere else in this
    # artefact, and because a recovery count without its base is unreadable.
    recovered = on_set & (judged == y)
    construction["judge_invoked_on_set_by_class"] = {
        c: int((on_set & (y == i)).sum()) for i, c in enumerate(classes)}
    construction["judge_recovers_by_class"] = {
        c: int((recovered & (y == i)).sum()) for i, c in enumerate(classes)}
    # A judge answer may name any label of the set, including one no member
    # proposed -- which is the only way an item of this set can be recovered at
    # all, since every member is wrong on it. Counted rather than assumed.
    construction["judge_recovered_naming_unproposed_class"] = int(sum(
        1 for j in np.flatnonzero(recovered)
        if rows[int(ids[j])]["label"] not in {str(c) for c, _ in rows[int(ids[j])]["candidates"]}))

    # THE OTHER SIDE OF THOSE RECOVERIES. On the irreducible set "turned right into wrong" is zero by
    # definition, because every component is wrong on every item of it, so the
    # question is only answerable on the FULL disagreement set -- the items the
    # judge is actually called on. The reference is the best single component,
    # which is what Section 4.2 compares every strategy against.
    best = max(off["single"], key=lambda c: off["single"][c]["macro_f1"])
    bp = off["single"][best]["pred"]
    dis = off["disagree"]
    judge_rows_dis = [rows.get(int(ids[j])) for j in np.flatnonzero(dis)]
    construction["judge_on_disagreement"] = {
        "reference": best,
        "n_disagreement": int(dis.sum()),
        "judge_named_unproposed_class": int(sum(
            1 for r in judge_rows_dis
            if r and r.get("label") is not None
            and r["label"] not in {str(c) for c, _ in r["candidates"]})),
        "turned_reference_right_to_wrong": int((dis & (bp == y) & (judged != y)).sum()),
        "turned_reference_wrong_to_right": int((dis & (bp != y) & (judged == y)).sum()),
    }

    consensus = {}
    for kk in range(0, len(COMPONENTS) + 1):
        m = k_wrong == kk
        if not m.any():
            continue
        w, r = conf[:, m][wrong[:, m]], conf[:, m][~wrong[:, m]]
        consensus[str(kk)] = {"n_items": int(m.sum()),
                              "mean_conf_wrong_components": float(w.mean()) if w.size else None,
                              "mean_conf_right_components": float(r.mean()) if r.size else None}

    _nid = ids.astype(np.int64)
    inj = (_nid < INJECTED_BELOW) | (_nid >= INJECTED_FROM)
    cohort = {}
    for i, c in enumerate(classes):
        m = y == i
        a, b = m & inj, m & ~inj
        ka, kb = int((nobody & a).sum()), int((nobody & b).sum())
        cohort[c] = {"injected_n": int(a.sum()), "injected_irreducible": ka,
                     "injected_rate": ka / int(a.sum()) if a.any() else None, "injected_ci_95": wilson(ka, int(a.sum())),
                     "other_n": int(b.sum()), "other_irreducible": kb,
                     "other_rate": kb / int(b.sum()) if b.any() else None, "other_ci_95": wilson(kb, int(b.sum()))}

    auroc = {c: auroc_confidence(conf[j], ~wrong[j]) for j, c in enumerate(COMPONENTS)}
    stage = np.asarray(off["cascade"]["stage"])
    cascade_fitted = {
        "stage_order": members,
        "thresholds_per_fold": [{"fold": th["fold"], "t1": th["t1"], "t2": th["t2"]}
                                for th in off["cascade"]["thresholds"]],
        "share_beyond_first_stage": float((stage >= 1).mean()),
        "share_to_last_stage": float((stage == len(members) - 1).mean()),
    }
    return {"n_items": int(len(y)), "classes": classes, "size_per_class": size, "construction": construction,
            "consensus_by_k_wrong": consensus, "injected_cohort": cohort, "auroc_confidence_error_detection": auroc,
            "cascade_fitted": cascade_fitted,
            "machines": {c: recs[c]["meta"].get("machine") for c in COMPONENTS}}


def main(argv=None) -> int:
    args = measuring_main_guard(__doc__, OUT, argv)
    if args is None:
        return 0
    from src.component_store import ComponentStore
    store = ComponentStore()
    cells = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for st in e3.SUBTASKS:
            cells[st] = analyse_subtask(st, e3.load_subtask(st, COMPONENTS, store))
    payload = {
        "experiment": "E5 irreducible items (Section 4.5 rework, THESIS_LOG NS 124 k-o)",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": e3._git_commit(), "seed": e3.SEED, "edition": e3.EDITION,
        "definitions": {
            "irreducible": "items no component classifies correctly (all six, primary seed)",
            "injected": "short id or 19-digit 2021 Twitter id (data_profile._is_injected, E60); marks provenance, not label quality",
            "auroc": "probability that a correct prediction has higher confidence than a wrong one (ties half), "
                     "defined on the stored calibrated pooled out-of-fold probabilities (what the cascade routes on); "
                     "no invariance to temperature scaling is claimed -- temperatures are fitted per fold",
            "ci": "95 % Wilson score interval",
            "judge_on_set": "the 2025 judge's stored label (results/e3_judge, no calls), applied by "
                            "e3_combination.assemble_judge including its fallback to the soft vote; "
                            "recovery counted where the resulting prediction equals the gold label",
        },
        "cells": cells,
    }
    args.out.write_text(json.dumps(payload, indent=1, default=float) + "\n", encoding="utf-8")
    print("wrote {}".format(args.out))
    for st, c in cells.items():
        print("  {}: irreducible {}, judge invoked on {}, AUROC {}".format(
            st, c["construction"]["n_irreducible"], c["construction"]["judge_invoked_on_set"],
            {k: round(v, 3) for k, v in c["auroc_confidence_error_detection"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
