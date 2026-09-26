"""
e5_error_analysis.py -- where the components fail, not how often.

WHY THIS IS ITS OWN MEASUREMENT. Section 4.1 reports how much each component
scores; it cannot say what kind of item it loses, nor whether the pipeline could
tell a mistake from a correct answer at inference time. Both questions are
answerable from data already on disk: `run_component` stored every component's
pooled out-of-fold probabilities, gold labels, item ids and fold assignment, so
the errors themselves are on record and cost nothing to characterise.

WHAT IS COMPUTED
----------------
1. **Confusion structure.** The full matrix plus per-class recall, so a large
   raw block of error on a frequent class is not mistaken for a high error rate.
   On the four-class subtask this is the only place the reader can see which
   boundary is soft.
2. **Whether errors are confident.** Mean confidence on correct against wrong
   predictions, the share of errors above 0.9, and a threshold sweep: at a given
   confidence cutoff, what share of errors would be escalated and what share of
   CORRECT predictions would be escalated with them. Section 3.2's cascade
   routes on exactly this quantity, so its design assumption is testable here
   and nowhere else.
3. **The irreducible set.** Items that no component classifies correctly, with
   the class distribution beside the base rate the class holds in the pool, so
   a concentration can be read against what chance would give.

⚠️ SEED 42 ONLY, AND THE REASON IS A DEFECT `combination_headroom` WAS BORN FROM.
The store holds the TML family at three partition seeds and the neural arms at
one, so a glob over `*.npz` returns TWELVE arrays for six components and makes
any cross-component statement three-quarters classical. Every component here is
read at the primary seed by name, which is also the convention Section 3.3 fixes
for paired comparisons.

⚠️ THE IRREDUCIBLE SET MIXES MACHINES. TML ran on Darwin/arm64, the encoder and
LLM arms on Linux/x86_64. An intersection over all six components is
therefore a cross-platform statement; it is carried in the artefact rather than
silently dropped, because the alternative -- restricting to one platform -- would
drop the classical arm entirely.

The artefact is committed and is not replaced without --force.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.harness import measuring_main_guard

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STORE = RESULTS / "component_store"
E2 = RESULTS / "e2"

SUBTASKS = ["c2a", "dbo", "vio"]
COMPONENTS = ["tml_svm", "tml_xgboost", "tml_lightgbm",
              "encoder_1b", "llm_llammlein", "llm_qwen_fewshot"]
MINORITY = {"c2a": "True", "dbo": "subversive", "vio": "True"}
CUTOFFS = [0.70, 0.80, 0.90, 0.95]


def read(subtask: str, component: str, seed: int):
    z = np.load(STORE / f"{subtask}_2025" / f"{component}__seed{seed}.npz",
                allow_pickle=True)
    classes = np.asarray(z["classes"]).astype(str)
    proba = np.asarray(z["proba"], dtype=np.float64)
    return (np.asarray(z["y_true"]).astype(str),
            classes[proba.argmax(axis=1)],
            proba.max(axis=1),
            np.asarray(z["ids"]),
            classes)


def confusion(y_true, y_pred, classes) -> dict:
    return {t: {p: int(((y_true == t) & (y_pred == p)).sum()) for p in classes}
            for t in classes}


def _wilson(k: int, n: int, z: float = 1.96) -> list | None:
    """95 % Wilson score interval for k of n (Brown, Cai & DasGupta 2001).

    Section 3.1 says per-class results carry
    dispersion AND support, and the recall table carried only support. Without an
    interval the 60-item `subversive` column cannot support the directional claim
    the prose makes on it, and Section 3.3's rule applies within a component just
    as it does between components.
    """
    if not n:
        return None
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / (1 + z2 / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def per_class(y_true, y_pred, classes) -> dict:
    kw = dict(labels=list(classes), average=None, zero_division=0)
    out = {}
    for c, r, p, f in zip(classes,
                          recall_score(y_true, y_pred, **kw),
                          precision_score(y_true, y_pred, **kw),
                          f1_score(y_true, y_pred, **kw)):
        support = int((y_true == c).sum())
        out[c] = {"support": support, "recall": float(r), "precision": float(p), "f1": float(f),
                  "recall_ci_95": _wilson(int(round(float(r) * support)), support)}
    return out


def confidence_block(y_true, y_pred, conf) -> dict:
    wrong = y_true != y_pred
    right = ~wrong
    out = {
        "n_wrong": int(wrong.sum()),
        "n_right": int(right.sum()),
        "mean_confidence_right": float(conf[right].mean()) if right.any() else None,
        "mean_confidence_wrong": float(conf[wrong].mean()) if wrong.any() else None,
        "share_of_errors_above_0.9": float((conf[wrong] > 0.9).mean()) if wrong.any() else None,
        "escalation_sweep": {},
    }
    # What a cascade would actually do: escalate everything below the cutoff.
    for t in CUTOFFS:
        low = conf < t
        out["escalation_sweep"]["{:.2f}".format(t)] = {
            "errors_escalated": float(low[wrong].mean()) if wrong.any() else None,
            "correct_escalated": float(low[right].mean()) if right.any() else None,
            "share_of_pool_escalated": float(low.mean()),
        }
    return out


def main() -> int:
    args = measuring_main_guard(__doc__, RESULTS / "e5_error_analysis.json")
    if args is None:
        return 0

    cells = []
    for s in SUBTASKS:
        meta = json.loads((E2 / f"e2_{s}_2025_encoder_1b.json").read_text())["meta"]
        seed = meta["primary_seed"]
        base = None
        components, wrong_masks, ids_ref = {}, [], None
        for c in COMPONENTS:
            y, pred, conf, ids, classes = read(s, c, seed)
            if ids_ref is None:
                ids_ref, base = ids, Counter(y.tolist())
            elif not np.array_equal(ids, ids_ref):
                raise SystemExit(
                    "{}/{}: the stored item ids differ from the first component's. "
                    "Every statement below assumes the same items in the same "
                    "order, so this is refused rather than worked around.".format(s, c))
            wrong_masks.append(y != pred)
            components[c] = {
                "confusion": confusion(y, pred, classes),
                "per_class": per_class(y, pred, classes),
                "confidence": confidence_block(y, pred, conf),
                "largest_confusion_blocks": [
                    {"true": t, "predicted": p, "n": n}
                    for t, p, n in sorted(
                        ((t, p, int(((y == t) & (pred == p)).sum()))
                         for t in classes for p in classes if t != p),
                        key=lambda r: -r[2])[:3]],
            }

        nobody = np.logical_and.reduce(wrong_masks)
        y0, _, _, _, classes = read(s, COMPONENTS[0], seed)
        n = len(y0)
        cells.append({
            "subtask": s,
            "seed": seed,
            "n_items": n,
            "classes": list(classes),
            "class_base_rates": {c: base[c] / n for c in classes},
            "minority_class": MINORITY[s],
            "components": components,
            "nobody_correct": {
                "n": int(nobody.sum()),
                "share_of_pool": float(nobody.mean()),
                "class_counts": {c: int(((y0 == c) & nobody).sum()) for c in classes},
                "class_shares_within_set": {
                    c: (float(((y0 == c) & nobody).sum() / nobody.sum())
                        if nobody.sum() else None) for c in classes},
            },
        })

    payload = {
        "experiment": "E5 error analysis: the shape of the errors, over E2's stored predictions",
        "created": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "anchor": "Chapter 4 error analysis; the cascade's escalation rule (3.2); "
                  "the per-class reporting bands (3.3)",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                     capture_output=True, text=True).stdout.strip(),
        "primary_seed_only": (
            "Every component is read at its primary seed by name. The store holds "
            "the TML family at three partition seeds, so a glob would return twelve "
            "arrays for six components and make any cross-component statement "
            "three-quarters classical."),
        "machine_caveat": (
            "TML ran on Darwin/arm64, the encoder and LLM arms on Linux/x86_64 "
            "(#53/#65). `nobody_correct` intersects all six and is therefore a "
            "cross-platform statement. Restricting it to one platform would drop "
            "the classical arm entirely, so the caveat travels with the number."),
        "how_to_read_the_sweep": (
            "`errors_escalated` is the share of a component's WRONG predictions "
            "that fall below the cutoff, i.e. what a cascade would catch. "
            "`correct_escalated` is the share of its CORRECT predictions that fall "
            "below the same cutoff, i.e. what the same rule costs. Neither is a "
            "result about the cascade, which does not exist yet; they bound what "
            "any confidence threshold can do on these predictions."),
        "cells": cells,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print("  wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
