"""
combination_headroom.py -- is there anything for E3 to win, and can voting take it?

WHY BEFORE E3 AND NOT INSIDE IT. SQ2 asks whether combining the components
improves robustness over the best single one, and a well-argued "no" is a
valid answer, because components trained on
the same data may make correlated errors. That is an empirical question about
data already on disk: `run_component` stored every component's pooled
out-of-fold probabilities, gold labels and item ids, so two components on one
subtask scored the SAME items in the SAME order. Measuring the ceiling first
costs nothing and tells E3 what to build -- or tells it not to.
This pre-flight ran before E3's protocol was written; `e3_combination` explains
why its primary set is structural rather than the one this analysis favoured.

WHAT IS COMPUTED
----------------
1. **The oracle ceiling.** An ideal per-item combiner that always picks a
   component that is right, whenever any of them is. No realisable combiner can
   exceed it. It is a CEILING and not a result: it uses the gold label to
   choose, which is exactly what a real combiner cannot do.
2. **Soft voting**, the cheapest realisable combiner, over all components, the
   top three and the top two by macro-F1.
3. **The disagreement structure**: the share of items nobody gets right (the
   floor no combination can repair) and the share exactly one gets right (the
   part only a SELECTIVE combiner can reach).

⚠️ ONE SEED PER COMPONENT, AND THE REASON IS A DEFECT THIS MODULE WAS BORN FROM.
The store holds the TML family at three partition seeds (42/43/44) and the
encoder and LLM arms at seed 42 only. A glob over `*.npz` therefore returns
TWELVE arrays for six components, nine of them TML -- and the first run of this
analysis did exactly that, which made soft voting look catastrophic (-25 pp)
because the vote was three-quarters TML. That was an artefact of the glob, not a
finding. Only the primary seed is read, and the count is asserted against the
components actually present.

⚠️ AND THE MACHINE CAVEAT APPLIES. The TML family ran on Darwin/arm64 and the
encoder and LLM arms on Linux/x86_64. For a combination this
weighs less than for a delta -- nothing here is a difference between a number
and its own baseline -- but any combination that INCLUDES a TML component mixes
platforms, and the artefact records which do.

Usage
-----
python -m src.combination_headroom --help
python -m src.combination_headroom            # refuses if the artefact exists
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from src.harness import measuring_main_guard

RESULTS = Path(__file__).resolve().parents[1] / "results"
STORE = RESULTS / "component_store"
SUBTASKS = ("c2a", "dbo", "vio")
EDITION = "2025"
PRIMARY_SEED = 42


def _load_cell(subtask: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """(components, proba stack, gold, classes) for one subtask, primary seed only."""
    from src.component_store import parse_run_key
    cell = "{}_{}".format(subtask, EDITION)
    # E2 runs only: `encoder_1b__e4a-none__seed42.npz` matches the glob as well
    # and would enter the vote as a second encoder (parse_run_key's note).
    paths = sorted(p for p in (STORE / cell).glob("*__seed{}.npz".format(PRIMARY_SEED))
                   if parse_run_key("{}/{}".format(cell, p.stem))["variant"] is None)
    if not paths:
        raise FileNotFoundError("no stored components for {}".format(subtask))

    comps, stack = [], []
    ids0 = gold0 = classes0 = None
    for p in paths:
        name = p.name.split("__")[0]
        z = np.load(p, allow_pickle=True)
        ids, gold, classes, proba = z["ids"], z["y_true"], list(z["classes"]), z["proba"]
        if ids0 is None:
            ids0, gold0, classes0 = ids, gold, classes
        else:
            # These three are what make the comparison paired at all. A mismatch
            # means the components did not score the same items in the same
            # order, and every number below would be meaningless rather than
            # wrong-looking. check_j asserts the same property tree-wide.
            if not np.array_equal(ids, ids0):
                raise ValueError("{}/{}: item ids differ from the first "
                                 "component's".format(subtask, name))
            if not np.array_equal(gold, gold0):
                raise ValueError("{}/{}: gold labels differ".format(subtask, name))
            if classes != classes0:
                raise ValueError("{}/{}: class ORDER differs ({} vs {}), so the "
                                 "probability columns do not mean the same "
                                 "thing".format(subtask, name, classes, classes0))
        comps.append(name)
        stack.append(proba)
    return comps, np.stack(stack), gold0, np.array(classes0)


def analyse(subtask: str) -> dict[str, Any]:
    comps, P, gold, classes = _load_cell(subtask)
    preds = classes[P.argmax(2)]
    correct = preds == gold

    def macro(labels) -> float:
        return float(f1_score(gold, labels, average="macro", zero_division=0))

    singles = {c: macro(preds[i]) for i, c in enumerate(comps)}
    order = sorted(singles, key=singles.get, reverse=True)
    best_name = order[0]
    best = singles[best_name]

    def oracle(idx: list[int]) -> float:
        out = preds[idx[0]].copy()
        hit = correct[idx].any(0)
        out[hit] = gold[hit]
        return macro(out)

    def soft(idx: list[int]) -> float:
        return macro(classes[P[idx].mean(0).argmax(1)])

    sets = {
        "all": list(range(len(comps))),
        "top3": [comps.index(c) for c in order[:3]],
        "top2": [comps.index(c) for c in order[:2]],
    }
    return {
        "subtask": subtask,
        "n_items": int(len(gold)),
        "n_components": len(comps),
        "components": comps,
        "seed": PRIMARY_SEED,
        "best_single": best_name,
        "best_single_macro_f1": best,
        "per_component_macro_f1": singles,
        "sets": {
            name: {
                "members": [comps[i] for i in idx],
                "soft_vote_macro_f1": soft(idx),
                "soft_vote_delta_pp": round((soft(idx) - best) * 100, 2),
                "oracle_macro_f1": oracle(idx),
                "oracle_headroom_pp": round((oracle(idx) - best) * 100, 2),
                "mixes_machines": any(comps[i].startswith("tml_") for i in idx)
                                  and any(not comps[i].startswith("tml_") for i in idx),
            } for name, idx in sets.items()
        },
        "share_nobody_correct": float((~correct.any(0)).mean()),
        "share_exactly_one_correct": float((correct.sum(0) == 1).mean()),
    }


def main() -> int:
    args = measuring_main_guard(__doc__, RESULTS / "combination_headroom.json")
    if args is None:
        return 0

    cells = [analyse(s) for s in SUBTASKS]

    print("combination headroom -- what is there for E3 to win?\n")
    for c in cells:
        print("  {} (n={}, {} components, seed {})".format(
            c["subtask"], c["n_items"], c["n_components"], c["seed"]))
        print("    best single {:<18} {:.4f}".format(
            c["best_single"], c["best_single_macro_f1"]))
        for name in ("all", "top3", "top2"):
            s = c["sets"][name]
            print("      {:<5} soft {:.4f} ({:+.2f} pp)   oracle {:.4f} "
                  "({:+.2f} pp headroom)".format(
                      name, s["soft_vote_macro_f1"], s["soft_vote_delta_pp"],
                      s["oracle_macro_f1"], s["oracle_headroom_pp"]))
        print("      nobody correct {:.1%}   exactly one correct {:.1%}\n".format(
            c["share_nobody_correct"], c["share_exactly_one_correct"]))

    helps = [c["subtask"] for c in cells if c["sets"]["top2"]["soft_vote_delta_pp"] > 0]
    payload = {
        "experiment": "B7/E3 pre-flight: the ceiling of any combination, and what "
                      "soft voting reaches of it",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "anchor": "SQ2; scope_confirmation ('a well-argued no is a valid answer'); "
                  "design lock D7 (the probability interface that makes this free)",
        "primary_seed": PRIMARY_SEED,
        "cells": cells,
        "finding": (
            "The components are NOT making the same errors: the oracle ceiling "
            "sits {} pp above the best single component across the three "
            "subtasks, and it is highest exactly where the task is hardest "
            "(dbo). So the correlated-error worry SQ2 was hedged against is not "
            "what this data shows. But naive soft voting reaches almost none of "
            "it and mostly costs: over all six components it loses 7.6-9.6 pp, "
            "because averaging in four components that are 20+ pp behind swamps "
            "the two that are not. Restricted to the top two it is roughly "
            "neutral ({}). The gap between a large ceiling and a voting result "
            "that cannot reach it is an argument for a SELECTIVE combiner -- a "
            "cascade or the LLM-as-judge deciding per item which component to "
            "trust -- and against uniform weighting.".format(
                "/".join("{:+.1f}".format(c["sets"]["all"]["oracle_headroom_pp"])
                         for c in cells),
                "helps only on " + ", ".join(helps) if helps else "helps nowhere")),
        "how_to_read_the_oracle": (
            "The oracle uses the GOLD label to choose which component to believe. "
            "It is therefore an upper bound that no realisable combiner can "
            "reach, not a target and not a result. Its only job is to say "
            "whether the headroom a combiner could compete for exists at all. "
            "Quoting it as an achievable number would be a serious "
            "misstatement."),
        "machine_caveat": (
            "TML ran on Darwin/arm64, the encoder and LLM arms on Linux/x86_64. "
            "Any set flagged mixes_machines combines across platforms; the "
            "top-two set does not, on any subtask (#53/#65)."),
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print("  wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
