"""
significance_e4.py -- the paired significance layer over the E4 cells (Dimensions 1-3).

WHAT IT ANSWERS. Whether an E4 figure differs from the figure it is set against,
under the same decision rule Section 3.3 fixes for the component comparisons:
the paired bootstrap interval of the difference decides, McNemar is reported
beside it, and the reporting tiers gate any minority-class statement. Nothing is
re-run; `significance.compare` and `apply_multiplicity` are reused unchanged.

THE REFERENCES ARE FIXED, NOT DISCOVERED (written before any E4 p-value
existed):
1. **E4a** -- every condition against the SAME component's E4a `none` cell on
   the same subtask and edition. E4a runs three folds, so a model trains on two
   thirds of the pool; a five-fold E2 figure is never its reference.
2. **E4b** -- every fixed-configuration cell against its own E2 run: same
   five-fold partition, same items, so the difference IS Dimension 2's penalty.
3. **E4c** -- every pair of the six components on the frozen 2026 pool. It is
   the only paired test E4c admits. A component's own 2025 -> 2026 change is
   measured on different items and is therefore NOT tested here: it is reported
   as the two artefact intervals side by side (`e4c_edition_contrast`).

⚠️ SUPERSEDED AS PRIMARY: the primary analysis is one
family PER EXPERIMENT, in `multiplicity_by_experiment`. The paragraph below describes what
this module computed and still writes, and the artefact stays as that record.

ONE FAMILY. Holm (primary) and BY run over all comparisons of all three blocks
together. Per-block families are defensible and each is easier to clear, i.e.
each would favour this thesis -- so the widest one is taken, as 3.3 does for
the 45 component pairs. Its cost, stated before the run: a gain that would
survive within E4a alone can fail here.

SIGN CONVENTION. `compare(reference, treatment)` reports a - b. This module adds
`treatment_minus_reference_pp` and its interval so a reader never flips a sign
by hand: positive means the condition / fixed configuration / second component
scores higher.

Usage
-----
python -m src.significance_e4 --help
python -m src.significance_e4            # writes results/significance_e4.json (refuses to overwrite)
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from src import significance as sig
from src.component_store import ComponentStore, parse_run_key
from src.harness import measuring_main_guard

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "significance_e4.json"
SEED = sig.PRIMARY_SEED
BLOCKS = ("e4a", "e4b", "e4c")


def plan_pairs(keys: Iterable[str], seed: int = SEED) -> list[dict[str, str]]:
    """Every E4 comparison the keys admit, with its reference fixed by block.

    Refuses an E4a condition without its `none` cell and an E4b cell without its
    E2 run: a missing reference must stop the layer, not silently shrink the
    family (a smaller family is an easier one).
    """
    parsed = {}
    for k in keys:
        try:
            p = parse_run_key(k)
        except ValueError:
            continue
        if p["seed"] == seed:
            parsed[k] = p
    by_key = {(p["cell"], p["component"], p["variant"]): k for k, p in parsed.items()}
    plan: list[dict[str, str]] = []
    for k, p in sorted(parsed.items()):
        v = p["variant"]
        if v and v.startswith("e4a-") and v != "e4a-none":
            ref = by_key.get((p["cell"], p["component"], "e4a-none"))
            if ref is None:
                raise ValueError("{}: no E4a none cell to compare against".format(k))
            plan.append({"block": "e4a", "cell": p["cell"], "reference": ref, "treatment": k})
        elif v == "e4b":
            ref = by_key.get((p["cell"], p["component"], None))
            if ref is None:
                raise ValueError("{}: no E2 run to compare against".format(k))
            plan.append({"block": "e4b", "cell": p["cell"], "reference": ref, "treatment": k})
    e4c: dict[str, list[str]] = {}
    for k, p in parsed.items():
        if p["variant"] == "e4c":
            e4c.setdefault(p["cell"], []).append(k)
    for cell, ks in sorted(e4c.items()):
        for a, b in combinations(sorted(ks), 2):
            plan.append({"block": "e4c", "cell": cell, "reference": a, "treatment": b})
    return plan


def _load_pair(store: ComponentStore, a: str, b: str) -> tuple[dict, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # machine mix: `compare` records it
        recs = store.load_many([a, b])
    return recs[a], recs[b]


def run_layer(store: ComponentStore | None = None, n_bootstrap: int = sig.SIG_BOOTSTRAP_N
              ) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    store = store or ComponentStore()
    plan = plan_pairs(store.list_runs())
    blocks: dict[str, list[dict]] = {b: [] for b in BLOCKS}
    for i, q in enumerate(plan, 1):
        ra, rb = _load_pair(store, q["reference"], q["treatment"])
        r = sig.compare(ra, rb, key_a=q["reference"], key_b=q["treatment"],
                        n_bootstrap=n_bootstrap, seed=SEED)
        bs = r["bootstrap"]
        lo, hi = bs["ci_difference"]
        r.update({
            "block": q["block"], "cell": q["cell"],
            "reference": q["reference"], "treatment": q["treatment"],
            "treatment_minus_reference_pp": round(100 * (bs["macro_f1_b"] - bs["macro_f1_a"]), 2),
            "treatment_minus_reference_ci_pp": [round(-100 * hi, 2), round(-100 * lo, 2)],
            "decided_by_paired_ci": ("distinguishable" if bs["difference_ci_excludes_zero"]
                                     else "indistinguishable"),
        })
        blocks[q["block"]].append(r)
        print("  [{}/{}] {} {} -> {} {:+.2f} pp [{:+.2f}, {:+.2f}] {}".format(
            i, len(plan), q["block"], q["reference"], q["treatment"].split("/")[1],
            r["treatment_minus_reference_pp"], *r["treatment_minus_reference_ci_pp"],
            r["decided_by_paired_ci"]))
    mult = sig.apply_multiplicity(blocks)
    family = "every E4 comparison in this artefact (E4a, E4b, E4c together), fixed in THESIS_LOG NS 124 d"
    mult["family"] = family
    mult["family_choice_note"] = (
        "ONE family over all three blocks (m={}). Per-block families were defensible and each is "
        "easier to clear, i.e. each would favour the thesis; the widest one is taken, as 3.3 does "
        "for the component pairs.".format(mult["m"]))
    for rows in blocks.values():
        for r in rows:
            r["multiplicity"]["family"] = family
    return blocks, mult


def e4c_edition_contrast() -> list[dict[str, Any]]:
    """Per component and subtask: the E2 2025 interval beside the E4c 2026 one. Not a test."""
    out = []
    for path in sorted((RESULTS / "e4c").glob("e4c_*.json")):
        a = json.loads(path.read_text(encoding="utf-8"))
        e2 = RESULTS / "e2" / "e2_{}_2025_{}.json".format(a["subtask"], a["component"])
        if not e2.exists():
            continue
        b = json.loads(e2.read_text(encoding="utf-8"))
        p25, p26 = b["primary"], a["primary"]
        out.append({
            "subtask": a["subtask"], "component": a["component"],
            "macro_f1_2025_e2": p25["macro_f1"],
            "ci_2025": [p25.get("macro_f1_ci_lower"), p25.get("macro_f1_ci_upper")],
            "n_2025": p25.get("n_samples"),
            "macro_f1_2026_e4c": p26["macro_f1"],
            "ci_2026": [p26.get("macro_f1_ci_lower"), p26.get("macro_f1_ci_upper")],
            "n_2026": p26.get("n_samples"),
            "change_pp": round(100 * (p26["macro_f1"] - p25["macro_f1"]), 2),
            "intervals_overlap": not (p26.get("macro_f1_ci_upper") < p25.get("macro_f1_ci_lower")
                                      or p25.get("macro_f1_ci_upper") < p26.get("macro_f1_ci_lower")),
        })
    return out


def main(argv=None) -> int:
    args = measuring_main_guard(__doc__, OUT, argv)
    if args is None:
        return 0
    blocks, mult = run_layer()
    payload = {
        "experiment": "E4 significance layer (THESIS_LOG NS 124 d)",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "bootstrap_n": sig.SIG_BOOTSTRAP_N,
        "ci_level": sig.CI_LEVEL,
        "decision_rule": ("3.3: the paired bootstrap interval of the difference decides "
                          "(`decided_by_paired_ci`); McNemar is reported beside it; Holm on the "
                          "paired p-value over the one family below is the multiplicity correction"),
        "references": {
            "e4a": "the same component's E4a none cell, same subtask and edition, 3 folds; never E2",
            "e4b": "the same component's E2 run, same 5-fold partition (Dimension 2 penalty)",
            "e4c": "every pair of components on the frozen 2026 pool; no cross-edition test",
        },
        "counts": {b: len(v) for b, v in blocks.items()},
        "multiplicity": mult,
        "blocks": blocks,
        "e4c_edition_contrast": {
            "note": "UNPAIRED: 2025 E2 pooled OOF vs 2026 E4c, different items; two intervals side "
                    "by side, no test, and overlap implies nothing (schenker2001)",
            "rows": e4c_edition_contrast(),
        },
    }
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=float) + "\n",
                        encoding="utf-8")
    print("wrote {} ({} comparisons, one family)".format(args.out, mult["m"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
