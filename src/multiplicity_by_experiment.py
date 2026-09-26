"""
multiplicity_by_experiment.py -- Holm and BY with ONE FAMILY PER EXPERIMENT, from the stored p-values.

THE RULE (decided on 2026-09-15 on the principle, not on the results).
A family is the set of tests that together carry one claim.
The family therefore follows the claim, i.e. the experiment -- not the edition,
not the subtask:
  * E4a (m = 48): "do imbalance strategies help" -- ONE family, including the
    dbo 2026 cell, because that cell is part of the one E4a design;
  * E4b (m = 15): "what a fixed configuration costs";
  * E4c (m = 45): "the component ordering on 2026";
  * the ablations (B12, `b12_ablation`) cover TWO parent experiments -- E3 on 2025 (SQ2) and E3 stage 1 on
    2026 (Dimension 3) -- so it is two families of 18, one per parent. That is
    the same rule as E4a, not an exception to it: E4a's cells share one claim,
    the ablation's two editions do not.
E2 (m = 45) and E3/G3 (its strategy-vs-best comparisons) already follow it.

WHY FROM THE STORED p-VALUES. A paired bootstrap p-value does not depend on
the family, only its correction does. Re-running the bootstrap would reproduce
the same p-values at the same seed and cost minutes; reading them cannot drift.
`significance_e4.json` (one family, m = 108) and `b12_ablation.json` (one family,
m = 36) stay unchanged as the originally computed records; this module writes
the primary analysis beside them and states how many verdicts differ, as a
disclosure and not as an argument.

Usage
-----
python -m src.multiplicity_by_experiment --help
python -m src.multiplicity_by_experiment    # writes results/significance_e4_by_experiment.json
                                            # and results/b12_ablation_by_edition.json (refuses to overwrite)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import significance as sig
from src.harness import measuring_main_guard

RESULTS = Path(__file__).resolve().parents[1] / "results"
E4_SOURCE = RESULTS / "significance_e4.json"
B12_SOURCE = RESULTS / "b12_ablation.json"
E4_OUT = RESULTS / "significance_e4_by_experiment.json"
B12_OUT = RESULTS / "b12_ablation_by_edition.json"
ANCHOR = "THESIS_LOG NS 123 h24 (Simon, 2026-09-15); NS 124 l"


def correct_per_family(families: dict[str, list[dict]]) -> dict[str, Any]:
    """Holm and BY within each family separately. Rows need bootstrap/mcnemar p-values.

    Each row keeps the verdict it had under the ORIGINAL family in `original_multiplicity`,
    and gets the per-family correction in `multiplicity`.
    """
    out: dict[str, Any] = {"families": {}, "verdicts_differing_from_original": []}
    for name, rows in families.items():
        light = [{"bootstrap": {"p_value": r["bootstrap"]["p_value"]},
                  "mcnemar": {"p_value": r["mcnemar"]["p_value"]}} for r in rows]
        meta = sig.apply_multiplicity({name: light})
        family_rows = []
        for r, l in zip(rows, light):
            m = dict(l["multiplicity"], family=name)
            orig = r.get("multiplicity", {})
            row = {"reference": r.get("reference", r.get("a")), "treatment": r.get("treatment", r.get("b")),
                   "multiplicity": m,
                   "original_multiplicity": {k: orig.get(k) for k in ("m", "holm_bootstrap", "by_bootstrap",
                                                                      "survives_holm_bootstrap", "survives_by_bootstrap")}}
            for k in ("block", "cell", "strategy", "dropped", "edition", "subtask", "decided_by_paired_ci",
                      "treatment_minus_reference_pp", "treatment_minus_reference_ci_pp",
                      "contribution_pp", "contribution_ci_pp"):
                if k in r:
                    row[k] = r[k]
            for method in ("holm", "by"):
                key = "survives_{}_bootstrap".format(method)
                if orig.get(key) is not None and orig[key] != m[key]:
                    out["verdicts_differing_from_original"].append(
                        {"family": name, "method": method, "reference": row["reference"],
                         "treatment": row["treatment"], "p_bootstrap": m["p_bootstrap"],
                         "adjusted_original": orig.get("{}_bootstrap".format(method)),
                         "adjusted_per_experiment": m["{}_bootstrap".format(method)]})
            family_rows.append(row)
        out["families"][name] = {"m": meta["m"],
                                 "survive_holm": sum(x["multiplicity"]["survives_holm_bootstrap"] for x in family_rows),
                                 "survive_by": sum(x["multiplicity"]["survives_by_bootstrap"] for x in family_rows),
                                 "rows": family_rows}
    return out


def e4_families(src: dict) -> dict[str, list[dict]]:
    return {"e4a": src["blocks"]["e4a"], "e4b": src["blocks"]["e4b"], "e4c": src["blocks"]["e4c"]}


def b12_families(src: dict) -> dict[str, list[dict]]:
    fams: dict[str, list[dict]] = {"b12_2025": [], "b12_2026": []}
    for key, cell in sorted(src["cells"].items()):
        ed = key.rsplit("_", 1)[1]
        for vname, v in sorted(cell["variants"].items()):
            for strat, r in sorted(v.get("significance", {}).items()):
                fams["b12_" + ed].append(r)
    return fams


def main(argv=None) -> int:
    args = measuring_main_guard(__doc__, E4_OUT, argv)
    if args is None:
        return 0
    if B12_OUT.exists() and not args.force:
        print("SKIP: {} exists; re-measure with --force only if you mean to replace it.".format(B12_OUT))
        return 0
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for source, out, build, label in ((E4_SOURCE, args.out, e4_families, "E4"),
                                      (B12_SOURCE, B12_OUT, b12_families, "B12")):
        src = json.loads(source.read_text(encoding="utf-8"))
        res = correct_per_family(build(src))
        payload = {
            "experiment": "{} significance, one Holm family per experiment (primary)".format(label),
            "created": stamp, "anchor": ANCHOR,
            "source": source.name, "source_created": src["created"],
            "source_family_m": src["multiplicity"]["m"],
            "rule": "the family follows the claim (the experiment, or for B12 the parent experiment), "
                    "not the edition or the subtask; Holm on the paired bootstrap p-value is primary, BY beside it",
            "disclosure": "the source artefact was corrected as one family of m = {} before this rule was decided; "
                          "it stays unchanged. `verdicts_differing_from_original` is reported, not argued".format(
                              src["multiplicity"]["m"]),
            **res,
        }
        out.write_text(json.dumps(payload, indent=1, default=float) + "\n", encoding="utf-8")
        print("wrote {}".format(out))
        for name, f in res["families"].items():
            print("  {}: m={} Holm {} BY {}".format(name, f["m"], f["survive_holm"], f["survive_by"]))
        print("  verdicts differing from the m = {} family: {}".format(
            src["multiplicity"]["m"], len(res["verdicts_differing_from_original"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
