r"""
test_reported_numbers_rebuild.py -- rebuild every reported number from the raw
per-item output, and refuse any artefact whose headline figure does not follow
from its own probabilities.

WHY THIS IS THE BOTTOM OF THE CHAIN. Every figure reported from a result
artefact treats that artefact as ground truth: it reads `primary.macro_f1` and
formats it. If an artefact's headline number did not follow from the
probabilities the run actually produced, every downstream check would
propagate the wrong figure faithfully and call it verified. A one-off manual
rebuild is not a guard, so the rebuild is a suite.

The ground truth here is the `.npz` in `results/component_store/`: the per-item
probability matrix, the gold labels, the item ids, the class ordering and the
fold assignment, written by the run itself on the machine that produced it. This
suite recomputes the reported figures from those arrays with scikit-learn and
compares. It reads no JSON figure except the ones it is checking.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "results" / "e2"
STORE = ROOT / "results" / "component_store"

SUBTASKS = ["c2a", "dbo", "vio"]
COMPONENTS = ["tml_svm", "tml_xgboost", "tml_lightgbm",
              "encoder_1b", "llm_llammlein", "llm_qwen_fewshot"]
MINORITY = {"c2a": "True", "dbo": "subversive", "vio": "True"}

# Float32 probabilities and a different argmax tie-break are the only movement
# a correct pipeline can produce here. Anything above this is a real difference.
TOL = 5e-10

failures: list[str] = []
checks = 0


def rebuild(npz: Path):
    z = np.load(npz, allow_pickle=True)
    classes = np.asarray(z["classes"]).astype(str)
    y_true = np.asarray(z["y_true"]).astype(str)
    y_pred = classes[np.asarray(z["proba"]).argmax(axis=1)]
    return y_true, y_pred, classes, np.asarray(z["ids"]), np.asarray(z["fold"])


def ece_from(npz: Path, n_bins: int) -> float:
    """Expected calibration error on the top-class confidence, equal-width bins."""
    z = np.load(npz, allow_pickle=True)
    proba = np.asarray(z["proba"], dtype=np.float64)
    classes = np.asarray(z["classes"]).astype(str)
    y_true = np.asarray(z["y_true"]).astype(str)
    conf = proba.max(axis=1)
    correct = (classes[proba.argmax(axis=1)] == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def check(label: str, got: float, want: float, tol: float = TOL) -> None:
    global checks
    checks += 1
    if not np.isfinite(got) or abs(got - want) > tol:
        failures.append(f"{label}: artefact says {want!r}, the stored probabilities "
                        f"give {got!r} (delta {abs(got - want):.3e})")


def main() -> int:
    for s in SUBTASKS:
        for c in COMPONENTS:
            art = json.loads((E2 / f"e2_{s}_2025_{c}.json").read_text())
            reps = art.get("repeats") or []
            for rep in reps:
                seed = rep["seed"]
                npz = STORE / f"{s}_2025" / f"{c}__seed{seed}.npz"
                if not npz.exists():
                    failures.append(f"{s}/{c}/seed{seed}: no stored probabilities at "
                                    f"{npz.relative_to(ROOT)}; the reported figure "
                                    f"rests on nothing this suite can see")
                    continue
                y_true, y_pred, classes, ids, fold = rebuild(npz)

                check(f"{s}/{c}/seed{seed} pooled macro-F1",
                      f1_score(y_true, y_pred, average="macro", labels=classes,
                               zero_division=0),
                      rep["pooled_macro_f1"])

                per_class = dict(zip(classes, f1_score(
                    y_true, y_pred, average=None, labels=classes, zero_division=0)))
                for cls, want in rep["per_class_f1"].items():
                    check(f"{s}/{c}/seed{seed} F1[{cls}]", per_class[cls], want)

                # the pool the figure was computed over, and the fold layout
                check(f"{s}/{c}/seed{seed} n_items", float(len(ids)),
                      float(art["meta"]["pool_n"]), tol=0)
                check(f"{s}/{c}/seed{seed} distinct ids", float(len(set(ids.tolist()))),
                      float(len(ids)), tol=0)
                check(f"{s}/{c}/seed{seed} fold count",
                      float(len(set(fold.tolist()))),
                      float(art["meta"]["n_splits"]), tol=0)

                # calibration, recomputed from the same probabilities
                bins = art["primary"]["calibration"]["n_bins"]
                if seed == art["meta"]["primary_seed"]:
                    check(f"{s}/{c}/seed{seed} ECE",
                          ece_from(npz, bins), rep["ece"], tol=5e-3)

            # the headline figure the prose checker reads
            primary = next((r for r in reps if r["seed"] == art["meta"]["primary_seed"]),
                           None)
            if primary is not None:
                check(f"{s}/{c} primary.macro_f1 equals its own seed's pooled figure",
                      art["primary"]["macro_f1"], primary["pooled_macro_f1"])
                check(f"{s}/{c} primary minority F1",
                      art["primary"]["per_class"][MINORITY[s]]["f1"],
                      primary["per_class_f1"][MINORITY[s]])

    print(f"{checks} reported figures rebuilt from the stored per-item probabilities")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        print(f"\n{len(failures)} failure(s).")
        return 1
    print("Every reported figure follows from the probabilities the runs produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
