"""
Re-establish the E1 baselines on THIS machine, under the exact E1 protocol.

Purpose: the pinned E1 figures were produced on a Windows desktop, while every
E2-TML figure was produced on a MacBook. Any E1-vs-E2 delta is
therefore cross-platform. Running E1 here makes the pair same-platform.

The protocol mirrors notebooks/01_baseline_repro.ipynb exactly, reset_index
included. Writes to a NEW file; the pinned e1_*_baseline.json are not touched.

A delta between a figure measured here under the *current* data rules and the
desktop pin measured under the *previous* ones would be a platform change and a
rule change added together. The output therefore separates the two
generations and only ever subtracts within one of them. The pre-rule-change
pair is a recorded measurement rather than a live one, because the rules it
belongs to no longer exist in this tree; RECIPE below re-derives it.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.baselines import C2ABaseline, DBOBaseline, VIOBaselineOllama
from src.data_loading import load_split
from src.harness import (apply_data_quality_rules, env_pin,
                        evaluate_predictions, measuring_main_guard)

HOLDOUT_FRAC = 0.20
SEED = 42
RESULTS = Path(__file__).resolve().parents[1] / "results"

# The desktop figures, pinned 2026-08-07. Windows/x86-64, and measured under
# the data rules in force before 2026-08-18.
DESKTOP = {"dbo": 48.4109, "c2a": 56.4561}

# This machine under those same rules. Measured 2026-08-20 by checking out
# 6512394 of the development repository -- the last commit before
# `_drop_contradicting_labels` entered harness.py -- into a git worktree, linking the same data/codabench tree
# (`data_manifest --verify` clean, 21 files), and running this protocol against
# that harness with this venv. DBO reproduced 51.4021 and C2A 56.4561.
#
# It is a constant here for the reason the organiser figures are constants in
# e2_summary: it cannot be recomputed from anything in this tree, because the
# rules that produced it were replaced. What makes that acceptable is that it
# names its generation and carries the recipe to re-derive it.
RECIPE = ("git worktree add --detach <dir> 6512394; link experiments/data/codabench; "
          "run src/e1_platform_check.run() there with the current venv")
PRE_RULE_CHANGE = {
    "data_rules": "pre-2026-08-18: within-edition dedup, no contradicting-label drop",
    "data_rules_commit": "6512394",
    "this_machine": {"dbo": 51.4021261849572, "c2a": 56.4560563780586},
    "desktop": DESKTOP,
    "measured": "2026-08-20",
    "recipe": RECIPE,
}


def run(subtask: str, bootstrap: bool = False, **model_kwargs) -> dict:
    """
    The E1 protocol on this machine. The single implementation of it outside the
    notebook -- `src/e1_machine_baseline.py` composes its artefact from this
    rather than repeating the protocol, because two copies of a protocol drift.
    """
    # vio (E1b, 2026-09-14) is a few-shot LLM: `fit` trains nothing and `predict`
    # needs a served model, whose URL arrives through model_kwargs.
    cls = {"dbo": DBOBaseline, "c2a": C2ABaseline, "vio": VIOBaselineOllama}[subtask]
    full = load_split(subtask, "2025", "train")
    full, _ = apply_data_quality_rules(full, dedup=True, drop_overlap=False)
    train_df, val_df = train_test_split(
        full, test_size=HOLDOUT_FRAC, stratify=full["label"], random_state=SEED
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    model = cls(**model_kwargs).fit(train_df)
    metrics = evaluate_predictions(
        val_df["label"].values,
        model.predict(val_df),
        labels=list(cls.LABEL_ORDER) if hasattr(cls, "LABEL_ORDER") else None,
        bootstrap=bootstrap,
    )
    macro = 100 * metrics["macro_f1"]
    # No delta against DESKTOP here: that figure belongs to the previous data
    # rules, and subtracting across generations mixes two effects. The
    # same-generation comparison lives in the payload's platform_effect block.
    print(f"{subtask.upper():<4} this machine {macro:.4f}  "
          f"(n_train {len(train_df)}, n_val {len(val_df)})")
    return {
        "subtask": subtask,
        "macro_f1_this_machine": macro,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "metrics": metrics,
        "model_info": getattr(model, "info_", None),
    }


def main() -> int:
    args = measuring_main_guard(__doc__, RESULTS / "e1_platform_check.json")
    if args is None:
        return 0
    runs = []
    for st in ("dbo", "c2a"):
        try:
            runs.append(run(st))
        except Exception as exc:  # noqa: BLE001
            print(f"{st.upper():<4} FAILED: {type(exc).__name__}: {exc}")
            runs.append({"subtask": st, "error": f"{type(exc).__name__}: {exc}"})

    from src.component_store import compute_data_rules_id

    pin = env_pin()
    here = pin["platform"]

    # The question this file exists to answer, and the only comparison in it
    # that is sound: both sides under one set of data rules.
    platform_effect = [
        {
            "subtask": st,
            "this_machine": PRE_RULE_CHANGE["this_machine"][st],
            "desktop": PRE_RULE_CHANGE["desktop"][st],
            "delta_pp": (PRE_RULE_CHANGE["this_machine"][st]
                         - PRE_RULE_CHANGE["desktop"][st]),
            "reproduces_desktop": abs(PRE_RULE_CHANGE["this_machine"][st]
                                      - PRE_RULE_CHANGE["desktop"][st]) < 1e-3,
        }
        for st in ("dbo", "c2a")
    ]

    payload = {
        "experiment": "E1 platform check",
        "anchor": "THESIS_LOG Next Steps #53, restructured under #65",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": ("notebooks/01_baseline_repro.ipynb mirrored exactly: within-edition "
                     "dedup, no overlap drop, stratified 80/20 holdout, seed 42"),
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "env_pin": pin,

        # (1) The platform effect, both sides under the pre-2026-08-18 rules.
        # This is what Section 3.3 cites. DBO diverges because the organisers'
        # borrowed max_features=5000 cuts through 1,072 terms tied at the
        # boundary frequency and the tie order is platform-dependent; C2A, our
        # own feature space and SentenceBERT included, is bit-identical.
        "platform_effect": {
            "generation": PRE_RULE_CHANGE["data_rules"],
            "data_rules_commit": PRE_RULE_CHANGE["data_rules_commit"],
            "machines": {"this_machine": here, "desktop": "Windows/AMD64"},
            "measured": PRE_RULE_CHANGE["measured"],
            "recipe": PRE_RULE_CHANGE["recipe"],
            "runs": platform_effect,
        },

        # (2) The current generation, measured live. No delta: no desktop
        # counterpart exists under these rules, and inventing one by reaching
        # for the pin would mix a machine and a rule change.
        "current_generation": {
            "generation": "current: within-edition dedup after contradicting-label drop",
            "data_rules_id": compute_data_rules_id(),
            "machine": here,
            "desktop_counterpart": None,
            "desktop_counterpart_note": (
                "not measured; the desktop has not run these rules. Any delta "
                "against the 2026-08-07 pin would span a machine and a rule "
                "generation at once."),
            "runs": runs,
        },
    }
    out = args.out
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
