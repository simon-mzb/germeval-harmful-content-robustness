"""
Freeze the E4c (Dimension 3) evaluation pool before any model sees it.

Why this module exists rather than a line inside the E4c runner. Dimension 3 is
the thesis's only genuinely out-of-sample evidence: the 2025-trained components
are scored on 2026 data that no selection step touched. That claim is only true
once the cross-edition overlap is removed. GermEval 2026 is the same 2014-2016
source corpus as 2025 with added samples and a revised annotation regime, so
36-45% of every 2026 train split is verbatim 2025 train data (C2A 42.7%,
DBO 44.7%, VIO 36.2%). Evaluating against the raw 2026 split would score the
models on their own training items for nearly half the set.

Freezing the pool as an artefact -- ids, counts, class supports, reporting tiers and a
hash -- costs nothing and buys the property the evaluation design argues for:
the pool is fixed and recorded *before* any component is compared against it, so
no later choice can have been made with knowledge of it. A pool rebuilt on the
fly inside the runner would be equally correct and impossible to demonstrate.

The residual is expensive in the rare classes and the reporting tiers are written into
the artefact for exactly that reason: DBO subversive drops 53 -> 12 and VIO
glorification 27 -> 12, both into the report-only tier. A reader of the E4c table
must see that from the table, not from the prose.

Usage
-----
    python -m src.e4c_pool --write     # build and freeze
    python -m src.e4c_pool --verify    # re-derive and compare against the artefact
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from .data_loading import load_split
from .harness import apply_data_quality_rules, env_pin, tier_for_n

SUBTASKS = ("c2a", "dbo", "vio")
ARTEFACT = Path(__file__).resolve().parents[1] / "results" / "e4c_eval_pool.json"


def build_pool(subtask: str) -> dict[str, Any]:
    """
    Build the overlap-free 2026 evaluation pool for one subtask.

    Both editions get the contradiction rule and within-edition deduplication
    first, so the 2025 side that defines the overlap is exactly the pool the
    components were trained on -- `e2_runner.load_pool` applies the same two
    rules with the same defaults. Comparing against the raw 2025 file would
    remove slightly too little.

    No pool size is restated here, because a restated size goes stale the
    moment a data rule changes -- the sizes are returned in the artefact
    (`n_2025_train_dedup`) and tests/test_data_integrity.py re-derives the
    frozen pool and asserts it still holds no 2025 training text.
    """
    train_25, _ = apply_data_quality_rules(
        load_split(subtask, "2025", "train"), dedup=True, drop_overlap=False)
    train_26, _ = apply_data_quality_rules(
        load_split(subtask, "2026", "train"), dedup=True, drop_overlap=False)

    pool, _ = apply_data_quality_rules(
        train_26,
        dedup=False,  # already applied above
        drop_overlap=False,
        cross_edition_train_df=train_25,
        drop_cross_edition=True,
    )

    counts = pool["label"].value_counts().to_dict()
    return {
        "subtask": subtask,
        "n_2025_train_dedup": int(len(train_25)),
        "n_2026_train_dedup": int(len(train_26)),
        "n_removed_cross_edition": int(len(train_26) - len(pool)),
        "pct_removed": round((len(train_26) - len(pool)) / len(train_26) * 100, 2),
        "n_pool": int(len(pool)),
        "class_support": {str(k): int(v) for k, v in counts.items()},
        "class_tier": {str(k): tier_for_n(int(v)) for k, v in counts.items()},
        "ids_sha256": _hash_ids(pool),
        "ids": [str(i) for i in pool["id"].tolist()],
    }


def _hash_ids(pool) -> str:
    """Hash the sorted id list, so the pool's identity is checkable in one field."""
    joined = "\n".join(sorted(str(i) for i in pool["id"].tolist()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_all() -> dict[str, Any]:
    return {
        "created": date.today().isoformat(),
        "rule": ("2026 train, within-edition dedup, minus every text present in the "
                 "deduplicated 2025 train split (normalised strip+lower)"),
        "anchor": "THESIS_LOG Next Steps #49",
        "env_pin": env_pin(),
        "pools": [build_pool(st) for st in SUBTASKS],
    }


def _summarise(payload: dict[str, Any]) -> None:
    for p in payload["pools"]:
        print(f"\n{p['subtask'].upper()}: {p['n_2026_train_dedup']} -> {p['n_pool']} "
              f"(removed {p['n_removed_cross_edition']}, {p['pct_removed']}%)")
        for cls, n in sorted(p["class_support"].items(), key=lambda kv: -kv[1]):
            print(f"    {cls:<16} {n:>6}   [{p['class_tier'][cls]}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="build and freeze the pool")
    ap.add_argument("--verify", action="store_true", help="re-derive and compare")
    args = ap.parse_args()

    payload = build_all()

    if args.write:
        ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTEFACT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"wrote {ARTEFACT}")
        _summarise(payload)
        return

    if args.verify:
        if not ARTEFACT.exists():
            raise SystemExit(f"no artefact at {ARTEFACT} -- run --write first")
        stored = json.loads(ARTEFACT.read_text())
        ok = True
        for new, old in zip(payload["pools"], stored["pools"], strict=True):
            same = new["ids_sha256"] == old["ids_sha256"]
            ok &= same
            print(f"  {new['subtask']}: {'ok' if same else 'CHANGED'} "
                  f"(n={new['n_pool']} vs {old['n_pool']})")
        raise SystemExit(0 if ok else 1)

    _summarise(payload)


if __name__ == "__main__":
    main()
