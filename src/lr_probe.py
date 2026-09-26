"""lr_probe.py -- is the learning-rate grid's optimum on its own edge?

WHY THIS EXISTS. A first encoder measurement selected `lr = 2e-4` in 4 of 4
fits, and 2e-4 was the TOP of the then-configured grid `[1e-4, 2e-4]`. When
every fit lands on a boundary, the grid decided, not the data -- the same
finding that widened the SVM's C grid. Discovering it *after* an eight-hour
campaign means paying for the arm twice, so this settles it first, on one fold,
for about twenty minutes of A40 time.

WHAT IT DOES. One subtask, one fold, `lora_r` fixed, `lr` swept across a window
that brackets the current grid on BOTH sides. It reports the selection-slice
macro-F1 per lr and says whether the winner is interior.

WHAT IT DOES NOT DO. It does not produce a thesis number and does not touch the
component store or `results/e2/`. It is one fold at one seed: enough to see
whether the peak is inside the window, not enough to pin a value. If the winner
is interior, the existing two-point grid is defensible with the argument §3.3
already makes for the SVM; if it is on the edge again, the window moves before
the campaign starts.

Usage:
  .venv/bin/python -m src.lr_probe --subtask c2a --lrs 1e-4 2e-4 3e-4 5e-4
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import encoder_components as enc
# load_pool lives in e2_runner, not data_loading -- it is the E2 pool
# (edition-aware, quality rules applied by the caller), so the probe uses
# exactly the same entry point the campaign does.
from src.e2_runner import load_pool
from src.harness import (apply_data_quality_rules, env_pin, make_cv_splits,
                         resolve_classes)

OUT = Path(__file__).resolve().parent.parent / "results" / "measurements"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", default="c2a")
    ap.add_argument("--edition", default="2025")
    ap.add_argument("--component", default="encoder_1b")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[1e-4, 2e-4, 3e-4, 5e-4])
    ap.add_argument("--max-epochs", type=int, default=5)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing probe artefact for this cell")
    args = ap.parse_args()

    # The refusal comes BEFORE the twenty minutes of A40 time, not before the
    # write. These artefacts are the measured GPU unit costs
    # every arm budget is sized on; replacing one is a decision.
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "lr_probe_{}_{}_{}.json".format(
        args.subtask, args.edition, args.component)
    if out.exists() and not args.force:
        print("SKIP: {} already exists and this probe measures rather than "
              "reads.\n      Re-measure with --force only if you mean to "
              "replace it.".format(out))
        return 0

    device = enc.resolve_device(None)
    print("lr probe: {} / {} / {}  fold {}  device {}".format(
        args.component, args.subtask, args.edition, args.fold, device))
    print("  lora_r fixed at {}, sweeping lr over {}".format(
        args.lora_r, ["{:.0e}".format(x) for x in args.lrs]))

    pool = load_pool(args.subtask, args.edition)
    classes = resolve_classes(pool["label"].values)
    splits = list(make_cv_splits(pool, n_splits=args.n_splits, seed=args.seed))
    train_idx, val_idx = splits[args.fold]
    train_df = pool.iloc[train_idx].copy().reset_index(drop=True)
    val_df = pool.iloc[val_idx].copy().reset_index(drop=True)
    train_df, _ = apply_data_quality_rules(train_df, val_df, dedup=True, drop_overlap=True)
    print("  train n={}  classes={}".format(len(train_df), list(classes)))

    rows = []
    for lr in args.lrs:
        t0 = time.time()
        grid = [{"lora_r": args.lora_r, "lr": lr}]
        fit = enc.make_train_fn(args.component, classes, seed=args.seed,
                                device=device, max_epochs=args.max_epochs,
                                grid=grid)
        model = fit(train_df)
        rec = model.record
        score = float(rec["selection_macro_f1"])
        rows.append({
            "lr": lr,
            "selection_macro_f1": score,
            "selected_epoch": rec.get("selected_epoch"),
            "fit_diagnosis": rec.get("fit_diagnosis"),
            "seconds": round(time.time() - t0, 1),
        })
        print("  lr {:.0e}: selection macro-F1 {:.4f}  (epoch {}/{}, {:.0f}s)".format(
            lr, score, rec.get("selected_epoch"), args.max_epochs, rows[-1]["seconds"]))
        del model, fit

    best = max(rows, key=lambda r: r["selection_macro_f1"])
    lrs = [r["lr"] for r in rows]
    interior = best["lr"] not in (min(lrs), max(lrs))
    print()
    print("  best lr = {:.0e} at {:.4f}".format(best["lr"], best["selection_macro_f1"]))
    if interior:
        print("  INTERIOR -- the window brackets the optimum. A grid that spans it")
        print("  is defensible; the data chose, not the boundary.")
    else:
        print("  ON THE EDGE -- the window must MOVE before the campaign runs,")
        print("  or the campaign's answer is again 'the grid decided'.")

    out.write_text(json.dumps({
        "probe": "lr", "subtask": args.subtask, "edition": args.edition,
        "component": args.component, "fold": args.fold, "n_splits": args.n_splits,
        "seed": args.seed, "lora_r": args.lora_r, "max_epochs": args.max_epochs,
        "rows": rows, "best_lr": best["lr"], "optimum_is_interior": interior,
        "env_pin": env_pin(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("one fold, one seed: enough to see whether the peak is inside "
                 "the window, not enough to pin a value. Not a thesis number."),
    }, indent=2), encoding="utf-8")
    print("  wrote {}".format(out))
    return 0 if interior else 2


if __name__ == "__main__":
    raise SystemExit(main())
