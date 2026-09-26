"""test_encoder_cell.py -- the ENCODER half of run_component, on CPU.

`test_resume_integration.py` covers `run_component` for the TML family and says
in as many words that the encoder is left out because it needs CUDA. That was
true of the *numbers* and never of the *plumbing*, and the plumbing is where the
campaign's defects have actually been: the seed replicate loop, the fit-diagnosis
reporting, `fold_runtime_seconds_sum`, the per-epoch telemetry, the store
sidecar, the checkpoint cleanup. All of it ran only on paid hardware until
2026-08-28, and the check that finally exercised it was a scratch script that
was not kept -- which is the same defect one level up: a verification nobody can
re-run is a verification that happened once.

So this runs a REAL encoder cell end to end on ModernGBERT-134M, on a real (tiny)
GermEval slice, on CPU, and then feeds the artefact it produces through the
POST-CAMPAIGN path -- `e2_summary.collect_runs / summary_table / per_class_table
/ baseline_comparison` and the code the encoder notebook runs. That path had
never seen a real encoder artefact, and a break there wastes days after the
money is spent.

It is the slowest suite in the directory (~3 minutes). That is deliberate and it
is cheap against one wasted A40 hour.

Run: .venv/bin/python -m tests.test_encoder_cell
"""
from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import e2_runner, encoder_components as enc
from src.component_store import ComponentStore, compute_config_id
from src.e2_runner import load_pool, run_component, selection_diagnosis_for
from src.harness import make_cv_splits, resolve_classes

_PASS, _FAIL = 0, 0

COMPONENT = "encoder_134m"          # the 1B's protocol, at a size CPU can carry
SUBTASK, EDITION = "dbo", "2025"    # the four-class path, which is the risky one
N_SPLITS = 2
MAX_EPOCHS = 2
# Two cells on ONE axis, so the grid search really selects and the selection
# diagnosis has something to read. The protocol is untouched: this is a --grid
# override of the same kind `src.encoder_smoke` uses, not a matrix change.
GRID = [{"lora_r": 8, "lr": 3.0e-4}, {"lora_r": 8, "lr": 5.0e-4}]


def ok(m):
    global _PASS
    _PASS += 1
    print("  \033[32mPASS\033[0m  " + m)


def bad(m):
    global _FAIL
    _FAIL += 1
    print("  \033[31mFAIL\033[0m  " + m)


def tiny_pool(per_class=36):
    """A slice small enough for CPU that still holds all four DBO classes.

    Stratified by hand rather than sampled: `subversive` is 0.8% of the pool and
    a random slice of 200 would usually contain none, which would turn the only
    four-class test in the suite into a three-class one without saying so.
    """
    df = load_pool(SUBTASK, EDITION)
    parts = []
    for c in sorted(set(df["label"].astype(str))):
        sub = df[df["label"].astype(str) == c]
        parts.append(sub.head(min(per_class, len(sub))))
    out = pd.concat(parts).sample(frac=1.0, random_state=0).reset_index(drop=True)
    out["label"] = out["label"].astype(str)
    return out


def run_cell(tmp: Path, df):
    original_pool = e2_runner.load_pool
    original_grid = enc.GRIDS[COMPONENT]
    e2_runner.load_pool = lambda st, ed: df
    enc.GRIDS[COMPONENT] = GRID
    try:
        store = ComponentStore(root=tmp / "store")
        res = run_component(SUBTASK, EDITION, COMPONENT, n_splits=N_SPLITS,
                            max_epochs=MAX_EPOCHS, device="cpu", bootstrap_n=50,
                            store=store, out_dir=tmp / "out", ckpt_root=tmp / "ck")
        return res, store
    finally:
        e2_runner.load_pool = original_pool
        enc.GRIDS[COMPONENT] = original_grid


def _ckpt_snapshot() -> frozenset:
    root = Path(e2_runner.CKPT_ROOT)
    if not root.exists():
        return frozenset()
    return frozenset((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                     for p in root.rglob("*") if p.is_file())


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    # UNCHANGED, not EMPTY: a GPU campaign copies fold checkpoints into this
    # root. Do not run the suite while checkpoints are being copied in.
    ckpt_before = _ckpt_snapshot()
    print("\n=== the encoder half of run_component, on CPU ===")
    print("  (ModernGBERT-134M, {} folds + 2 replicates, {} grid cells, "
          "max_epochs {} -- a few minutes)".format(N_SPLITS, len(GRID), MAX_EPOCHS))
    try:
        df = tiny_pool()
        classes = resolve_classes(df["label"].values)
        assert len(classes) == 4, "the slice lost a class: {}".format(classes)
        res, store = run_cell(tmp, df)

        # --- the artefact's own shape ------------------------------------
        try:
            assert res["package"] == "B4" and res["meta"]["family"] == "encoder"
            assert res["meta"]["n_splits"] == N_SPLITS
            assert res["meta"]["config_id"] == compute_config_id(COMPONENT)
            assert len(res["repeats"]) == 1 and res["repeats"][0]["seed"] == 42, \
                "the encoder family must run ONE partition seed (D3)"
            assert [r["model_seed"] for r in res["seed_replicates"]] == [43, 44], \
                "the D3 replicates are seeds 43 and 44"
            ok("one partition seed, two training-seed replicates (D3 shape)")
        except AssertionError as e:
            bad("artefact shape: {}".format(e))

        # --- (a) out-of-fold construction, on a run we just produced ------
        try:
            rec = store.load("{}_{}/{}__seed42".format(SUBTASK, EDITION, COMPONENT))
            ids = np.asarray(rec["ids"])
            assert np.array_equal(ids, np.asarray(df["id"].values, dtype=np.int64))
            assert len(set(ids.tolist())) == len(ids)
            want = {int(i): str(l) for i, l in zip(df["id"].values, df["label"].values)}
            assert all(want[int(i)] == y for i, y in zip(ids, rec["y_true"]))
            assert np.allclose(rec["proba"].sum(axis=1), 1.0, atol=1e-5)
            fold = np.asarray(rec["fold"])
            assert (fold >= 0).all()
            expected = np.full(len(df), -1, dtype=np.int16)
            for f, (_, v) in enumerate(make_cv_splits(df, N_SPLITS, 42)):
                expected[v] = f
            assert np.array_equal(fold, expected)
            assert rec["classes"] == [str(c) for c in classes]
            assert rec["meta"]["machine"] and rec["meta"]["data_rules_id"]
            ok("a fresh encoder run: ids once each, labels right, rows sum to 1, folds reproduce")
        except AssertionError as e:
            bad("out-of-fold construction: {}".format(e))

        # --- (c) the nested protocol, as the fold records report it -------
        try:
            for rep in res["repeats"]:
                for i, d in enumerate(rep["folds_detail"]):
                    n = d["n_fit"] + d["n_select"] + d["n_calibrate"]
                    assert abs(d["n_fit"] / n - 0.80) < 0.02, (i, d["n_fit"], n)
                    assert abs(d["n_select"] / n - 0.10) < 0.02, (i, d["n_select"], n)
                    assert abs(d["n_calibrate"] / n - 0.10) < 0.02, (i, d["n_calibrate"], n)
                    assert d["inner_split_stratified"] is True, \
                        "fold {} fell back to an unstratified inner split".format(i)
                    assert d["temperature"]["temperature"] > 0
                    assert d["lora"]["alpha"] == 2 * d["lora"]["r"], "alpha is not tied to 2r"
                    assert d["adapted_modules"] > 0
                    assert d["precision"] == "fp32" and d["device"] == "cpu"
            ok("every fold records an 80/10/10 stratified inner split and a fitted temperature")
        except AssertionError as e:
            bad("nested protocol: {}".format(e))

        # --- v1.7's epoch code, on the real training path -----------------
        try:
            for rep in res["repeats"]:
                for i, d in enumerate(rep["folds_detail"]):
                    assert len(d["selection_grid"]) == len(GRID), \
                        "fold {} searched {} cells".format(i, len(d["selection_grid"]))
                    for cell in d["selection_grid"]:
                        trace = cell["epoch_trace"]
                        assert trace, "a grid cell recorded no epoch trace"
                        for ep in trace:
                            for key in ("epoch", "selection_macro_f1", "train_loss",
                                        "grad_norm", "pred_class_counts", "lr", "seconds"):
                                assert key in ep, "epoch telemetry is missing {}".format(key)
                            assert sum(ep["pred_class_counts"]) == d["n_select"]
                        stop = cell["stopping"]
                        assert stop["ceiling"] == MAX_EPOCHS
                        assert stop["patience"] == enc.PATIENCE
                        assert stop["epochs_run"] == len(trace)
                        # ⚠️ NOT "stopped early OR reached the ceiling". That
                        # disjunction is TRUE BY CONSTRUCTION at this ceiling and
                        # would be exactly the vacuous criterion review 4 found in
                        # encoder_smoke: with patience 2 a fit cannot stop before
                        # epoch 3, so at a ceiling of 2 the early-stopping branch
                        # is unreachable and a test that accepts either branch is
                        # testing nothing. State the reachability as a fact and
                        # check the real contract instead.
                        assert MAX_EPOCHS <= enc.PATIENCE, (
                            "this cell's ceiling is now high enough for early "
                            "stopping, so the assertion below is the wrong one")
                        assert stop["stopped_early"] is False, \
                            "a fit stopped early at a ceiling where that is impossible"
                        assert stop["epochs_run"] == MAX_EPOCHS
                        assert stop["ceiling_bound"] == (cell["best_epoch"] == MAX_EPOCHS)
                    assert 1 <= d["selected_epoch"] <= MAX_EPOCHS
            ok("per-epoch telemetry complete; at a ceiling of {} early stopping is "
               "unreachable and no fit claimed it".format(MAX_EPOCHS))
        except AssertionError as e:
            bad("epoch code: {}".format(e))

        # --- and the branch the cell above CANNOT reach -------------------
        # One extra fit, at a ceiling that makes early stopping possible, so the
        # v1.7 stopping rule is exercised rather than assumed. The contract is
        # exact in both directions and does not depend on which branch this
        # particular slice happens to take.
        try:
            from transformers import AutoTokenizer
            from src.tml_components import _three_way_split

            fit_df, sel_df, _, _ = _three_way_split(df, 42)
            tok = AutoTokenizer.from_pretrained(enc.MODEL_IDS[COMPONENT])
            fit_ids, _ = enc._encode(tok, enc._texts(fit_df), enc.MAX_SEQ_LEN)
            sel_ids, _ = enc._encode(tok, enc._texts(sel_df), enc.MAX_SEQ_LEN)
            index = {c: i for i, c in enumerate(classes)}
            y_fit = np.array([index[v] for v in fit_df["label"].values])
            y_sel = np.array([index[v] for v in sel_df["label"].values])
            ceiling = enc.PATIENCE + 2
            _, best_epoch, _, trace, stop = enc._fit_one(
                COMPONENT, GRID[0], len(classes), fit_ids, y_fit, sel_ids, y_sel,
                tok.pad_token_id, 42, "cpu", ceiling)
            assert stop["ceiling"] == ceiling and stop["patience"] == enc.PATIENCE
            assert stop["epochs_run"] == len(trace)
            assert stop["stopped_early"] == (stop["epochs_run"] < ceiling), \
                "stopped_early disagrees with how many epochs actually ran"
            if stop["stopped_early"]:
                assert stop["epochs_run"] == best_epoch + enc.PATIENCE, (
                    "stopped after {} epochs with the best at {} and patience {}"
                    .format(stop["epochs_run"], best_epoch, enc.PATIENCE))
                assert not stop["ceiling_bound"]
            else:
                assert best_epoch <= ceiling
                assert stop["ceiling_bound"] == (best_epoch == ceiling)
            ok("early stopping at a reachable ceiling: {} ({} epochs, best {})".format(
                "fired" if stop["stopped_early"] else "did not fire, ceiling reached",
                stop["epochs_run"], best_epoch))
        except AssertionError as e:
            bad("early stopping: {}".format(e))
        except Exception as e:
            bad("early stopping: unexpected {}: {}".format(type(e).__name__, e))

        # --- (f) the seed replicate, on the real path -----------------------
        try:
            _, val_idx = make_cv_splits(df, N_SPLITS, e2_runner.PRIMARY_SEED)[
                e2_runner.ENCODER_FIXED_FOLD]
            for rep in res["seed_replicates"]:
                assert rep["fold"] == e2_runner.ENCODER_FIXED_FOLD
                assert rep["partition_seed"] == e2_runner.PRIMARY_SEED
                assert rep["selected_params"] in GRID
                assert rep["runtime_seconds"] > 0
                assert rep["resumed"] is False
                assert rep["fit_diagnosis"] is not None, \
                    "the replicate dropped its fit diagnosis again (#81b)"
            disp = res["dispersion"]
            fold0 = res["repeats"][0]["per_fold_macro_f1"][e2_runner.ENCODER_FIXED_FOLD]
            want = float(np.std([fold0] + [r["macro_f1"] for r in res["seed_replicates"]],
                                ddof=1))
            assert abs(disp["replicate_sd_macro_f1"] - want) < 1e-12, \
                "replicate_sd is not the spread of fold 0 across the three training seeds"
            assert len(val_idx) > 0
            ok("the replicates are fold 0 retrained, and replicate_sd is their spread")
        except AssertionError as e:
            bad("D3 replicate: {}".format(e))

        # --- cost fields, which the measured-unit-cost check reads -------------------------
        try:
            rep = res["repeats"][0]
            per_fold = [d["runtime_seconds"] for d in rep["folds_detail"]]
            assert rep["fold_runtime_seconds_sum"] is not None
            assert abs(rep["fold_runtime_seconds_sum"] - sum(per_fold)) < 1e-6
            assert res["resumed"] is False and rep["resumed_folds"] == []
            assert res["runtime_seconds"] >= rep["fold_runtime_seconds_sum"]
            ok("fold_runtime_seconds_sum is the sum of the folds, and the cell is not flagged resumed")
        except AssertionError as e:
            bad("cost fields: {}".format(e))

        # --- the checkpoint half, encoder-specific ------------------------
        try:
            assert not list((tmp / "ck").glob("{}_{}_{}_*".format(SUBTASK, EDITION, COMPONENT))), \
                "a finished encoder cell left its fold checkpoints behind"
            ok("a finished encoder cell deletes its own fold and replicate checkpoints")
        except AssertionError as e:
            bad("checkpoint cleanup: {}".format(e))

        # --- the selection diagnosis, recorded in the artefact ------------
        try:
            diag = res["selection_diagnosis"]
            assert diag is not None and diag["n_fits"] == N_SPLITS + 2, diag
            assert set(diag["axes"]) == {"lr"}, \
                "the grid varies lr only, so lr is the only axis to check: {}".format(
                    list(diag["axes"]))
            assert sum(diag["axes"]["lr"]["selected_counts"].values()) == N_SPLITS + 2
            assert diag == selection_diagnosis_for(res)
            ok("the artefact carries a selection diagnosis over folds and replicates")
        except AssertionError as e:
            bad("selection diagnosis: {}".format(e))

        # --- the stopping evidence, recorded for 3.3 ----------------------
        try:
            rep = res["stopping_evidence"]
            assert rep is not None, "no stopping evidence was written into the artefact"
            assert rep["n_fits"] == N_SPLITS * len(GRID), rep["n_fits"]
            assert rep["ceiling"] == MAX_EPOCHS and rep["patience"] == enc.PATIENCE
            # At this ceiling early stopping is unreachable (see above), so the
            # report must say the budget decided -- which is the OPPOSITE verdict
            # from the one a healthy campaign should produce, and getting it here
            # is what proves the two branches are distinguishable at all.
            assert rep["n_stopped_early"] == 0
            assert rep["rising_at_stop"] == 0 and rep["rising_at_stop_share"] is None
            assert sum(rep["best_epoch_counts"].values()) == rep["n_fits"]
            assert rep["mean_epochs_run"] == MAX_EPOCHS
            # `ceiling_bound` means STILL IMPROVING at the ceiling, not merely
            # reaching it -- a fit that peaked at epoch 1 and ran to 2 is not
            # bound. Derived from the trace rather than asserted, so whichever
            # branch of the verdict this slice produces is the one checked.
            assert rep["n_ceiling_bound"] == rep["best_epoch_counts"].get(
                str(MAX_EPOCHS), 0), (rep["n_ceiling_bound"], rep["best_epoch_counts"])
            if rep["n_ceiling_bound"]:
                assert "BUDGET, not the data" in rep["verdict"], rep["verdict"]
            else:
                assert "stopped early" in rep["verdict"], rep["verdict"]
            assert rep == e2_runner.stopping_report(res)
            ok("the artefact carries stopping evidence, and it reads the ceiling case correctly")
        except AssertionError as e:
            bad("stopping evidence: {}".format(e))

        # --- THE POST-CAMPAIGN PATH ---------------------------------------
        # e2_summary and the encoder notebook, against a real encoder artefact.
        try:
            from src import e2_summary
            runs = e2_summary.collect_runs(tmp / "out")
            assert len(runs) == 1 and runs[0]["meta"]["family"] == "encoder"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summary = e2_summary.summary_table(runs)
                per_class = e2_summary.per_class_table(runs)
                comparison = e2_summary.baseline_comparison(summary)
            row = summary.iloc[0]
            assert row["n_folds"] == N_SPLITS, row["n_folds"]
            assert row["seed_sd_basis"].startswith("fold 0 replicates"), row["seed_sd_basis"]
            assert row["machine"] and not row["resumed"]
            assert not row["runtime_is_wall_clock"], \
                "the encoder row fell back to wall clock; D5 reads this column"
            assert len(per_class) == len(classes)
            assert set(per_class["tier"]) <= {"interpret", "ci_gated", "report_only"}
            assert len(comparison) == 1 and comparison.iloc[0]["subtask"] == SUBTASK
            ok("e2_summary reads a real encoder artefact: summary, per-class, baseline comparison")
        except AssertionError as e:
            bad("e2_summary: {}".format(e))
        except Exception as e:
            bad("e2_summary: unexpected {}: {}".format(type(e).__name__, e))

        try:
            # exactly the frames notebooks/03_e2_encoder.ipynb builds by hand
            encoder_runs = [r for r in runs if r["meta"].get("family") == "encoder"]
            rows = []
            for r in encoder_runs:
                cap = r["meta"].get("max_epochs")
                for rep in r["repeats"]:
                    for fold, detail in enumerate(rep["folds_detail"]):
                        rows.append({"fold": fold, **detail["selected_params"],
                                     "epoch": detail["selected_epoch"], "max_epochs": cap,
                                     "at_cap": cap is not None and detail["selected_epoch"] == cap,
                                     "sel_f1": round(detail["selection_macro_f1"], 4),
                                     "temp": round(detail["temperature"]["temperature"], 3)})
            choices = pd.DataFrame(rows)
            assert len(choices) == N_SPLITS and choices["max_epochs"].nunique() == 1
            variance = [{"fold": 0,
                         "macro_f1": [round(100 * s, 2) for s in
                                      [r["repeats"][0]["per_fold_macro_f1"][0]]
                                      + [x["macro_f1"] for x in r["seed_replicates"]]],
                         "sd": round(100 * r["dispersion"]["replicate_sd_macro_f1"], 3)}
                        for r in encoder_runs]
            assert len(variance[0]["macro_f1"]) == 3
            fold_sums = [rep.get("fold_runtime_seconds_sum") for rep in encoder_runs[0]["repeats"]]
            rep_sum = sum(x["runtime_seconds"] for x in encoder_runs[0]["seed_replicates"])
            assert all(f is not None for f in fold_sums) and rep_sum > 0
            ok("the encoder notebook's own cells run against a real encoder artefact")
        except AssertionError as e:
            bad("notebook cells: {}".format(e))
        except Exception as e:
            bad("notebook cells: unexpected {}: {}".format(type(e).__name__, e))

        # --- and nothing landed in the project's own results/ -------------
        try:
            ck = Path(e2_runner.CKPT_ROOT)
            assert _ckpt_snapshot() == ckpt_before, \
                "the test changed the real checkpoint root: {}".format(ck)
            assert not (Path(e2_runner.RESULTS_DIR)
                        / "e2_{}_{}_{}.json".format(SUBTASK, EDITION, COMPONENT)).exists(), \
                "the test wrote an artefact into the project's results/e2/"
            assert not (ComponentStore().root / "{}_{}".format(SUBTASK, EDITION)
                        / "{}__seed42.npz".format(COMPONENT)).exists(), \
                "the test wrote into the project's component store"
            ok("no test artefact landed in the project's results/")
        except AssertionError as e:
            bad("isolation: {}".format(e))

    except Exception as e:
        bad("the cell did not run at all: {}: {}".format(type(e).__name__, e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 50)
    print("  {} passed, {} failed".format(_PASS, _FAIL))
    print("=" * 50)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
