"""test_resume_integration.py -- resume, but with the REAL components.

tests/test_fold_resume.py proves the mechanism against a stand-in model. That
leaves the question the thesis actually cares about: do the components that
produce Chapter 4's numbers resume identically? They carry an inner grid
search and per-fold temperature scaling, either of which could depend on
process state in a way a toy model would never reveal.

So this runs the real TF-IDF components on a real GermEval slice, kills the run
mid-cell, resumes it, and requires the same out-of-fold matrix and the same
pooled estimate. The second half does the same through `run_component`, i.e.
the whole e2_runner path including the component store and the checkpoint
cleanup, with every artefact redirected into a temp dir so the project's own
results are never touched.

The encoder is deliberately NOT here: it needs CUDA, and GPU training is not
bit-reproducible across processes anyway (which is why `resumed_folds` is
recorded in the artefact instead of pretended away).

Run: .venv/bin/python -m tests.test_resume_integration
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from src import e2_runner
from src.component_store import ComponentStore
from src.e2_runner import component_api, load_pool, run_component
from src.harness import resolve_classes, run_cv

_PASS, _FAIL = 0, 0
SLICE_N = 500
COMPONENTS = ["tml_svm", "tml_xgboost", "tml_lightgbm"]


def ok(m):
    global _PASS; _PASS += 1; print("  \033[32mPASS\033[0m  " + m)


def bad(m):
    global _FAIL; _FAIL += 1; print("  \033[31mFAIL\033[0m  " + m)


def small_pool(subtask="dbo", edition="2025", n=SLICE_N):
    return load_pool(subtask, edition).sample(n=n, random_state=0).reset_index(drop=True)


def cv(df, component, classes, ckpt=None, fp="itest", crash_after=None, n_splits=3):
    api = component_api(component, n_jobs=2)
    inner = api["make_train_fn"](classes, 42)
    state = {"n": 0}

    def train_fn(train_df):
        if crash_after is not None and state["n"] >= crash_after:
            raise RuntimeError("simulated crash")
        state["n"] += 1
        return inner(train_df)

    res = run_cv(df, train_fn, predict_proba_fn=api["predict_proba_fn"],
                 classes=classes, n_splits=n_splits, seed=42,
                 on_fold=lambda i, m, v, p: dict(m.record, fold=i),
                 checkpoint_dir=ckpt, checkpoint_fingerprint=fp if ckpt else None,
                 bootstrap=True, bootstrap_n=200, verbose=False)
    return res, state


# ---------------------------------------------------------------------------

def check_real_components_resume_identically(tmp: Path):
    df = small_pool()
    classes = resolve_classes(df["label"].values)
    for comp in COMPONENTS:
        ref, ref_state = cv(df, comp, classes)
        assert ref_state["n"] == 3, "{}: reference trained {} folds".format(comp, ref_state["n"])

        ckpt = tmp / ("ck_" + comp)
        try:
            cv(df, comp, classes, ckpt=ckpt, crash_after=1)
        except RuntimeError as e:
            assert "simulated crash" in str(e)
        else:
            raise AssertionError("{}: the simulated crash did not happen".format(comp))
        assert (ckpt / "fold00.npz").exists(), "{}: fold 0 was not checkpointed".format(comp)

        got, got_state = cv(df, comp, classes, ckpt=ckpt)
        assert got["resumed_folds"] == [0], "{}: resumed {}".format(comp, got["resumed_folds"])
        assert got_state["n"] == 2, "{}: retrained {} folds, expected 2".format(comp, got_state["n"])

        np.testing.assert_array_equal(ref["oof_proba"], got["oof_proba"])
        assert ref["pooled"]["macro_f1"] == got["pooled"]["macro_f1"], comp
        assert json.dumps(ref["pooled"], sort_keys=True, default=str) == \
               json.dumps(got["pooled"], sort_keys=True, default=str), comp
        # the per-fold temperature and selected params must survive the restore
        assert ref["fold_payloads"] == got["fold_payloads"], comp
    ok("all three TML components resume to identical numbers")


def check_full_run_component_path(tmp: Path):
    """The e2_runner path end to end: results JSON, component store, cleanup."""
    df = small_pool()
    original = e2_runner.load_pool
    e2_runner.load_pool = lambda st, ed: df
    try:
        def run_cell(out, ckpt, store_root, crash_at=None):
            store = ComponentStore(root=store_root)
            if crash_at is not None:
                api = e2_runner.component_api
                calls = {"n": 0}

                def wrapped(component, **kw):
                    a = dict(api(component, **kw))
                    inner = a["make_train_fn"]

                    def mk(classes, seed):
                        f = inner(classes, seed)

                        def t(train_df):
                            if calls["n"] >= crash_at:
                                raise RuntimeError("simulated crash")
                            calls["n"] += 1
                            return f(train_df)
                        return t
                    a["make_train_fn"] = mk
                    return a
                e2_runner.component_api = wrapped
                try:
                    return run_component("dbo", "2025", "tml_svm", seeds=(42,),
                                         n_splits=3, bootstrap_n=100, store=store,
                                         out_dir=out, ckpt_root=ckpt)
                finally:
                    e2_runner.component_api = api
            return run_component("dbo", "2025", "tml_svm", seeds=(42,),
                                 n_splits=3, bootstrap_n=100, store=store,
                                 out_dir=out, ckpt_root=ckpt)

        # reference: uninterrupted, checkpointing off entirely
        ref = run_cell(tmp / "out_ref", None, tmp / "store_ref")

        # interrupted after two folds, then resumed
        out2, ck2, st2 = tmp / "out2", tmp / "ck2", tmp / "store2"
        try:
            run_cell(out2, ck2, st2, crash_at=2)
        except RuntimeError as e:
            assert "simulated crash" in str(e)
        else:
            raise AssertionError("the simulated crash did not happen")
        assert sorted(p.name for p in (ck2 / "dbo_2025_tml_svm_seed42").glob("*.npz")) \
            == ["fold00.npz", "fold01.npz"]

        got = run_cell(out2, ck2, st2)
        assert got["repeats"][0]["resumed_folds"] == [0, 1], got["repeats"][0]["resumed_folds"]

        # the numbers the thesis would print
        for field in ("pooled_macro_f1", "pooled_weighted_f1", "fold_sd_macro_f1",
                      "per_fold_macro_f1", "per_class_f1", "ece", "brier",
                      "selected_params_per_fold", "temperature_per_fold"):
            assert ref["repeats"][0][field] == got["repeats"][0][field], \
                "{} differs after resume".format(field)
        assert ref["dispersion"]["mean_pooled_macro_f1"] == got["dispersion"]["mean_pooled_macro_f1"]

        # the component store's out-of-fold probabilities must match too --
        # they are what E3 consumes without retraining
        a = np.load(next((tmp / "store_ref").rglob("*.npz")), allow_pickle=False)
        b = np.load(next((tmp / "store2").rglob("*.npz")), allow_pickle=False)
        np.testing.assert_array_equal(a["proba"], b["proba"])

        # and a finished cell leaves no checkpoints behind
        assert not list(ck2.glob("dbo_2025_tml_svm_*")), \
            "checkpoints survived a completed cell: {}".format(list(ck2.iterdir()))
        ok("run_component: same results JSON, same store, checkpoints cleaned up")
    finally:
        e2_runner.load_pool = original


def check_resumed_cell_reports_the_full_compute_time(tmp: Path):
    """The trap this guards. `runtime_seconds` is wall clock, so on a resumed
    cell it covers only the retrained folds -- and notebooks/03_e2_encoder.ipynb
    divides exactly that field by 3600 to report the encoder arm's run hours.
    The D5 measured-unit-cost gate is the entire reason the first pod session
    exists, so a resumed cell silently reporting a third of its cost would
    corrupt the one number that session is for."""
    df = small_pool(n=300)
    original_pool, original_api = e2_runner.load_pool, e2_runner.component_api
    e2_runner.load_pool = lambda st, ed: df

    # give the component a per-fold runtime, as the encoder's record has
    def timed_api(component, **kw):
        a = dict(original_api(component, **kw))
        inner = a["make_train_fn"]

        def mk(classes, seed):
            f = inner(classes, seed)

            def t(train_df):
                m = f(train_df)
                m.record = dict(m.record, runtime_seconds=10.0)
                return m
            return t
        a["make_train_fn"] = mk
        return a
    e2_runner.component_api = timed_api

    try:
        def cell(out, ckpt, store, crash_at=None):
            api_now = e2_runner.component_api
            if crash_at is not None:
                calls = {"n": 0}

                def crashing(component, **kw):
                    a = dict(api_now(component, **kw))
                    inner = a["make_train_fn"]

                    def mk(classes, seed):
                        f = inner(classes, seed)

                        def t(train_df):
                            if calls["n"] >= crash_at:
                                raise RuntimeError("simulated crash")
                            calls["n"] += 1
                            return f(train_df)
                        return t
                    a["make_train_fn"] = mk
                    return a
                e2_runner.component_api = crashing
            try:
                return run_component("dbo", "2025", "tml_svm", seeds=(42,), n_splits=3,
                                     bootstrap_n=50, store=ComponentStore(root=store),
                                     out_dir=out, ckpt_root=ckpt)
            finally:
                e2_runner.component_api = api_now

        ref = cell(tmp / "o1", None, tmp / "s1")
        assert ref["repeats"][0]["fold_runtime_seconds_sum"] == 30.0,             ref["repeats"][0]["fold_runtime_seconds_sum"]
        assert ref["resumed"] is False

        o2, c2, s2 = tmp / "o2", tmp / "c2", tmp / "s2"
        try:
            cell(o2, c2, s2, crash_at=2)
        except RuntimeError:
            pass
        got = cell(o2, c2, s2)

        assert got["resumed"] is True, "the cell must declare that it was resumed"
        assert got["repeats"][0]["resumed_folds"] == [0, 1]
        # the honest field survives the resume ...
        assert got["repeats"][0]["fold_runtime_seconds_sum"] == 30.0,             "resumed cell reports {} s of compute, not 30".format(
                got["repeats"][0]["fold_runtime_seconds_sum"])
        # ... while the wall clock genuinely is shorter, which is the trap
        assert got["repeats"][0]["runtime_seconds"] < ref["repeats"][0]["runtime_seconds"]
        ok("a resumed cell reports its full compute time, and flags itself")
    finally:
        e2_runner.load_pool, e2_runner.component_api = original_pool, original_api


def _ckpt_snapshot() -> frozenset:
    root = Path(e2_runner.CKPT_ROOT)
    if not root.exists():
        return frozenset()
    return frozenset((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                     for p in root.rglob("*") if p.is_file())


# Taken in main() before the first check runs.
_CKPT_BEFORE: frozenset | None = None


def check_project_artefacts_untouched(tmp: Path):
    """The tests must not write into results/ -- a polluted component store
    would be discovered much later and be very hard to explain.

    UNCHANGED, not EMPTY: during a GPU campaign, fold checkpoints are copied
    into this very root, so "empty" would fail on the campaign's own copies.
    What the check protects is that a TEST writes nothing there. Do not run the
    suite while checkpoints are being copied in."""
    ck = Path(e2_runner.CKPT_ROOT)
    assert _CKPT_BEFORE is not None, "main() took no checkpoint snapshot"
    assert _ckpt_snapshot() == _CKPT_BEFORE, \
        "a test changed the real checkpoint root: {}".format(ck)
    ok("no test artefact landed in the project's results/")


def main() -> int:
    global _CKPT_BEFORE
    _CKPT_BEFORE = _ckpt_snapshot()
    checks = [(n, f) for n, f in sorted(globals().items())
              if n.startswith("check_") and callable(f)]
    print("\n=== resume with the real components ({} checks) ===".format(len(checks)))
    for name, fn in checks:
        d = Path(tempfile.mkdtemp())
        try:
            fn(d)
        except AssertionError as e:
            bad("{}: {}".format(name[6:].replace("_", " "), e))
        except Exception as e:
            bad("{}: unexpected {}: {}".format(name[6:].replace("_", " "), type(e).__name__, e))
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print("\n" + "=" * 50)
    print("  {} passed, {} failed".format(_PASS, _FAIL))
    print("=" * 50)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
