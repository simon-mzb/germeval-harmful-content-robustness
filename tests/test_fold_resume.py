"""test_fold_resume.py -- does fold-level resume produce the same numbers?

The whole value of checkpointing is that an interrupted campaign can be
restarted without redoing finished folds. The whole RISK of it is that the
restart quietly produces something slightly different, which would be
undetectable in the thesis because nobody re-runs a 30-hour arm to compare.

So the load-bearing assertion here is equality: a run interrupted after fold 2
and resumed must yield the same out-of-fold matrix, the same per-fold metrics,
the same pooled estimate and the same bootstrap interval as a run that was
never interrupted. Everything else in this file guards the ways a checkpoint
could be reused when it must not be.

Written as a standalone script rather than a pytest suite: pytest is not a
project dependency, and adding one would rewrite `uv.lock` -- which the pod
consumes with `uv sync --frozen` -- days before the first pod session.
The rest of the project verifies itself the same way (verify_data_numbers.py,
e1_baseline_set_check.py).

Run: .venv/bin/python -m tests.test_fold_resume
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.harness import (load_fold_checkpoint, run_cv, save_fold_checkpoint)

# --- a minimal harness, in the style of the bash suite ---------------------
_PASS, _FAIL = 0, 0


def ok(msg):
    global _PASS; _PASS += 1; print("  \033[32mPASS\033[0m  " + msg)


def bad(msg):
    global _FAIL; _FAIL += 1; print("  \033[31mFAIL\033[0m  " + msg)


class raises:
    """with raises(RuntimeError, "different run"): ..."""

    def __init__(self, exc, match=""):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError(
                "expected {} matching {!r}, nothing raised".format(
                    self.exc.__name__, self.match))
        if not issubclass(t, self.exc):
            return False
        if self.match and self.match not in str(v):
            raise AssertionError(
                "expected {!r} in {!r}".format(self.match, str(v)))
        return True

CLASSES = ["a", "b", "c"]
FP = "test-fingerprint-v1"


def make_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array([CLASSES[i % len(CLASSES)] for i in range(n)])
    rng.shuffle(labels)
    return pd.DataFrame({
        "id": [f"id{i}" for i in range(n)],
        "description": [f"text nummer {i} mit wort{i % 7}" for i in range(n)],
        "label": labels,
    })


class Counter:
    """A deterministic 'component' that records how often it was trained.

    The probabilities depend only on the training fold's content, so an
    identical fold must give identical output -- which is what lets the
    equality assertion below mean something.
    """

    def __init__(self):
        self.trainings = 0

    def train_fn(self, train_df):
        self.trainings += 1
        counts = train_df["label"].value_counts()
        w = np.array([counts.get(c, 0) for c in CLASSES], dtype=np.float64)
        model = type("M", (), {})()
        model.w = w / w.sum()
        model.record = {"selected_params": {"n_train": int(len(train_df))},
                        "temperature": {"temperature": 1.0}}
        return model

    @staticmethod
    def predict_proba_fn(model, val_df, classes):
        # deterministic per row, and genuinely row-dependent
        h = np.array([abs(hash(t)) % 1000 for t in val_df["description"]], dtype=np.float64)
        base = np.outer(np.ones(len(val_df)), model.w)
        jitter = np.stack([(h + k * 37) % 11 for k in range(len(classes))], axis=1)
        p = base + jitter / 100.0
        return p / p.sum(axis=1, keepdims=True)


def run(df, ckpt=None, fingerprint=FP, seed=42, n_splits=5, comp=None, **kw):
    comp = comp or Counter()
    res = run_cv(
        df, comp.train_fn,
        predict_proba_fn=comp.predict_proba_fn,
        classes=CLASSES, n_splits=n_splits, seed=seed,
        on_fold=lambda i, m, v, p: dict(m.record, fold=i),
        checkpoint_dir=ckpt, checkpoint_fingerprint=fingerprint if ckpt else None,
        bootstrap=True, bootstrap_n=200, verbose=False, **kw)
    return res, comp


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


def check_a_checkpoint_survives_a_pod_change(tmp):
    """A fetched checkpoint must restore onto a DIFFERENT pod.

    This is what makes an unattended campaign survivable without a network
    volume: checkpoints are copied off the rented machine while it runs, and if
    it is lost -- terminated, or a restart that finds no free GPU -- a
    replacement machine resumes instead of restarting from zero.

    It works because the fingerprint is
    `component|subtask|edition|config_id|data_rules_id|matrix_version`, with
    nothing machine-specific in it. That is a property worth pinning: adding a
    machine or an env field to the fingerprint would silently make every
    checkpoint non-portable, and the campaign would quietly retrain instead.
    """
    import shutil

    pod_a = Path(tmp) / "pod_a" / "ck"
    fp = "encoder_1b|c2a|2025|CFG|RULES|1.5"
    kw = dict(fingerprint=fp, seed=42, n_splits=5,
              classes=["False", "True"], val_idx=np.array([3, 1, 4, 1, 5]))
    proba = np.random.RandomState(0).rand(5, 2).astype(np.float32)
    save_fold_checkpoint(pod_a, 0, fold_proba=proba, payload={"selected_epoch": 4}, **kw)

    pod_b = Path(tmp) / "pod_b" / "ck"
    shutil.copytree(pod_a, pod_b)

    got = load_fold_checkpoint(pod_b, 0, **kw)
    if got is None:
        bad("a checkpoint does NOT restore onto a different pod")
        return
    ok("a checkpoint restores onto a different pod")
    restored, payload = got
    (ok if np.array_equal(restored.astype(np.float32), proba)
     else bad)("  ...bit-identically")
    (ok if payload.get("selected_epoch") == 4 else bad)("  ...carrying its payload")

    # And it must still refuse a genuinely different run, from the new pod too.
    other = dict(kw, fingerprint="encoder_1b|c2a|2025|OTHER|RULES|1.5")
    try:
        load_fold_checkpoint(pod_b, 0, **other)
        bad("  ...but a different config_id was ACCEPTED on the new pod")
    except RuntimeError:
        ok("  ...while a different config_id is still refused")


def check_resumed_run_equals_uninterrupted_run(tmp_path: Path):
    df = make_df()

    # (1) the reference: never interrupted, no checkpoints involved at all
    ref, ref_comp = run(df)
    assert ref_comp.trainings == 5
    assert ref["resumed_folds"] == []

    # (2) a run that dies during fold 2, after folds 0 and 1 are on disk
    ckpt = tmp_path / "ck"
    boom = Counter()
    original = boom.train_fn

    def dying(train_df):
        if boom.trainings == 2:
            raise RuntimeError("simulated crash in fold 2")
        return original(train_df)

    boom.train_fn = dying
    with raises(RuntimeError, match="simulated crash"):
        run(df, ckpt=ckpt, comp=boom)
    assert boom.trainings == 2, "two folds should have completed before the crash"
    assert sorted(p.name for p in ckpt.glob("*.npz")) == ["fold00.npz", "fold01.npz"]

    # (3) the resume
    resumed, res_comp = run(df, ckpt=ckpt)
    assert resumed["resumed_folds"] == [0, 1]
    assert res_comp.trainings == 3, "only the three unfinished folds may retrain"

    # (4) equality, the point of the whole exercise
    np.testing.assert_array_equal(ref["oof_proba"], resumed["oof_proba"])
    np.testing.assert_array_equal(ref["oof_pred"], resumed["oof_pred"])
    assert ref["pooled"]["macro_f1"] == resumed["pooled"]["macro_f1"]
    assert ref["pooled"]["calibration"] == resumed["pooled"]["calibration"]
    assert ref["mean_macro_f1"] == resumed["mean_macro_f1"]
    assert ref["std_macro_f1"] == resumed["std_macro_f1"]
    assert [f["macro_f1"] for f in ref["folds"]] == [f["macro_f1"] for f in resumed["folds"]]
    assert ref["mean_per_class_f1"] == resumed["mean_per_class_f1"]
    # the bootstrap interval too -- it is seeded, so it must not move either
    assert json.dumps(ref["pooled"], sort_keys=True, default=str) == \
           json.dumps(resumed["pooled"], sort_keys=True, default=str)
    # and the on_fold records survive the round trip
    assert ref["fold_payloads"] == resumed["fold_payloads"]


def check_resume_of_a_complete_run_trains_nothing(tmp_path: Path):
    df = make_df()
    ckpt = tmp_path / "ck"
    first, c1 = run(df, ckpt=ckpt)
    assert c1.trainings == 5
    second, c2 = run(df, ckpt=ckpt)
    assert c2.trainings == 0
    assert second["resumed_folds"] == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(first["oof_proba"], second["oof_proba"])


# ---------------------------------------------------------------------------
# Refusals: every way a checkpoint could belong to a different computation
# ---------------------------------------------------------------------------

def check_different_fingerprint_is_refused(tmp_path: Path):
    df = make_df()
    ckpt = tmp_path / "ck"
    run(df, ckpt=ckpt)
    with raises(RuntimeError, match="different run"):
        run(df, ckpt=ckpt, fingerprint="something-else")


def check_different_seed_is_refused(tmp_path: Path):
    df = make_df()
    ckpt = tmp_path / "ck"
    run(df, ckpt=ckpt, seed=42)
    with raises(RuntimeError, match="different run"):
        run(df, ckpt=ckpt, seed=43)


def check_different_split_count_is_refused(tmp_path: Path):
    df = make_df()
    ckpt = tmp_path / "ck"
    run(df, ckpt=ckpt, n_splits=5)
    with raises(RuntimeError, match="different run"):
        run(df, ckpt=ckpt, n_splits=4)


def check_different_class_ordering_is_refused(tmp_path: Path):
    df = make_df()
    ckpt = tmp_path / "ck"
    run(df, ckpt=ckpt)
    comp = Counter()
    with raises(RuntimeError, match="different run"):
        run_cv(df, comp.train_fn, predict_proba_fn=comp.predict_proba_fn,
               classes=["c", "b", "a"], n_splits=5, seed=42,
               checkpoint_dir=ckpt, checkpoint_fingerprint=FP, verbose=False)


def check_different_validation_indices_are_refused(tmp_path: Path):
    """Same fingerprint, same seed, different partition -- only the stored
    val_idx can catch this, which is why it is stored."""
    df = make_df()
    ckpt = tmp_path / "ck"
    run(df, ckpt=ckpt)
    # a different dataset of the same shape reshuffles the stratified folds
    with raises(RuntimeError, match="validation indices"):
        run(make_df(seed=7), ckpt=ckpt)


def check_replicate_cannot_reuse_a_main_fold_checkpoint(tmp_path: Path):
    """The real trap: a seed replicate runs the SAME fold index over the SAME
    validation indices as the main run. Nothing but the fingerprint separates
    them, so the model seed has to be in it."""
    df = make_df()
    ckpt = tmp_path / "ck"
    run(df, ckpt=ckpt, fingerprint="cell|xyz")
    with raises(RuntimeError, match="different run"):
        run(df, ckpt=ckpt, fingerprint="cell|xyz|replicate43")


# ---------------------------------------------------------------------------
# Misuse and durability
# ---------------------------------------------------------------------------

def check_checkpoint_dir_without_fingerprint_is_rejected(tmp_path: Path):
    df = make_df()
    comp = Counter()
    with raises(ValueError, match="requires checkpoint_fingerprint"):
        run_cv(df, comp.train_fn, predict_proba_fn=comp.predict_proba_fn,
               classes=CLASSES, checkpoint_dir=tmp_path / "ck", verbose=False)


def check_hard_label_mode_refuses_checkpointing(tmp_path: Path):
    df = make_df()
    comp = Counter()
    with raises(NotImplementedError, match="probability mode only"):
        run_cv(df, comp.train_fn,
               predict_fn=lambda m, v: np.array(["a"] * len(v)),
               n_splits=5, checkpoint_dir=tmp_path / "ck",
               checkpoint_fingerprint=FP, verbose=False)


def check_no_checkpoint_dir_leaves_behaviour_untouched(tmp_path: Path):
    """Checkpointing off must be the old code path, including the new keys
    being present but empty rather than the result changing shape."""
    df = make_df()
    res, _ = run(df)
    assert res["resumed_folds"] == []
    assert len(res["fold_payloads"]) == 5
    assert list(tmp_path.iterdir()) == []


def check_writes_are_atomic(tmp_path: Path):
    """A crash mid-write must leave the old file or none -- never a truncated
    one that would load as a valid but wrong fold."""
    ckpt = tmp_path / "ck"
    proba = np.full((10, 3), 1 / 3)
    save_fold_checkpoint(ckpt, 0, fingerprint=FP, seed=42, n_splits=5,
                         classes=CLASSES, val_idx=np.arange(10),
                         fold_proba=proba, payload={"x": 1})
    assert list(ckpt.glob("*.tmp")) == [], "no temporary file may survive"
    got = load_fold_checkpoint(ckpt, 0, fingerprint=FP, seed=42, n_splits=5,
                               classes=CLASSES, val_idx=np.arange(10))
    assert got is not None
    np.testing.assert_array_equal(got[0], proba)
    assert got[1] == {"x": 1}


def check_numpy_records_survive_checkpointing(tmp_path: Path):
    """The encoder's model.record carries numpy scalars (torch produces them).
    A bare json.dumps raises on those, so the FIRST fine-tune of the first
    campaign would have died at the first checkpoint write. No TML component
    would ever have shown it -- their records are plain Python."""
    df = make_df()
    ckpt = tmp_path / "ck"

    class NumpyRecord(Counter):
        def train_fn(self, train_df):
            m = super().train_fn(train_df)
            m.record = {
                "selected_params": {"lr": np.float32(2e-5), "r": np.int64(16)},
                "selected_epoch": np.int32(3),
                "temperature": {"temperature": np.float64(1.07)},
                "flag": np.bool_(True),
                "curve": np.array([0.1, 0.2], dtype=np.float32),
            }
            return m

    ref, _ = run(df, comp=NumpyRecord())
    first, _ = run(df, ckpt=ckpt, comp=NumpyRecord())
    second, c2 = run(df, ckpt=ckpt, comp=NumpyRecord())
    assert c2.trainings == 0, "the second run should have restored every fold"

    # restored == freshly computed, as objects, not merely as printed numbers
    assert first["fold_payloads"] == second["fold_payloads"]
    assert ref["fold_payloads"] == second["fold_payloads"]
    # and the types are plain Python on both paths, so the artefact is stable
    p0 = second["fold_payloads"][0]
    assert isinstance(p0["selected_params"]["lr"], float), type(p0["selected_params"]["lr"])
    assert isinstance(p0["selected_epoch"], int)
    assert isinstance(p0["curve"], list)


def check_missing_checkpoint_returns_none_rather_than_raising(tmp_path: Path):
    assert load_fold_checkpoint(tmp_path / "nope", 3, fingerprint=FP, seed=42,
                                n_splits=5, classes=CLASSES,
                                val_idx=np.arange(5)) is None


def check_shape_mismatch_is_caught(tmp_path: Path):
    ckpt = tmp_path / "ck"
    save_fold_checkpoint(ckpt, 0, fingerprint=FP, seed=42, n_splits=5,
                         classes=CLASSES, val_idx=np.arange(10),
                         fold_proba=np.full((10, 3), 1 / 3), payload=None)
    with raises(RuntimeError, match="validation indices"):
        load_fold_checkpoint(ckpt, 0, fingerprint=FP, seed=42, n_splits=5,
                             classes=CLASSES, val_idx=np.arange(9))


# ---------------------------------------------------------------------------

def main() -> int:
    checks = [(n, f) for n, f in sorted(globals().items())
              if n.startswith("check_") and callable(f)]
    print("\n=== fold-level checkpoint / resume ({} checks) ===".format(len(checks)))
    for name, fn in checks:
        tmp = Path(tempfile.mkdtemp())
        try:
            fn(tmp)
            ok(name[len("check_"):].replace("_", " "))
        except AssertionError as e:
            bad("{}: {}".format(name[len("check_"):].replace("_", " "), e))
        except Exception as e:
            bad("{}: unexpected {}: {}".format(
                name[len("check_"):].replace("_", " "), type(e).__name__, e))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 50)
    print("  {} passed, {} failed".format(_PASS, _FAIL))
    print("=" * 50)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
