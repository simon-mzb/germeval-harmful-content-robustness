"""
harness.py -- shared evaluation harness for all experiments (E0-E5).

Implements the evaluation protocol of the thesis (Section 3.3) and the
data-quality rules applied to every training pool (`apply_data_quality_rules`).

All experiments — baselines, components, combinations — run through this
module with identical folds, metrics, and seed handling so that differences
in measured performance reflect the systems, not the evaluation code.

Usage
-----
from src.harness import (
    apply_data_quality_rules,
    make_cv_splits,
    evaluate_predictions,
    run_cv,
    check_g0_counts,
    env_pin,
    pooled_oof_report,
    one_hot_proba,
    hard_label_proba_fn,
    resolve_classes,
    tier_for_n,
)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.calibration import calibration_report
from src.data_loading import load_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Organiser-reported train/test counts (felser2025, Table specs). The loader
# must reproduce these exactly.
G0_TARGETS = {
    "c2a": {"train": 6840, "test": 2982},
    "dbo": {"train": 7454, "test": 3194},
    "vio": {"train": 7783, "test": 3335},
}

# Default CV settings (§3.3).
DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42
N_SEEDS = 3  # number of training seeds for variance estimation
BOOTSTRAP_N = 1000  # bootstrap samples for confidence intervals


# ---------------------------------------------------------------------------
# Data-quality rules
# ---------------------------------------------------------------------------

def _drop_contradicting_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove every occurrence of a text that appears under two different labels.

    Measured on the released data (src/data_profile.py, section
    `label_contradictions`): DBO 2025 train holds 87 such texts, each occurring
    exactly twice under two ids, 86 of them under the pair agitation/nothing,
    and 86 of them in the injected cohort. Because the class is small the
    proportional effect is not: 27.5% of all agitation items are in a
    contradicting pair.

    Deduplicating by first occurrence would resolve the contradiction by file
    order rather than by evidence, and measured against the organisers' own
    2026 revision that rule keeps the label they later changed in all 55
    resolvable cases. Resolving from the 2026 labels is not available either,
    because it would import later-edition information into a training set whose
    relation to that edition is what Dimension 3 measures. Dropping both copies
    is the only resolution independent of the later edition, and a text
    annotated two incompatible ways carries no usable supervision anyway.

    Runs before deduplication: afterwards only one copy would remain and the
    contradiction would be invisible.
    """
    if "label" not in df.columns:
        return df
    norm = df["description"].str.strip().str.lower()
    n_labels = df.groupby(norm)["label"].transform("nunique")
    mask = n_labels == 1
    dropped = int((~mask).sum())
    if dropped > 0:
        texts = int(norm[~mask].nunique())
        print(f"  [contradiction-drop] dropped {dropped} rows over {texts} texts "
              f"with conflicting labels")
    return df[mask].reset_index(drop=True)


def _drop_within_edition_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove near-duplicate texts within a split (normalised: strip + lowercase).

    EDA found: C2A-2025-train 66 dupes (0.97%), DBO 156 (2.1%), VIO 63 (0.81%).
    Keeps the first occurrence; drops subsequent duplicates.
    """
    norm = df["description"].str.strip().str.lower()
    mask = ~norm.duplicated(keep="first")
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"  [dedup] dropped {dropped} within-edition duplicates")
    return df[mask].reset_index(drop=True)


def _drop_train_test_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drop from train_df any text that also appears in test_df (normalised).

    EDA found leakage in 2025: DBO 124 texts (3.9% of test), C2A 28 (0.94%),
    VIO 38 (1.15%). 2026 is clean (<0.4%). Applied only when a test split is
    passed to `apply_data_quality_rules`; the E2 pool passes none, and
    `e2_runner`'s module docstring says why the rule stays off there.
    """
    test_norm = set(test_df["description"].str.strip().str.lower())
    train_norm = train_df["description"].str.strip().str.lower()
    mask = ~train_norm.isin(test_norm)
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"  [overlap-drop] dropped {dropped} train texts present in test")
    return train_df[mask].reset_index(drop=True)


def _drop_cross_edition_overlap(
    eval_df: pd.DataFrame,
    other_edition_train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drop from eval_df any text present in another edition's train split (normalised).

    This is the Dimension-3 (E4c) guard and it protects the *evaluation* set, which
    is the opposite direction from _drop_train_test_overlap. GermEval 2026 is the
    same 2014-2016 source corpus as 2025 with added samples and a revised annotation
    regime, so the editions overlap heavily: measured 2026-train against 2025-train,
    C2A 6751/15819 (42.7%), DBO 7077/15847 (44.7%), VIO 5944/16420 (36.2%).

    Without this, a 2025-trained model evaluated on 2026 data is scored on its own
    training items for 36-45% of the set, and the resulting macro-F1 is inflated for
    the worst possible reason. The residual class supports it leaves fall into
    the reporting tiers, because removal costs the rare classes dearly (DBO
    subversive 53 -> 12, VIO glorification 27 -> 12).
    """
    other_norm = set(other_edition_train_df["description"].str.strip().str.lower())
    eval_norm = eval_df["description"].str.strip().str.lower()
    mask = ~eval_norm.isin(other_norm)
    dropped = (~mask).sum()
    if dropped > 0:
        print(
            f"  [cross-edition-drop] dropped {dropped} of {len(eval_df)} eval texts "
            f"({dropped / len(eval_df) * 100:.1f}%) present in the other edition's train split"
        )
    return eval_df[mask].reset_index(drop=True)


def apply_data_quality_rules(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    *,
    dedup: bool = True,
    drop_contradictions: bool = True,
    drop_overlap: bool = True,
    cross_edition_train_df: pd.DataFrame | None = None,
    drop_cross_edition: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Apply the data-quality rules to a train split.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split (must have 'description' and 'label' columns).
    test_df : pd.DataFrame | None
        Test split used only for overlap detection; not modified.
    dedup : bool
        Apply within-edition deduplication (default True).
    drop_contradictions : bool
        Drop texts carrying two different labels (default True). Must run
        before `dedup`, which would otherwise hide them. See
        `_drop_contradicting_labels` for the decision and its rationale.
    drop_overlap : bool
        Drop train texts that also appear in test (default True).
    cross_edition_train_df : pd.DataFrame | None
        The *other* edition's train split, used only for overlap detection.
    drop_cross_edition : bool
        Drop rows that also appear in cross_edition_train_df. **Default False on
        purpose**: this rule applies to E4c evaluation pools only, and switching it
        on by default would silently change every E1/E2/E3 number, all of which are
        single-edition work. E4c must set it explicitly.

    Returns
    -------
    (cleaned_train_df, test_df)  — test_df is returned unchanged.
    """
    if drop_contradictions:
        train_df = _drop_contradicting_labels(train_df)
    if dedup:
        train_df = _drop_within_edition_duplicates(train_df)
    if drop_overlap and test_df is not None:
        train_df = _drop_train_test_overlap(train_df, test_df)
    if drop_cross_edition and cross_edition_train_df is not None:
        train_df = _drop_cross_edition_overlap(train_df, cross_edition_train_df)
    return train_df, test_df


def make_vio_fresh_holdout(
    train_df: pd.DataFrame,
    holdout_frac: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a fresh stratified holdout from VIO-2026-train for cross-edition eval.

    Required because 76.98% of VIO-2025-test is inside VIO-2026-train
    (EDA finding) — using 2025-test for cross-edition VIO is contaminated.
    C2A (0.27%) and DBO (0.35%) are clean; only VIO needs this.

    Not used by any experiment: Dimension 3 (E4c) evaluates on the frozen,
    overlap-free 2026 pool of `e4c_pool` instead.

    Parameters
    ----------
    train_df : pd.DataFrame
        VIO-2026-train (after data-quality rules).
    holdout_frac : float
        Fraction of 2026-train to reserve as fresh holdout (default 0.2).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (reduced_train_df, holdout_df)
    """
    from sklearn.model_selection import train_test_split

    train_part, holdout = train_test_split(
        train_df,
        test_size=holdout_frac,
        stratify=train_df["label"],
        random_state=seed,
    )
    print(
        f"  [vio-holdout] reserved {len(holdout)} rows as fresh holdout "
        f"({holdout_frac:.0%} of 2026-train); training on {len(train_part)}"
    )
    return train_part.reset_index(drop=True), holdout.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CV splits
# ---------------------------------------------------------------------------

def make_cv_splits(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return stratified k-fold split indices for df.

    Uses 'label' column for stratification. Returns a list of
    (train_indices, val_indices) arrays — same interface as sklearn.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df, df["label"]))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = BOOTSTRAP_N,
    ci: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for a scalar metric.

    Returns (lower, upper) bounds at the given CI level.
    """
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    alpha = (1 - ci) / 2
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))


def evaluate_predictions(
    y_true,
    y_pred,
    *,
    labels: list | None = None,
    bootstrap: bool = True,
    bootstrap_n: int = BOOTSTRAP_N,
    ci: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """
    Compute the full metric suite for a single fold or held-out evaluation.

    Primary metric: macro-averaged F1 (§3.3, in line with GermEval shared task).
    Also reports: per-class F1, weighted F1, classification report, confusion matrix.
    Optionally adds bootstrap CI on macro-F1.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted labels.
    labels : list | None
        Class labels in desired order (inferred if None).
    bootstrap : bool
        Compute bootstrap CI on macro-F1 (default True).
    bootstrap_n : int
        Number of bootstrap samples.
    ci : float
        Confidence interval level (default 0.95).
    seed : int
        Bootstrap RNG seed.

    Returns
    -------
    dict with keys:
        macro_f1, weighted_f1, per_class_f1, report (str),
        confusion_matrix, [macro_f1_ci_lower, macro_f1_ci_upper]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=str)

    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = {str(l): float(s) for l, s in zip(labels, per_class)}

    # Support per class, and the classes this evaluation set does not contain at
    # all. Recorded because `labels` is the *global* class ordering: a class
    # with zero support still enters per_class_f1 as 0.0 under zero_division=0,
    # and macro_f1 averages it in. That is not "predicted badly", it is
    # undefined, and it silently deflates the fold score and inflates fold_sd.
    # The value is reported rather than repaired here on purpose -- redefining
    # macro_f1 would move the pinned E1 baselines and is a design decision,
    # not a bug fix.
    per_class_support = {str(l): int((y_true == l).sum()) for l in labels}
    zero_support = [k for k, v in per_class_support.items() if v == 0]

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    result: dict[str, Any] = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
        "zero_support_classes": zero_support,
        "report": report,
        "confusion_matrix": cm,
        "labels": [str(l) for l in labels],
        "n_samples": int(len(y_true)),
    }

    if bootstrap:
        def _macro(yt, yp):
            return f1_score(yt, yp, labels=labels, average="macro", zero_division=0)

        lo, hi = _bootstrap_ci(y_true, y_pred, _macro, n_bootstrap=bootstrap_n, ci=ci, seed=seed)
        result["macro_f1_ci_lower"] = lo
        result["macro_f1_ci_upper"] = hi
        result["ci_level"] = ci

    return result


# ---------------------------------------------------------------------------
# Probability interface
# ---------------------------------------------------------------------------

# Reporting tiers, keyed on the number of *evaluation* instances of a class
# in the pooled out-of-fold vector. Mirrors configs/e2_matrix.yaml.
REPORTING_TIERS = {
    "interpret":   {"min_n": 100, "rule": "reported and interpreted normally"},
    "ci_gated":    {"min_n": 30, "max_n": 99,
                    "rule": "directional claim only where bootstrap CIs do not overlap"},
    "report_only": {"max_n": 29,
                    "rule": "count and score for completeness; carries no claim"},
}


def tier_for_n(n: int) -> str:
    """Return the reporting tier for a class with n evaluation instances."""
    if n >= 100:
        return "interpret"
    if n >= 30:
        return "ci_gated"
    return "report_only"


def resolve_classes(labels) -> list:
    """
    Fix the global classes_ ordering for a dataset.

    The ordering is derived once from the full labelled pool and then passed
    into every fold, so that a class missing from one training fold cannot
    silently shift a probability column. Sorted by string form so the
    ordering is stable across runs, editions and dtypes (bool or str).
    """
    return sorted(set(labels), key=str)


def one_hot_proba(y_pred, classes) -> np.ndarray:
    """
    Hard-label adapter: turn predicted labels into a degenerate
    probability matrix with all mass on the predicted class.

    A component that cannot supply a distribution stays usable, but the
    resulting record must carry uncalibrated=True so that it is visible in the
    results table rather than silently averaged into the combination layer.
    """
    classes = list(classes)
    lookup = {c: i for i, c in enumerate(classes)}
    proba = np.zeros((len(y_pred), len(classes)), dtype=np.float64)
    for i, v in enumerate(y_pred):
        proba[i, lookup[v]] = 1.0
    return proba


def hard_label_proba_fn(predict_fn: Callable) -> Callable:
    """
    Wrap a hard-label predict_fn as a predict_proba_fn for run_cv.

    The returned callable carries `uncalibrated = True`; run_cv reads that
    attribute and propagates the flag into the result record.
    """
    def _proba_fn(model, df, classes):
        return one_hot_proba(predict_fn(model, df), classes)

    _proba_fn.uncalibrated = True
    return _proba_fn


def _bootstrap_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    n_bootstrap: int = BOOTSTRAP_N,
    ci: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
    """
    One bootstrap pass yielding the CI for macro-F1 and for every per-class F1.

    Running a single resampling loop keeps the macro interval and the per-class
    intervals consistent with each other (they are computed on the same
    resamples), which matters because the reporting rules gate directional claims on whether
    per-class intervals overlap.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    macro = np.empty(n_bootstrap)
    per_class = np.empty((n_bootstrap, len(classes)))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        per_class[b] = f1_score(yt, yp, labels=classes, average=None, zero_division=0)
        macro[b] = per_class[b].mean()
    alpha = (1 - ci) / 2
    macro_ci = (float(np.quantile(macro, alpha)), float(np.quantile(macro, 1 - alpha)))
    class_ci = {
        str(c): (float(np.quantile(per_class[:, j], alpha)),
                 float(np.quantile(per_class[:, j], 1 - alpha)))
        for j, c in enumerate(classes)
    }
    return macro_ci, class_ci


def pooled_oof_report(
    y_true,
    proba: np.ndarray,
    classes: list,
    *,
    bootstrap: bool = True,
    bootstrap_n: int = BOOTSTRAP_N,
    ci: float = 0.95,
    seed: int = DEFAULT_SEED,
    uncalibrated: bool = False,
) -> dict[str, Any]:
    """
    Score one pooled out-of-fold probability matrix under the reporting-tier rules.

    Pooled out-of-fold rather than the mean of per-fold scores: at the class
    sizes this project works with, a fold with zero true positives contributes
    a hard 0.0 and biases the per-fold mean downward as an artefact of the
    split, and pooling also makes the metric invariant to the fold count, which
    E4a needs at three folds.

    Every class carries its evaluation support, its reporting tier and, when
    bootstrap is on, its confidence interval, so that the interpretation rule
    travels with the number instead of living only in the prose.
    """
    y_true = np.asarray(y_true)
    classes = list(classes)
    y_pred = np.asarray([classes[i] for i in proba.argmax(axis=1)], dtype=y_true.dtype)

    macro_f1 = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    macro_ci = None
    class_ci: dict[str, tuple[float, float]] = {}
    if bootstrap:
        macro_ci, class_ci = _bootstrap_f1(
            y_true, y_pred, classes, n_bootstrap=bootstrap_n, ci=ci, seed=seed
        )

    classes_report = {}
    for j, c in enumerate(classes):
        n_c = int((y_true == c).sum())
        rec: dict[str, Any] = {
            "f1": float(per_class[j]),
            "support": n_c,
            "tier": tier_for_n(n_c),
        }
        if str(c) in class_ci:
            rec["ci_lower"], rec["ci_upper"] = class_ci[str(c)]
        classes_report[str(c)] = rec

    result: dict[str, Any] = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": classes_report,
        "labels": [str(c) for c in classes],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "n_samples": int(len(y_true)),
        "aggregation": "pooled_out_of_fold",
        "calibration": calibration_report(proba, y_true, classes, uncalibrated=uncalibrated),
    }
    if macro_ci is not None:
        result["macro_f1_ci_lower"], result["macro_f1_ci_upper"] = macro_ci
        result["ci_level"] = ci
        result["bootstrap_n"] = bootstrap_n
    return result


# ---------------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fold-level checkpointing
# ---------------------------------------------------------------------------
# Why this exists. `e2_runner` already skips a *component cell* whose results
# JSON is present, but a cell is the whole of one component on one subtask --
# every fold, seed and replicate inside it. On the encoder arm that is the
# expensive unit: 66 fine-tunes spread over three cells, so a crash in the
# middle throws away up to a third of the arm. Long unattended GPU runs make
# that a routine cost rather than a rare one.
#
# The invariant this must not break: a resumed run has to produce the SAME
# numbers as an uninterrupted one. Two properties buy that, and both were
# checked rather than assumed:
#   * a fold's training is a deterministic function of (params, fold data,
#     seed) -- `_build_model` calls torch.manual_seed(seed) and the loaders
#     carry their own seeded generator, so no global RNG walks across folds
#     and skipping fold 2 cannot move fold 3;
#   * everything downstream of the folds (pooled OOF, bootstrap CIs) is
#     computed at the end from the assembled matrix with an explicit seed.
# tests/test_fold_resume.py asserts the equality end to end rather than
# trusting either bullet.
#
# A checkpoint is refused, loudly, unless it was written by the same
# configuration: the caller passes a fingerprint (e2_runner builds it from
# config_id + data_rules_id + matrix version), and seed, split count, class
# ordering and the fold's exact validation indices must all match. Silently
# retraining on a mismatch would be worse than failing -- it would mix two
# configurations inside one pooled estimate, which is undetectable afterwards.

_CKPT_VERSION = 1


def _json_default(obj):
    """Numpy scalars and arrays -> JSON. Shared by save_results and the fold
    checkpoint so the two cannot diverge: the checkpoint originally used a bare
    json.dumps, which serialises a TF-IDF component's record fine and raises on
    the encoder's, where torch puts numpy floats into `model.record`, and no
    TML test could have seen it."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Not serialisable: {}".format(type(obj)))


def _normalise_payload(payload):
    """Put an on_fold record through the JSON round trip it will take anyway.

    Applied whether or not checkpointing is on, so a restored fold and a fresh
    one hold the same Python types -- otherwise a resumed run would carry
    floats where an uninterrupted one carried np.float32, and 'identical
    results' would quietly stop meaning identical objects."""
    if payload is None:
        return None
    return json.loads(json.dumps(payload, default=_json_default))


def _fold_ckpt_path(ckpt_dir: Path, fold_idx: int) -> Path:
    return Path(ckpt_dir) / "fold{:02d}.npz".format(fold_idx)


def _ckpt_meta(fingerprint, seed, n_splits, classes) -> dict:
    return {
        "version": _CKPT_VERSION,
        "fingerprint": str(fingerprint),
        "seed": int(seed),
        "n_splits": int(n_splits),
        "classes": [str(c) for c in classes],
    }


def save_fold_checkpoint(ckpt_dir, fold_idx, *, fingerprint, seed, n_splits,
                         classes, val_idx, fold_proba, payload=None) -> Path:
    """Persist one fold's out-of-fold block. Written atomically: a crash during
    the write leaves the old file or no file, never a truncated one."""
    path = _fold_ckpt_path(ckpt_dir, fold_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = _ckpt_meta(fingerprint, seed, n_splits, classes)
    meta["fold"] = int(fold_idx)
    # The handle is opened explicitly because np.savez appends '.npz' to a
    # *path* that does not already end in it -- which silently wrote
    # 'fold00.npz.tmp.npz' and left the rename below pointing at nothing.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            fold_proba=np.asarray(fold_proba, dtype=np.float64),
            val_idx=np.asarray(val_idx, dtype=np.int64),
            meta=np.array(json.dumps(meta)),
            payload=np.array(json.dumps(payload, default=_json_default)),
        )
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def load_fold_checkpoint(ckpt_dir, fold_idx, *, fingerprint, seed, n_splits,
                         classes, val_idx):
    """Return (fold_proba, payload) for a usable checkpoint, or None if there
    is none. Raises if one exists but was written by a different run."""
    path = _fold_ckpt_path(ckpt_dir, fold_idx)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        payload = json.loads(str(z["payload"]))
        fold_proba = z["fold_proba"]
        stored_val = z["val_idx"]

    want = _ckpt_meta(fingerprint, seed, n_splits, classes)
    want["fold"] = int(fold_idx)
    for key, expected in want.items():
        if meta.get(key) != expected:
            raise RuntimeError(
                "Fold checkpoint {} was written by a different run: {} is {!r}, "
                "this run has {!r}. Delete the checkpoint directory or point "
                "--checkpoint-dir somewhere else; reusing it would mix two "
                "configurations inside one pooled estimate.".format(
                    path, key, meta.get(key), expected)
            )
    if not np.array_equal(stored_val, np.asarray(val_idx, dtype=np.int64)):
        raise RuntimeError(
            "Fold checkpoint {} holds different validation indices than this "
            "run's split. Same fingerprint, different partition -- refusing to "
            "reuse it.".format(path)
        )
    if fold_proba.shape != (len(val_idx), len(classes)):
        raise RuntimeError(
            "Fold checkpoint {} has shape {}, expected {}.".format(
                path, fold_proba.shape, (len(val_idx), len(classes)))
        )
    return fold_proba, payload


def run_cv(
    df: pd.DataFrame,
    train_fn: Callable[[pd.DataFrame], Any],
    predict_fn: Callable[[Any, pd.DataFrame], np.ndarray] | None = None,
    *,
    predict_proba_fn: Callable[[Any, pd.DataFrame, list], np.ndarray] | None = None,
    classes: list | None = None,
    return_proba: bool = False,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    apply_qr: bool = True,
    bootstrap: bool = True,
    bootstrap_n: int = BOOTSTRAP_N,
    ci: float = 0.95,
    on_fold: Callable[[int, Any, pd.DataFrame, np.ndarray], None] | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_fingerprint: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run stratified k-fold CV and aggregate metrics across folds.

    Two modes, selected by which prediction callable is supplied.

    *Hard-label mode* (`predict_fn`, `return_proba=False`) is the original
    interface and is byte-for-byte unchanged: per-fold labels are inferred by
    `evaluate_predictions` exactly as before. Nothing that depends on the E0/E1
    figures is affected by the probability extension.

    *Probability mode* (`predict_proba_fn`, or `return_proba=True`) implements
    the probability interface: every fold returns an (n_items, n_classes) matrix under one
    globally fixed `classes_` ordering, the out-of-fold matrices are pooled
    into a single prediction per item, and the pooled vector is scored under
    the reporting-tier rules. Per-fold scores are retained as the dispersion
    statistic, not as the point estimate.

    Parameters
    ----------
    df : pd.DataFrame
        Full labelled dataset (description + label columns).
    train_fn : callable
        train_fn(train_df) -> model. Any inner split the component needs
        (hyperparameter selection, calibration) happens inside this callable,
        on the fold training portion only.
    predict_fn : callable | None
        predict_fn(model, val_df) -> array of hard labels.
    predict_proba_fn : callable | None
        predict_proba_fn(model, val_df, classes) -> (n_items, n_classes) array
        whose rows sum to 1 and whose column j is classes[j]. Set the attribute
        `uncalibrated = True` on the callable (see hard_label_proba_fn) if the
        distribution is degenerate.
    classes : list | None
        Global classes_ ordering. Inferred from df['label'] when omitted.
    return_proba : bool
        Force probability mode. With only a predict_fn supplied, the hard
        labels are lifted through the one-hot adapter and flagged uncalibrated.
    on_fold : callable | None
        on_fold(fold_idx, model, val_df, fold_proba) -- hook used by the
        component store to persist fitted components and their fold outputs.
        Whatever it RETURNS is collected into result['fold_payloads'] and, when
        checkpointing is on, stored with the fold and restored on resume. It
        must therefore be JSON-serialisable; return None if there is nothing to
        carry.
    checkpoint_dir : str | Path | None
        When given, each completed fold is written there and a rerun restores
        the finished folds instead of retraining them. Probability mode only.
        Requires checkpoint_fingerprint. Off by default, and when off this
        function behaves exactly as it did before checkpointing existed.
    checkpoint_fingerprint : str | None
        Identity of the configuration producing the folds. A checkpoint whose
        fingerprint, seed, split count, class ordering or validation indices
        differ is refused rather than reused.

    Returns
    -------
    dict. Always: folds, mean_macro_f1, std_macro_f1, mean_per_class_f1,
    n_splits, seed, fold_payloads, resumed_folds. In probability mode
    additionally: pooled (the pooled_oof_report), oof_proba, oof_pred,
    oof_true, oof_index, classes, uncalibrated.
    """
    if predict_fn is None and predict_proba_fn is None:
        raise ValueError("Supply either predict_fn or predict_proba_fn.")

    proba_mode = predict_proba_fn is not None or return_proba
    uncalibrated = bool(getattr(predict_proba_fn, "uncalibrated", False))
    if proba_mode and predict_proba_fn is None:
        predict_proba_fn = hard_label_proba_fn(predict_fn)
        uncalibrated = True

    if proba_mode and classes is None:
        classes = resolve_classes(df["label"].values)

    if checkpoint_dir is not None:
        if not proba_mode:
            raise NotImplementedError(
                "Fold checkpointing is implemented for probability mode only. "
                "Hard-label mode is the original path and is deliberately left "
                "byte-for-byte as it was."
            )
        if not checkpoint_fingerprint:
            raise ValueError(
                "checkpoint_dir requires checkpoint_fingerprint -- without it a "
                "checkpoint from another configuration could not be detected."
            )

    splits = make_cv_splits(df, n_splits=n_splits, seed=seed)
    fold_results = []
    fold_payloads: list[Any] = []
    resumed_folds: list[int] = []

    n_items = len(df)
    oof_proba = np.full((n_items, len(classes)), np.nan) if proba_mode else None
    oof_pred = np.empty(n_items, dtype=object) if proba_mode else None

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        val_fold = df.iloc[val_idx].copy().reset_index(drop=True)

        # A finished fold is restored instead of retrained. The metrics are
        # RECOMPUTED from the restored matrix rather than stored alongside it,
        # so a resumed fold and a fresh one go through exactly the same scoring
        # code and cannot drift apart.
        restored = None
        if checkpoint_dir is not None:
            restored = load_fold_checkpoint(
                checkpoint_dir, fold_idx,
                fingerprint=checkpoint_fingerprint, seed=seed,
                n_splits=n_splits, classes=classes, val_idx=val_idx,
            )

        if restored is not None:
            fold_proba, payload = restored
            resumed_folds.append(fold_idx)
            if verbose:
                print(f"  Fold {fold_idx}: restored from checkpoint")
        else:
            train_fold = df.iloc[train_idx].copy().reset_index(drop=True)

            # Apply data-quality rules within each fold
            train_fold, _ = apply_data_quality_rules(
                train_fold, val_fold, dedup=apply_qr, drop_overlap=True
            )

            model = train_fn(train_fold)

            if proba_mode:
                fold_proba = np.asarray(predict_proba_fn(model, val_fold, classes), dtype=np.float64)
                if fold_proba.shape != (len(val_fold), len(classes)):
                    raise ValueError(
                        "predict_proba_fn returned {} for fold {}, expected {}.".format(
                            fold_proba.shape, fold_idx, (len(val_fold), len(classes))
                        )
                    )
                row_sums = fold_proba.sum(axis=1)
                if not np.allclose(row_sums, 1.0, atol=1e-6):
                    raise ValueError(
                        "Fold {} probability rows do not sum to 1 "
                        "(min {:.6f}, max {:.6f}).".format(
                            fold_idx, float(row_sums.min()), float(row_sums.max())
                        )
                    )
                payload = _normalise_payload(
                    on_fold(fold_idx, model, val_fold, fold_proba) if on_fold is not None else None)
                if checkpoint_dir is not None:
                    save_fold_checkpoint(
                        checkpoint_dir, fold_idx,
                        fingerprint=checkpoint_fingerprint, seed=seed,
                        n_splits=n_splits, classes=classes, val_idx=val_idx,
                        fold_proba=fold_proba, payload=payload,
                    )
            else:
                payload = None
                y_pred_hard = predict_fn(model, val_fold)
                metrics = evaluate_predictions(val_fold["label"].values, y_pred_hard, bootstrap=False)

        if proba_mode:
            y_pred = np.asarray([classes[i] for i in fold_proba.argmax(axis=1)])
            oof_proba[val_idx] = fold_proba
            oof_pred[val_idx] = y_pred
            metrics = evaluate_predictions(
                val_fold["label"].values, y_pred, labels=classes, bootstrap=False
            )

        fold_payloads.append(payload)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)
        if verbose:
            print(f"  Fold {fold_idx}: macro-F1 = {metrics['macro_f1']:.4f}")

    macro_scores = [r["macro_f1"] for r in fold_results]
    mean_macro = float(np.mean(macro_scores))
    std_macro = float(np.std(macro_scores, ddof=1))

    # Average per-class F1 across folds, over the folds that actually contain
    # the class. A class absent from a fold's validation part is undefined
    # there, not scored 0.0, and averaging the 0.0 in would understate exactly
    # the rare classes Dimension 1 is about. `folds_present` travels alongside
    # so a thin average cannot be read as a solid one; a class present in no
    # fold at all yields None rather than a number. This field is a dispersion
    # aid -- Chapter 4 reports the *pooled* per-class figures, where every
    # class carries its full support.
    all_classes = sorted(set().union(*[set(r["labels"]) for r in fold_results]), key=str)
    mean_per_class: dict[str, float | None] = {}
    folds_present: dict[str, int] = {}
    for cls in all_classes:
        scores = [r["per_class_f1"][cls] for r in fold_results
                  if r.get("per_class_support", {}).get(cls, 1) > 0]
        folds_present[cls] = len(scores)
        mean_per_class[cls] = float(np.mean(scores)) if scores else None

    if verbose:
        print(f"  CV mean macro-F1 = {mean_macro:.4f} +/- {std_macro:.4f}")

    result: dict[str, Any] = {
        "folds": fold_results,
        "mean_macro_f1": mean_macro,
        "std_macro_f1": std_macro,
        "mean_per_class_f1": mean_per_class,
        "mean_per_class_folds_present": folds_present,
        "n_splits": n_splits,
        "seed": seed,
        "fold_payloads": fold_payloads,
        # Which folds came off disk. On a bit-reproducible component this is
        # bookkeeping; on the encoder, where GPU training is not bit-identical
        # across processes anyway, it is provenance -- a resumed cell mixes
        # folds trained in different processes and the artefact must say so.
        "resumed_folds": resumed_folds,
    }

    if proba_mode:
        if np.isnan(oof_proba).any():
            raise RuntimeError(
                "Out-of-fold matrix has unfilled rows; the folds did not "
                "partition the data."
            )
        y_true = df["label"].values
        result["classes"] = [str(c) for c in classes]
        result["uncalibrated"] = uncalibrated
        result["oof_proba"] = oof_proba
        result["oof_pred"] = oof_pred
        result["oof_true"] = y_true
        result["oof_index"] = (df["id"].values if "id" in df.columns
                               else np.arange(n_items))
        result["pooled"] = pooled_oof_report(
            y_true, oof_proba, classes,
            bootstrap=bootstrap, bootstrap_n=bootstrap_n, ci=ci, seed=seed,
            uncalibrated=uncalibrated,
        )
        if verbose:
            p = result["pooled"]
            print(f"  Pooled OOF macro-F1 = {p['macro_f1']:.4f} "
                  f"(ECE {p['calibration']['ece']:.4f}, "
                  f"Brier {p['calibration']['brier']:.4f})")

    return result


# ---------------------------------------------------------------------------
# Environment provenance
# ---------------------------------------------------------------------------

def _requires_python_floor() -> str | None:
    """The declared interpreter floor, read from pyproject rather than hardcoded."""
    import tomllib

    path = Path(__file__).parent.parent / "pyproject.toml"
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)["project"]["requires-python"]
    except Exception:
        return None


# Set by env_pin(); read by baselines._get_sbert. See the block inside env_pin.
_ENV_PIN_CALLED = False


def env_pin() -> dict[str, Any]:
    """
    Record the resolved environment that produced a result.

    Every gate file and every result JSON carries one of these. A number
    without its stack is not reproducible, which is the lesson of the
    52.01 -> 48.41 episode: the same code and data gave macro-F1 differing by
    ~3.6 pp across dependency stacks, and no stack had been recorded.

    This is a *measurement*, so a gate file must recompute it on every run and
    must never carry an old one forward -- only the human judgement fields are
    preserved. torch and its device are included
    because the C2A baseline encodes through SentenceBERT, so the device is
    part of that number's provenance.

    Call this **after** all model runs of the process, once. It imports
    lightgbm and torch, and a SentenceBERT run afterwards loads a second
    OpenMP runtime into the same process, which exits 139 (the run that
    matters is the *next* one, not the previous one).
    """
    import platform

    import scipy
    import sklearn

    # The ordering contract above is enforced, not merely documented: a dict
    # literal such as {"env_pin": env_pin(), "runs": [...]} evaluates top to
    # bottom, runs SentenceBERT after the pin and dies with SIGSEGV.
    # `baselines.BaselineBase._get_sbert` reads this flag and refuses with the
    # reason, which turns a segmentation fault into a sentence naming the fix.
    global _ENV_PIN_CALLED
    _ENV_PIN_CALLED = True

    pin: dict[str, Any] = {
        "python": platform.python_version(),
        "requires_python_floor": _requires_python_floor(),
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
    }
    for name in ("xgboost", "lightgbm"):
        try:
            pin[name] = __import__(name).__version__
        except Exception:
            pin[name] = None
    try:
        import torch
        pin["torch"] = torch.__version__
        pin["cuda_available"] = bool(torch.cuda.is_available())
        pin["cuda_device"] = (torch.cuda.get_device_name(0)
                              if torch.cuda.is_available() else None)
        # The accelerator a library picks when it is not told which to use.
        # `cuda_available` says whether CUDA *exists*,
        # not what ran, and the two come apart exactly where it matters: the
        # canonical C2A baseline 57.4000 was produced with cuda_available
        # false, and it still served SentenceBERT on mps:0, because
        # `baselines.py` passes no device and the library chooses. So a
        # provenance record that stops at cuda_available describes that number
        # as a CPU measurement, which it is not.
        #
        # Recorded, not enforced. The machine key is OS/architecture and
        # cannot see the device, so two runs on one machine -- one forced to
        # CPU, one on CUDA -- pass the same-machine guard; making the device a
        # blocking condition would be a rule with no measured effect behind it
        # (C2A re-measured after the CUDA install gave a bit-identical figure).
        # This field makes such a mismatch auditable afterwards, which is what
        # the guard cannot do.
        #
        # Computed from torch alone on purpose: importing sentence_transformers
        # here would pull the second OpenMP runtime into every env_pin() call.
        # Verified on Darwin/arm64 that the two agree --
        # `sentence_transformers.util.get_device_name()` returns "mps" and a
        # loaded SentenceTransformer reports device mps:0, against "mps" here.
        mps = bool(getattr(torch.backends, "mps", None)
                   and torch.backends.mps.is_available())
        pin["mps_available"] = mps
        pin["torch_auto_device"] = ("cuda" if torch.cuda.is_available()
                                    else "mps" if mps else "cpu")
    except Exception:
        pin["torch"] = None
        pin["cuda_available"] = False
        pin["cuda_device"] = None
        pin["mps_available"] = False
        pin["torch_auto_device"] = None

    # Platform and the NLTK corpus. Platform,
    # because identical package versions turned out NOT to be sufficient for
    # identical numbers: the DBO baseline gives 48.4109 on Windows/x86_64 and
    # 51.4021 on macOS/arm64 off the same versions and the same data, so a
    # figure without its platform is not attributable. The NLTK corpus, because
    # it is downloaded data outside uv.lock -- its absence is why the C2A
    # baseline cannot run on a fresh machine at all. Recording presence turns a silent
    # environment fact into part of the run's provenance, the same move
    # data_manifest.json makes for the inputs.
    pin["platform"] = f"{platform.system()}/{platform.machine()}"
    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab/german/")
        pin["nltk_punkt_tab"] = True
    except Exception:
        pin["nltk_punkt_tab"] = False
    return pin


# ---------------------------------------------------------------------------
# G0 check
# ---------------------------------------------------------------------------

def check_g0_counts(edition: str = "2025") -> dict[str, dict]:
    """
    Split-count check (printed as G0): verify that our data loader reproduces the organizer-reported
    train/test split counts (felser2025, Table specs).

    Parameters
    ----------
    edition : str
        "2025" (default) — only 2025 has published counts to verify against.

    Returns
    -------
    dict mapping subtask -> {
        "train_expected", "train_actual", "train_ok",
        "test_expected",  "test_actual",  "test_ok",
        "pass": bool
    }
    """
    results = {}
    all_pass = True

    for subtask, targets in G0_TARGETS.items():
        row: dict[str, Any] = {}
        for split in ("train", "test"):
            expected = targets[split]
            try:
                df = load_split(subtask, edition, split)
                actual = len(df)
            except Exception as exc:
                actual = -1
                print(f"  [G0] ERROR loading {subtask}/{edition}/{split}: {exc}")
            ok = actual == expected
            row[f"{split}_expected"] = expected
            row[f"{split}_actual"] = actual
            row[f"{split}_ok"] = ok
            if not ok:
                all_pass = False
                print(
                    f"  [G0] FAIL {subtask} {split}: expected {expected}, got {actual}"
                )
            else:
                print(f"  [G0] OK   {subtask} {split}: {actual}")

        row["pass"] = row["train_ok"] and row["test_ok"]
        results[subtask] = row

    results["_all_pass"] = all_pass
    if all_pass:
        print("\n  ✓ G0 PASSED — all split counts match organizer targets.")
    else:
        print("\n  ✗ G0 FAILED — fix the data loader before running any model.")

    return results


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str | Path) -> None:
    """Save a results dict as JSON (numpy scalars serialised automatically)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"  Saved results to {path}")


# ---------------------------------------------------------------------------
# Measuring modules: describe on --help, refuse to overwrite
# ---------------------------------------------------------------------------

def measuring_main_guard(description: str, out_path: str | Path, argv=None):
    """Argument parsing plus an overwrite refusal for a module that MEASURES.

    Why. A `--help` probe of a module that takes no arguments starts a real
    run: the probe *is* the invocation. So the two properties are enforced
    here:

    1. **`--help` describes rather than executes.** argparse exits before the
       caller does any work, which is only true if the parse happens FIRST --
       hence a guard called at the top of `main`, not a check before the write.
    2. **A committed artefact is not silently replaced.** Every caller of this
       writes a file that is in git and that some figure, table or claim rests
       on. Overwriting it is a decision, so it needs `--force`.

    The pattern is `e1_machine_baseline`'s, lifted to one place so every
    measuring module inherits it rather than re-deriving it.

    Returns the parsed args (with `.out` as a Path) or **None** when the caller
    should do nothing -- the artefact exists and `--force` was not given. A
    caller that ignores the None writes anyway, so callers return on it:

        def main() -> int:
            args = measuring_main_guard(__doc__, RESULTS / "thing.json")
            if args is None:
                return 0
            ...
            args.out.write_text(...)
            return 0
    """
    import argparse

    out_path = Path(out_path)
    ap = argparse.ArgumentParser(
        description=(description or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--force", action="store_true",
                    help="overwrite the artefact if it already exists")
    ap.add_argument("--out", default=str(out_path),
                    help="artefact path (default: %(default)s)")
    args = ap.parse_args(argv)
    args.out = Path(args.out)

    if args.out.exists() and not args.force:
        print("SKIP: {} already exists and this module measures rather than "
              "reads.\n      Re-measure with --force only if you mean to "
              "replace it.".format(args.out))
        return None
    return args
