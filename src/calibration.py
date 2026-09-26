"""
calibration.py -- probability calibration and calibration metrics.

Implements the calibration half of the component probability interface
declared in `configs/e2_matrix.yaml` (calibration.method = temperature_scaling;
metrics ECE, Brier, reliability).

Two rules govern everything in this module and must not be relaxed:

1. Calibration is fitted inside the fold, on a held-out slice of that fold's
   training portion, never on its validation part. Fitting a temperature on
   evaluation data leaks the evaluation signal into the confidence estimates
   that the cascade and the judge then consume (3.2, subsec:combination).
2. Calibration metrics are computed on the pooled out-of-fold probabilities
   (the protocol's aggregation rule), not per fold and averaged.

Temperature scaling operates on log-probabilities rather than on raw logits,
because the component interface is a probability matrix: a component may
be an SVM behind Platt scaling, a gradient-boosted tree with a native
predict_proba, or an LLM with normalised label-token probabilities, none of
which expose comparable logits. Dividing log p by T and renormalising is the
same one-parameter family as logit scaling whenever the probabilities came
from a softmax, and it is well defined when they did not (guo2017).

Usage
-----
from src.calibration import fit_temperature, apply_temperature, calibration_report
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

# Probabilities are clipped before any log, so that a hard 0 (one-hot adapter,
# or a tree model that assigns zero mass to a class) cannot produce -inf.
_EPS = 1e-12

DEFAULT_N_BINS = 15
_T_BOUNDS = (0.05, 20.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_index(y_true, classes: Sequence) -> np.ndarray:
    """Map label values to column indices of the classes_ ordering."""
    lookup = {c: i for i, c in enumerate(classes)}
    try:
        return np.asarray([lookup[v] for v in y_true], dtype=int)
    except KeyError as exc:
        raise ValueError(
            "Label {!r} is not in the declared classes_ ordering {!r}. "
            "The ordering must be fixed globally, not inferred per fold.".format(
                exc.args[0], list(classes)
            )
        ) from None


def _normalise(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, None)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """
    Rescale a probability matrix by a temperature and renormalise.

    T > 1 softens the distribution (lowers confidence), T < 1 sharpens it.
    T == 1 is the identity up to clipping and renormalisation.
    """
    proba = _normalise(proba)
    if temperature == 1.0:
        return proba
    logp = np.log(proba) / float(temperature)
    logp -= logp.max(axis=1, keepdims=True)     # stabilise before exp
    scaled = np.exp(logp)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(proba: np.ndarray, y_true, classes: Sequence) -> dict[str, Any]:
    """
    Fit a single temperature by minimising the negative log-likelihood.

    Parameters
    ----------
    proba : (n_items, n_classes) array
        Probabilities produced by the component on the calibration slice of
        the fold training portion.
    y_true : array-like
        Gold labels for that slice.
    classes : sequence
        The classes_ ordering; column j of proba is classes[j].

    Returns
    -------
    dict with keys temperature, nll_before, nll_after, n_calibration_items,
    fitted (bool), and note (only when the fit was skipped).

    A degenerate slice (fewer than two distinct gold labels, or fewer items
    than classes) returns T = 1.0 with fitted = False rather than a
    temperature fitted on nothing.
    """
    proba = _normalise(proba)
    y_idx = _as_index(y_true, classes)
    n = len(y_idx)

    def _nll(t: float) -> float:
        p = apply_temperature(proba, t)
        return float(-np.mean(np.log(np.clip(p[np.arange(n), y_idx], _EPS, None))))

    nll_before = _nll(1.0)

    if n < max(2, len(classes)) or len(np.unique(y_idx)) < 2:
        return {
            "temperature": 1.0,
            "nll_before": nll_before,
            "nll_after": nll_before,
            "n_calibration_items": int(n),
            "fitted": False,
            "note": "degenerate calibration slice; temperature left at 1.0",
        }

    res = minimize_scalar(_nll, bounds=_T_BOUNDS, method="bounded",
                          options={"xatol": 1e-4})
    t_hat = float(res.x)
    nll_after = _nll(t_hat)

    # Never let the fit make the likelihood worse than the identity.
    if nll_after > nll_before:
        t_hat, nll_after = 1.0, nll_before

    return {
        "temperature": t_hat,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "n_calibration_items": int(n),
        "fitted": True,
    }


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def reliability_bins(
    proba: np.ndarray,
    y_true,
    classes: Sequence,
    n_bins: int = DEFAULT_N_BINS,
) -> list[dict[str, Any]]:
    """
    Bin the top-label confidences for a reliability diagram.

    Returns one record per non-empty bin with the bin edges, the number of
    items, the mean confidence and the observed accuracy. The diagram itself
    is plotted in the results chapter; this module only produces the data, so
    that the figure and the reported ECE cannot drift apart.
    """
    proba = _normalise(proba)
    y_idx = _as_index(y_true, classes)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_idx).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, Any]] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        out.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "n": n_b,
            "mean_confidence": float(conf[mask].mean()),
            "accuracy": float(correct[mask].mean()),
        })
    return out


def expected_calibration_error(
    proba: np.ndarray,
    y_true,
    classes: Sequence,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """
    Top-label expected calibration error with equal-width confidence bins.

    ECE = sum_b (n_b / n) * |accuracy(b) - mean confidence(b)|, where the
    confidence of an item is the probability assigned to its predicted class
    (guo2017). Equal-width bins are used rather than equal-mass bins so that
    the value is comparable across components with different confidence
    distributions.
    """
    bins = reliability_bins(proba, y_true, classes, n_bins=n_bins)
    n_total = sum(b["n"] for b in bins)
    if n_total == 0:
        return float("nan")
    return float(
        sum(b["n"] / n_total * abs(b["accuracy"] - b["mean_confidence"]) for b in bins)
    )


def brier_score(proba: np.ndarray, y_true, classes: Sequence) -> float:
    """
    Multiclass Brier score: mean over items of the squared distance between
    the predicted distribution and the one-hot gold vector.

    Range [0, 2]; lower is better. For a binary task this is twice the usual
    binary Brier score, which is the standard multiclass convention and is
    stated here so that the numbers in Chapter 4 are not misread.
    """
    proba = _normalise(proba)
    y_idx = _as_index(y_true, classes)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def calibration_report(
    proba: np.ndarray,
    y_true,
    classes: Sequence,
    *,
    n_bins: int = DEFAULT_N_BINS,
    uncalibrated: bool = False,
) -> dict[str, Any]:
    """
    Full calibration record for one pooled out-of-fold probability matrix.

    uncalibrated=True marks a component whose probabilities are one-hot (the
    hard-label adapter of the component interface). Its ECE and Brier score are still computed and
    reported, but the flag travels with them so that a component which cannot
    express uncertainty is visible in the results table rather than silently
    averaged into the combination layer.
    """
    return {
        "ece": expected_calibration_error(proba, y_true, classes, n_bins=n_bins),
        "brier": brier_score(proba, y_true, classes),
        "n_bins": n_bins,
        "reliability_bins": reliability_bins(proba, y_true, classes, n_bins=n_bins),
        "uncalibrated": bool(uncalibrated),
        "n_items": int(len(np.asarray(y_true))),
    }
