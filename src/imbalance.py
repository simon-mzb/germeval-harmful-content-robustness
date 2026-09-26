"""
imbalance.py -- the imbalance conditions of E4a, in one place.

WHY ONE MODULE AND NOT THREE COPIES. Class weighting and random oversampling
apply to the classical models, the encoder and the fine-tuned LLM; focal loss to
the two neural families. If each family carried its own copy of "balanced
weights" or "oversample to the majority", the comparison ACROSS families -- which
is what Dimension 1 reports -- would rest on three implementations that could
drift apart silently. So the
definition lives here, the parameters live in `e2_matrix.yaml` under
`e4_protocol`, and the components only call.

THE DEFINITIONS (pre-registered in `e4_protocol`, argued in 3.3):

  * none -- the reference cell, exactly the E2 training objective.
  * class_weighting -- inverse class frequency on the FIT slice, the `balanced`
    formula n / (k * n_c) that the organisers' baseline uses (3.2 cites it).
    Under this formula the expected weight over the fit slice is exactly 1, so
    the loss keeps its scale and the learning rate keeps its meaning; that is
    why the weighted loss is the plain mean of w_i * l_i and NOT torch's
    "weighted mean", which renormalises by the batch's weight sum and so
    re-balances every batch to scale 1 whatever its composition.
  * focal_loss -- lin2017, gamma from the matrix, no alpha term: an alpha would
    re-introduce class weighting inside the focal cell and the two conditions
    would no longer be separable.
  * random_oversampling -- minority items drawn WITH replacement until every
    class present in the fit slice reaches the majority count (buda2018's
    full-balance setting). FIT slice only: the selection and calibration slices
    must keep the natural distribution, or epoch selection and temperature
    scaling would be tuned on a population the evaluation never sees. And after
    deduplication, which `run_cv` applies to the fold before `train_fn` is
    called -- the ordering constraint of 3.2.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CONDITIONS = ("none", "class_weighting", "focal_loss", "random_oversampling")


def protocol(matrix: dict[str, Any]) -> dict[str, Any]:
    """The `e4_protocol` block, refused rather than defaulted when absent."""
    block = matrix.get("e4_protocol")
    if not block:
        raise ValueError(
            "e2_matrix.yaml carries no `e4_protocol` block: the E4 conditions "
            "have no declared parameters, and a default here would be a protocol "
            "decision nobody recorded")
    return block


def family_key(component: str, family: str) -> str:
    """The key `imbalance_applicability` uses: TML by component, neural by family."""
    return component if family == "tml" else family


def check_applicable(condition: str, component: str, family: str,
                     matrix: dict[str, Any]) -> None:
    """Refuse a condition the E4a protocol does not define for this component."""
    if condition not in CONDITIONS:
        raise ValueError("unknown imbalance condition {!r}; the protocol defines {}".format(
            condition, ", ".join(CONDITIONS)))
    allowed = matrix["imbalance_applicability"].get(condition, [])
    key = family_key(component, family)
    if key not in allowed:
        raise ValueError(
            "the protocol does not define {!r} for {} (applicability key {!r}); it is "
            "defined for {}. The comparison runs where each strategy is defined, "
            "not as a crossed grid (3.2 subsec:imbalance).".format(
                condition, component, key, allowed))


def focal_gamma(matrix: dict[str, Any]) -> float:
    return float(protocol(matrix)["e4a"]["conditions"]["focal_loss"]["gamma"])


def balanced_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class weights n / (k * n_c) over the classes PRESENT in `y`.

    A class absent from the fit slice gets weight 1.0; it has no item to weight,
    so the value is never used, and 0 would make a later reader think it was
    deliberately silenced.
    """
    y = np.asarray(y)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = counts > 0
    k = int(present.sum())
    w = np.ones(n_classes, dtype=np.float64)
    w[present] = len(y) / (k * counts[present])
    return w


def oversample_indices(y: np.ndarray, seed: int) -> np.ndarray:
    """Indices into `y`: every original item once, plus minority draws to balance.

    Deterministic in `seed`. The originals come first and in order, so the
    oversampled slice contains the natural one as a prefix -- which is what lets
    a test assert that nothing was dropped.
    """
    y = np.asarray(y)
    counts = np.bincount(y)
    target = int(counts.max())
    rng = np.random.RandomState(seed)
    extra = []
    for c in np.flatnonzero(counts):
        pool = np.flatnonzero(y == c)
        need = target - len(pool)
        if need > 0:
            extra.append(rng.choice(pool, size=need, replace=True))
    if not extra:
        return np.arange(len(y))
    return np.concatenate([np.arange(len(y))] + extra)


def classification_loss(logits, target, *, weight=None, gamma: float = 0.0):
    """Mean over items of w_{y_i} * (1 - p_i)^gamma * CE_i, in float32.

    gamma = 0 and weight = None is plain cross-entropy, identical to what a
    HuggingFace sequence classifier computes internally; the suite asserts that
    equality so the `none` path cannot drift from E2.
    """
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    logpt = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    loss = -logpt
    if gamma:
        loss = loss * (1.0 - logpt.exp()).clamp(min=0.0) ** gamma
    if weight is not None:
        loss = loss * weight.to(loss.device, loss.dtype)[target]
    return loss.mean()


def token_loss(logits, labels, *, item_weight=None, gamma: float = 0.0):
    """Causal-LM loss on the verbaliser positions, with the E4a conditions.

    `labels` is -100 everywhere except the label tokens (llm_protocol.objective);
    positions are shifted by one exactly as HuggingFace does. Only the selected
    positions are upcast to float32 -- the full vocabulary tensor is never
    copied, because copying it ran out of GPU memory on this arm.

    `item_weight` (shape [batch]) is broadcast to every label token of its item.
    The reduction is the mean over label tokens, which is what HuggingFace's
    loss is, so gamma = 0 with no weight reproduces `model(**batch).loss`.
    """
    import torch

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    sel_logits = shift_logits[mask].float()
    sel_labels = shift_labels[mask]
    logp = torch.log_softmax(sel_logits, dim=-1)
    logpt = logp.gather(-1, sel_labels.unsqueeze(-1)).squeeze(-1)
    loss = -logpt
    if gamma:
        loss = loss * (1.0 - logpt.exp()).clamp(min=0.0) ** gamma
    if item_weight is not None:
        rows = torch.arange(labels.shape[0], device=labels.device)
        row_of_token = rows.unsqueeze(1).expand_as(shift_labels)[mask]
        loss = loss * item_weight.to(loss.device, loss.dtype)[row_of_token]
    return loss.mean()
