"""
encoder_components.py -- the fine-tuned encoder component of the pipeline (E2).

ModernGBERT (1B primary, 134M reference) adapted with LoRA, speaking the same
probability interface as the classical components. The fold
protocol is `encoder_protocol` in `configs/e2_matrix.yaml`; it is deliberately
the same shape as `tml_protocol`, so that 3.3 describes one protocol and two
component families obey it.

Protocol inside one CV fold
---------------------------
The fold training portion is split, stratified, into three disjoint parts, by
the *same* function the classical components use:

    fit (80%)       -> fine-tunes every grid configuration
    select (10%)    -> scores them after every epoch; the best (configuration,
                       epoch) pair wins
    calibrate (10%) -> temperature scaling, on the winner only

The validation part of the fold is touched by nothing except the final
prediction, so hyperparameter selection, epoch selection and calibration are
all nested inside the fold.

Three recorded deviations from `configs/e2_matrix.yaml` v1.1, all resolved in
v1.2 before the encoder arm ran:

* `epochs` is not a grid axis. The model is evaluated on the selection slice
  after every epoch and the best epoch's adapter is kept, so one fine-tune
  answers the whole epoch axis. This is the `n_estimators` precedent of v1.1
  applied to a second family, and it is the better decision procedure for the
  same reason: the length is chosen per fit rather than once for all fits.
* `lora_alpha` is tied to `lora_r` at alpha = 2r. Crossing the two axes
  confounds adapter capacity with update scaling -- LoRA scales its update by
  alpha/r, so r=8/alpha=16 and r=16/alpha=32 differ in capacity at *identical*
  scaling while the other two cells differ in scaling at fixed capacity. Tying
  them fixes the scaling at the conventional value and leaves r as the pure
  capacity axis.
* Together these re-cut the grid from 16 cells to 4, which takes the E2 encoder
  arm from 246 fine-tunes to 66.

Unlike the classical components this module does **not** need the
`_align_columns` machinery: the classification head is always built with the
global number of classes, so a class missing from a fit slice occupies its own
column and simply never receives mass. There is no column to shift.

Usage
-----
from src.encoder_components import ENCODER_COMPONENTS, make_train_fn, predict_proba_fn
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.calibration import apply_temperature, fit_temperature
from src.imbalance import (balanced_weights, classification_loss, focal_gamma,
                           oversample_indices)
from src.preprocessing import clean_encoder
# The inner split is imported rather than reimplemented on purpose. Every
# component must see the identical fit/select/calibrate partition of a fold, or
# a difference between components stops being a difference between models.
from src.tml_components import INNER_SPLIT, _three_way_split


MODEL_IDS = {
    "encoder_1b": "LSX-UniWue/ModernGBERT_1B",
    "encoder_134m": "LSX-UniWue/ModernGBERT_134M",
}

# configs/e2_matrix.yaml v1.2: lora_r x lr, alpha tied at 2r, epochs by selection.
def _grid_from_matrix(component: str) -> list[dict[str, Any]]:
    """The search grid, read from configs/e2_matrix.yaml -- the ONE source.

    Read, never restated. A grid hardcoded here beside the one the matrix
    declares has two failure modes, the second worse than the first:

      1. Editing the matrix (say, to widen the lr window) changes nothing; the
         run uses the old window regardless.
      2. `config_id` is computed FROM THE MATRIX. So a stored run would carry
         the identity of a configuration it never used, in a way no check can
         catch, because every check compares the matrix against itself.

    `inherits` is resolved the same way `compute_config_id` resolves it, so
    encoder_134m keeps picking up the 1B's grid rather than restating it.
    There is deliberately NO fallback: if the matrix cannot be read, that is a
    broken checkout and it should stop, not quietly train something else.
    """
    from src.component_store import load_matrix

    block = load_matrix()["components"][component]
    if "grid" not in block and block.get("inherits"):
        block = load_matrix()["components"][block["inherits"]]
    grid = block["grid"]
    keys = sorted(grid)                      # deterministic cell order
    cells: list[dict[str, Any]] = [{}]
    for k in keys:
        values = grid[k] if isinstance(grid[k], list) else [grid[k]]
        cells = [dict(c, **{k: v}) for c in cells for v in values]
    return cells


GRIDS: dict[str, list[dict[str, Any]]] = {
    name: _grid_from_matrix(name) for name in MODEL_IDS
}

ENCODER_COMPONENTS = tuple(MODEL_IDS)

def _epoch_selection_from_matrix() -> dict[str, Any]:
    """The epoch-stopping rule, read from configs/e2_matrix.yaml -- the ONE source.

    Same reason as `_grid_from_matrix`: a constant restated here would leave
    `config_id` -- computed FROM the matrix -- describing a configuration the
    code never ran the moment the two disagreed. There is deliberately no
    fallback: an unreadable matrix is a broken checkout and should stop.
    """
    from src.component_store import load_matrix

    sel = load_matrix()["encoder_protocol"]["epoch_selection"]
    return {"max_epochs": int(sel["max_epochs"]), "patience": int(sel["patience"])}


_EPOCH_SELECTION = _epoch_selection_from_matrix()

# The CEILING, not the stopping criterion: v1.7 closes the epoch axis with early
# stopping on the selection slice.
#
# ⚠️ IT IS NOT "ONLY A COMPUTE BOUND". The linear schedule is built over
# `total_steps = steps_per_epoch * max_epochs`, so this value rescales the
# learning rate at EVERY step, including every step of a fit that stops at
# epoch 3. v1.8 withdrew the dominance claim for exactly this reason; the same
# reason makes the ceiling a PROTOCOL parameter that also happens to bound
# compute, declared in the matrix like effective_batch_size. Changing it invalidates
# comparability with existing runs even for fits that never approach it.
MAX_EPOCHS = _EPOCH_SELECTION["max_epochs"]
# Consecutive epochs without a new best before training stops. Derived from the
# 14 epoch traces this project has produced: the longest non-improving run that
# was still followed by a new best is 1, never 2. See e2_matrix.yaml v1.7.
PATIENCE = _EPOCH_SELECTION["patience"]
MAX_SEQ_LEN = 256

# Fixed rather than swept, and stated rather than left to a library default --
# the under-reporting Zimmerman, Fox & Kruschwitz (2018) criticise is exactly
# this class of value.
FIXED = {
    "effective_batch_size": 32,
    "per_device_batch_size": 16,     # memory detail; effective size is what binds
    "eval_batch_size": 64,
    "warmup_ratio": 0.10,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lora_dropout": 0.05,
    "lora_target_modules": ["attn.Wqkv", "attn.Wo"],
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
}


# ---------------------------------------------------------------------------
# Device and precision
# ---------------------------------------------------------------------------

def resolve_device(requested: str | None = None) -> str:
    """
    cuda where it exists, else cpu. `requested` overrides and is the only way
    to reach mps.

    MPS is deliberately not auto-selected, and the reason is stability rather
    than speed. On the MacBook (macOS 15.7.5, torch 2.12.1) the 384-example
    smoke run had not finished after 40 minutes on MPS and ended in a kernel
    panic that rebooted the machine; the same run takes 812 s on CPU, so MPS
    was perhaps threefold slower -- unpleasant, but survivable. The reboot is
    not. ModernBERT's attention path is the likely cause. Since the target
    device is CUDA, MPS adds risk to the smoke test without adding evidence
    about the arm. Pass device="mps" explicitly if it is ever worth re-testing.
    """
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def autocast_dtype(device: str):
    """
    bf16 on CUDA, fp32 everywhere else.

    The MPS path exists for the smoke test only. It proves the code path runs;
    it does not reproduce the numbers, because the real arm trains in bf16 on
    CUDA and that is a different numerical path.
    """
    import torch

    return torch.bfloat16 if device == "cuda" else None


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _texts(df: pd.DataFrame) -> list[str]:
    return clean_encoder(df["description"]).tolist()


def _encode(tokeniser, texts: list[str], max_seq_len: int):
    """
    Token ids per text, truncated to max_seq_len, plus the truncation rate.

    Padding is deliberately *not* applied here. The median item is around a
    dozen words while max_seq_len is 256, so padding the whole slice to a fixed
    width would spend most of the compute on padding; the collate function pads
    each batch to its own longest member instead.
    """
    full = tokeniser(texts, add_special_tokens=True, truncation=False)["input_ids"]
    lengths = np.array([len(ids) for ids in full])
    truncated = [ids[:max_seq_len] for ids in full]
    return truncated, {
        "truncation_rate": float((lengths > max_seq_len).mean()),
        "token_len_mean": float(lengths.mean()),
        "token_len_p95": float(np.percentile(lengths, 95)),
        "token_len_max": int(lengths.max()),
    }


def _make_loader(ids: list[list[int]], labels: np.ndarray | None, pad_id: int,
                 batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _Slice(Dataset):
        def __len__(self):
            return len(ids)

        def __getitem__(self, i):
            return i

    def _collate(batch_idx):
        rows = [ids[i] for i in batch_idx]
        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(rows), width), dtype=torch.long)
        for j, r in enumerate(rows):
            input_ids[j, :len(r)] = torch.tensor(r, dtype=torch.long)
            attn[j, :len(r)] = 1
        out = {"input_ids": input_ids, "attention_mask": attn}
        if labels is not None:
            out["labels"] = torch.tensor([labels[i] for i in batch_idx], dtype=torch.long)
        return out

    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(_Slice(), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, collate_fn=_collate)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_model(component: str, params: dict, n_classes: int, seed: int):
    """A fresh LoRA-adapted sequence classifier. The head is trained in full."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(seed)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_IDS[component], num_labels=n_classes,
    )
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=params["lora_r"],
        lora_alpha=2 * params["lora_r"],          # tied; see the module docstring
        lora_dropout=FIXED["lora_dropout"],
        target_modules=FIXED["lora_target_modules"],
        # The classification head is randomly initialised and must therefore be
        # trained outright rather than through a low-rank update of noise.
        modules_to_save=["classifier"],
        bias="none",
    )
    return get_peft_model(base, cfg)


def _param_counts(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    adapted = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A.default"))
    return {"trainable_params": int(trainable), "total_params": int(total),
            "adapted_modules": int(adapted)}


def _trainable_snapshot(model) -> dict[str, Any]:
    return {n: p.detach().to("cpu").clone()
            for n, p in model.named_parameters() if p.requires_grad}


# ---------------------------------------------------------------------------
# Train / predict primitives
# ---------------------------------------------------------------------------

def _logits(model, loader, device: str) -> np.ndarray:
    import torch

    model.eval()
    out = []
    dtype = autocast_dtype(device)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            if dtype is not None:
                with torch.autocast(device_type=device, dtype=dtype):
                    logits = model(**batch).logits
            else:
                logits = model(**batch).logits
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def diagnose_fit(epoch_trace, selected_epoch, max_epochs, selection_macro_f1,
                 n_classes, majority_share=None):
    """Say whether a fine-tune ran out of budget, overfitted, or never learned.

    A fine-tune that runs for hours and then turns out to over- or underfit is
    thrown-away time. Best-epoch selection already protects the NUMBER -- an overfitting epoch 4
    never wins over epoch 2. What it does not do is tell anyone the run sat at
    the edge of its budget, and nobody reads an epoch trace by hand.

    So this turns the trace into a verdict that a human sees. It judges, it
    does not correct: changing max_epochs mid-campaign would make folds
    incomparable, which is worse than the thing it fixes.

    Returned `flags` are the ones that matter for the thesis:
      budget_binding    best epoch IS the ceiling AND early stopping never
                        fired -- the fit was still setting new bests after the
                        full budget and two patience windows. Under the fixed
                        cap this flag meant much less (see the code); since v1.7
                        it is a genuine finding. The honest fix is a larger
                        ceiling, decided once and applied to every fold.
      collapsed_early   best epoch is 1 and the score then falls -- the model
                        overfits within one pass, so lr or lora_r is too high.
      no_better_than_majority  the score never beats predicting the majority
                        class. On DBO's 0.80% subversive class this is the
                        failure that looks like success.
      degenerate        the score never moved at all -- nothing was learned.
    """
    flags, notes = [], []
    scores = [float(e["selection_macro_f1"]) for e in epoch_trace
              if "selection_macro_f1" in e] if epoch_trace else []

    # ⚠️ WHAT THIS FLAG MEANS CHANGED WITH v1.7, and it got much stronger.
    # Under the fixed cap it fired whenever the best epoch happened to be the
    # last one, which on a 5-epoch budget is neither rare nor alarming -- the
    # vio probe hit it immediately. Under early stopping the run only reaches
    # the ceiling by finding a NEW BEST there, having already survived
    # `patience` chances to plateau on the way. So the flag now says: this fit
    # was still improving after eight passes and two patience windows. That is
    # a finding about the model, not a budget accident, and it is the honest
    # replacement for a cap nobody could justify.
    if selected_epoch is not None and max_epochs and selected_epoch >= max_epochs:
        flags.append("budget_binding")
        notes.append(
            "best epoch = {} = the CEILING, and early stopping never triggered: "
            "the fit was still setting new bests after {} passes. Raise the "
            "ceiling in configs/e2_matrix.yaml deliberately, for ALL folds, and "
            "re-cost the arm.".format(selected_epoch, max_epochs))

    if len(scores) >= 2:
        if selected_epoch == 1 and scores[-1] < scores[0]:
            flags.append("collapsed_early")
            notes.append(
                "peaked at epoch 1 then fell ({:.4f} -> {:.4f}): overfits within "
                "one pass. lr or lora_r is too high.".format(scores[0], scores[-1]))
        if max(scores) - min(scores) < 1e-6:
            flags.append("degenerate")
            notes.append("the selection score never moved -- nothing was learned.")

    if majority_share is not None and selection_macro_f1 is not None:
        # A majority-only classifier scores this much macro-F1: it gets one
        # class right and every other class zero.
        floor = (2 * majority_share / (majority_share + 1)) / max(1, n_classes)
        if selection_macro_f1 <= floor + 1e-9:
            flags.append("no_better_than_majority")
            notes.append(
                "macro-F1 {:.4f} does not beat the majority-only floor {:.4f}."
                .format(selection_macro_f1, floor))

    return {"flags": flags, "notes": notes,
            "epoch_scores": scores,
            "selected_epoch": selected_epoch,
            "max_epochs": max_epochs,
            "ok": not flags}


def _batch_loss(model, batch, condition, class_weights, gamma):
    """The training loss of one batch under an E4a condition (src/imbalance.py).

    `none` and `random_oversampling` return the model's own loss untouched --
    oversampling changes WHICH items are in the batch, not how they are scored --
    so the E2 path is the same call it always was.
    """
    if condition in ("none", "random_oversampling"):
        return model(**batch).loss
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    return classification_loss(
        model(**inputs).logits, batch["labels"],
        weight=class_weights if condition == "class_weighting" else None,
        gamma=gamma if condition == "focal_loss" else 0.0)


def _fit_one(component, params, n_classes, fit_ids, y_fit, sel_ids, y_sel,
             pad_id, seed, device, max_epochs, patience=None,
             condition="none", class_weights=None, gamma=0.0):
    """
    Fine-tune one grid point, keeping the best epoch by selection macro-F1.

    v1.7: the epoch axis is closed by EARLY STOPPING ON THE SELECTION SLICE --
    training stops after `patience` consecutive epochs without a new best, and
    `max_epochs` is the ceiling. This is the `n_estimators` -> early stopping
    move of v1.1 applied to the third axis, and it makes the encoder consistent
    with `tml_protocol`, which has closed its length axis this way since v1.1
    (max_rounds 1000, patience 30). The encoder's fixed cap was the outlier in
    this project.

    ⚠️ WITHDRAWN in v1.8, kept here because the wrong version of it would have
    reached 3.3: this docstring claimed the change "cannot select a worse epoch
    than the fixed cap could -- the best epoch is restored either way, so
    widening the range can only add candidates". FALSE. The schedule below is
    built over `steps_per_epoch * max_epochs`, so a larger ceiling is a
    different learning-rate trajectory, not a longer one: the ceiling-8 fits
    are a different set of fits, not a superset of the ceiling-5 fits. What
    survives is the reason the change was made, and it is enough on its own --
    "we stopped at five" stops being an assertion. Since v1.9 the rule is also
    MEASURED rather than argued: `e2_runner.stopping_report` reads these traces
    back and says how often a stop happened while the trace was still climbing.

    Returns the model with the best epoch's parameters restored, the per-epoch
    selection trace, and how the run ended, so the CEILING can be checked for
    bindingness the way `best_iteration` is checked for the boosted trees.
    """
    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    model = _build_model(component, params, n_classes, seed).to(device)
    accum = max(1, FIXED["effective_batch_size"] // FIXED["per_device_batch_size"])
    train_loader = _make_loader(fit_ids, y_fit, pad_id,
                                FIXED["per_device_batch_size"], True, seed)
    sel_loader = _make_loader(sel_ids, None, pad_id, FIXED["eval_batch_size"], False, seed)

    steps_per_epoch = max(1, -(-len(train_loader) // accum))
    total_steps = steps_per_epoch * max_epochs
    optimiser = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=params["lr"], weight_decay=FIXED["weight_decay"],
        betas=FIXED["adam_betas"], eps=FIXED["adam_eps"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimiser, int(FIXED["warmup_ratio"] * total_steps), total_steps,
    )
    dtype = autocast_dtype(device)
    cw = (torch.tensor(class_weights, dtype=torch.float32, device=device)
          if class_weights is not None else None)

    if patience is None:
        patience = PATIENCE
    best = {"epoch": None, "score": -1.0, "state": None}
    trace = []
    since_best = 0
    stopped_early = False
    for epoch in range(1, max_epochs + 1):
        # PER-EPOCH TELEMETRY: if a paid run turns out to be unusable, it can
        # still be read as a long real test run rather than thrown away. The selection score alone cannot
        # tell "the model never learned" from "the model learned the wrong
        # thing" from "the machine was throttling". Four numbers can, and all
        # four are free -- they are quantities the training loop already has in
        # its hands.
        t_epoch = time.time()
        model.train()
        optimiser.zero_grad(set_to_none=True)
        loss_sum, loss_n, gnorm_sum, gnorm_n = 0.0, 0, 0.0, 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if dtype is not None:
                with torch.autocast(device_type=device, dtype=dtype):
                    loss = _batch_loss(model, batch, condition, cw, gamma)
            else:
                loss = _batch_loss(model, batch, condition, cw, gamma)
            (loss / accum).backward()
            loss_sum += float(loss.detach()); loss_n += 1
            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                # clip_grad_norm_ RETURNS the pre-clipping norm, so recording it
                # costs nothing and is the one number that separates "diverged"
                # from "never moved".
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    FIXED["max_grad_norm"],
                )
                gnorm_sum += float(gnorm); gnorm_n += 1
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)

        pred = _logits(model, sel_loader, device).argmax(axis=1)
        score = float(f1_score(y_sel, pred, average="macro", zero_division=0))
        trace.append({
            "epoch": epoch,
            "selection_macro_f1": score,
            # Did it train at all? A flat loss is a dead run; a rising one is a
            # diverging one. Neither is visible in macro-F1 until it is too late.
            "train_loss": round(loss_sum / max(1, loss_n), 6),
            # Did it diverge or stall? Pre-clip, so clipping cannot hide it.
            "grad_norm": round(gnorm_sum / max(1, gnorm_n), 4),
            # Did it collapse onto one class? The end-of-fit majority check fires
            # only once and only at the end; this shows the collapse happening.
            "pred_class_counts": np.bincount(pred, minlength=n_classes).tolist(),
            "lr": round(float(scheduler.get_last_lr()[0]), 9),
            # Is the machine behaving? An epoch that suddenly doubles is
            # throttling or contention, and it is invisible in any score.
            "seconds": round(time.time() - t_epoch, 1),
        })
        if score > best["score"]:
            best = {"epoch": epoch, "score": score, "state": _trainable_snapshot(model)}
            since_best = 0
        else:
            since_best += 1
            if patience and since_best >= patience:
                # The LR schedule was built for `max_epochs` steps and is now cut
                # short, which is the normal and intended consequence of early
                # stopping: the epochs that do not run are the ones that were not
                # improving. Nothing is restored from them, so the schedule's
                # unused tail cannot reach the selected model.
                stopped_early = True
                break

    model.load_state_dict(best["state"], strict=False)
    peak_gb = reserved_gb = None
    if device == "cuda":
        try:
            # BOTH, and the second is the one a headroom claim must use.
            # `max_memory_allocated` excludes the caching allocator's
            # reserved-but-unallocated pool: on one LLM cost measurement it read
            # 26.61 GB while nvidia-smi showed the card holding 41.85 GB of 48.
            # `peak_gpu_gb` keeps its meaning so earlier encoder artefacts stay
            # comparable; the reserved figure is added beside it.
            peak_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            reserved_gb = round(torch.cuda.max_memory_reserved() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            peak_gb = reserved_gb = None
    return model, best["epoch"], best["score"], trace, {
        "epochs_run": len(trace),
        # Up to three models can be alive at once inside the grid loop (the
        # incumbent, the loser, and the one being built). 48 GB has held it, but
        # only a measurement says by how much.
        "peak_gpu_gb": peak_gb,
        "peak_gpu_reserved_gb": reserved_gb,
        "stopped_early": stopped_early,
        "patience": patience,
        "ceiling": max_epochs,
        # The ceiling binding now means something much stronger than it did under
        # a fixed cap: the model was still finding new bests when the compute
        # guard stopped it, having already survived `patience` chances to plateau.
        "ceiling_bound": (not stopped_early) and best["epoch"] == max_epochs,
    }


# ---------------------------------------------------------------------------
# The fitted component
# ---------------------------------------------------------------------------

class FittedEncoderComponent:
    """An encoder component fitted on one fold training portion."""

    def __init__(self, component: str, classes: list, record: dict[str, Any]):
        self.component = component
        self.classes_ = list(classes)
        self.record = record
        self.model_ = None
        self.tokeniser_ = None
        self.device_ = "cpu"
        self.max_seq_len_ = MAX_SEQ_LEN
        self.temperature_: float = 1.0

    def _proba(self, df: pd.DataFrame) -> np.ndarray:
        from scipy.special import softmax

        ids, _ = _encode(self.tokeniser_, _texts(df), self.max_seq_len_)
        loader = _make_loader(ids, None, self.tokeniser_.pad_token_id,
                              FIXED["eval_batch_size"], False, 0)
        return softmax(_logits(self.model_, loader, self.device_), axis=1)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """(n_items, n_classes) in the global classes_ ordering, rows summing to 1."""
        return apply_temperature(self._proba(df), self.temperature_)


def _fit_component(component: str, train_df: pd.DataFrame, classes: list, *,
                   seed: int, device: str | None = None,
                   max_seq_len: int = MAX_SEQ_LEN,
                   max_epochs: int = MAX_EPOCHS,
                   grid: list[dict[str, Any]] | None = None,
                   condition: str = "none") -> FittedEncoderComponent:
    """Fit one encoder component on a fold training portion under the nested protocol.

    `condition` is an E4a imbalance condition; it touches the FIT slice only, so
    epoch selection and temperature scaling still see the natural distribution.
    """
    import torch
    from transformers import AutoTokenizer

    t0 = time.time()
    device = resolve_device(device)
    class_index = {c: i for i, c in enumerate(classes)}
    fit_df, sel_df, cal_df, stratified = _three_way_split(train_df, seed)

    tokeniser = AutoTokenizer.from_pretrained(MODEL_IDS[component])
    fit_ids, tok_stats = _encode(tokeniser, _texts(fit_df), max_seq_len)
    sel_ids, _ = _encode(tokeniser, _texts(sel_df), max_seq_len)
    cal_ids, _ = _encode(tokeniser, _texts(cal_df), max_seq_len)

    y_fit = np.array([class_index[v] for v in fit_df["label"].values])
    y_sel = np.array([class_index[v] for v in sel_df["label"].values])
    pad_id = tokeniser.pad_token_id

    # E4a. The NATURAL fit slice fixes the majority share the fit diagnosis
    # compares a collapse against, whatever the condition then trains on.
    majority_share = (float(np.bincount(y_fit).max()) / len(y_fit)
                      if len(y_fit) else None)
    class_weights, gamma = None, 0.0
    if condition == "class_weighting":
        class_weights = balanced_weights(y_fit, len(classes))
    elif condition == "focal_loss":
        from src.component_store import load_matrix
        gamma = focal_gamma(load_matrix())
    elif condition == "random_oversampling":
        idx = oversample_indices(y_fit, seed)
        fit_ids = [fit_ids[i] for i in idx]
        y_fit = y_fit[idx]
    elif condition != "none":
        raise ValueError("{} cannot train under {!r}".format(component, condition))

    selection: list[dict[str, Any]] = []
    best = None
    for params in (grid if grid is not None else GRIDS[component]):
        model, best_epoch, score, trace, stop_info = _fit_one(
            component, params, len(classes), fit_ids, y_fit, sel_ids, y_sel,
            pad_id, seed, device, max_epochs,
            condition=condition, class_weights=class_weights, gamma=gamma,
        )
        selection.append({"params": dict(params), "selection_macro_f1": score,
                          "best_epoch": best_epoch, "epoch_trace": trace,
                          "stopping": stop_info})
        superseded = None if best is None else best[2]
        if best is None or score > best[0]:
            best = (score, params, model, best_epoch)
        else:
            superseded = model
        del superseded          # frees whichever model did not survive
        if device == "cuda":
            torch.cuda.empty_cache()

    best_score, best_params, best_model, best_epoch = best
    fitted = FittedEncoderComponent(component, classes, {})
    fitted.model_ = best_model
    fitted.tokeniser_ = tokeniser
    fitted.device_ = device
    fitted.max_seq_len_ = max_seq_len

    # Temperature scaling on the calibration slice, which neither the grid nor
    # the epoch choice has seen.
    from scipy.special import softmax

    cal_loader = _make_loader(cal_ids, None, pad_id, FIXED["eval_batch_size"], False, seed)
    raw_cal = softmax(_logits(best_model, cal_loader, device), axis=1)
    temp = fit_temperature(raw_cal, list(cal_df["label"].values), classes)
    fitted.temperature_ = temp["temperature"]

    fitted.record = {
        "component": component,
        "grid_overridden": grid is not None,
        "model_id": MODEL_IDS[component],
        "selected_params": dict(best_params),
        "selection_macro_f1": best_score,
        "selected_epoch": best_epoch,
        "fit_diagnosis": diagnose_fit(
            next((c["epoch_trace"] for c in selection
                  if c["best_epoch"] == best_epoch
                  and c["selection_macro_f1"] == best_score), None),
            best_epoch, max_epochs, best_score, len(classes),
            majority_share=majority_share),
        "selection_grid": selection,
        "temperature": temp,
        "imbalance_condition": condition,
        "class_weights": (None if class_weights is None
                          else [round(float(w), 6) for w in class_weights]),
        "focal_gamma": gamma if condition == "focal_loss" else None,
        "n_fit": int(len(fit_df)),
        "n_fit_trained": int(len(y_fit)),
        "n_select": int(len(sel_df)),
        "n_calibrate": int(len(cal_df)),
        "classes_present_in_fit": [str(classes[i]) for i in sorted(set(y_fit.tolist()))],
        "inner_split_stratified": bool(stratified),
        "device": device,
        "precision": "bf16" if device == "cuda" else "fp32",
        "max_seq_len": int(max_seq_len),
        "max_epochs": int(max_epochs),          # the CEILING since v1.7
        "patience": int(PATIENCE),
        # Realised length of the winning fit. Under a fixed cap this was always
        # max_epochs and therefore not worth recording; under early stopping it
        # is the quantity that says what the arm actually cost and whether the
        # ceiling was anywhere near binding.
        "stopping": next((c["stopping"] for c in selection
                          if c["best_epoch"] == best_epoch
                          and c["selection_macro_f1"] == best_score), None),
        "epochs_run_per_cell": [c["stopping"]["epochs_run"] for c in selection],
        "effective_batch_size": int(FIXED["effective_batch_size"]),
        "per_device_batch_size": int(FIXED["per_device_batch_size"]),
        "gradient_accumulation_steps": int(
            max(1, FIXED["effective_batch_size"] // FIXED["per_device_batch_size"])),
        "lora": {"r": best_params["lora_r"], "alpha": 2 * best_params["lora_r"],
                 "dropout": FIXED["lora_dropout"],
                 "target_modules": list(FIXED["lora_target_modules"])},
        "tokenisation": tok_stats,
        "runtime_seconds": float(time.time() - t0),
        # Absolute wall-clock bounds, not just a duration. A duration cannot be
        # lined up against anything else that happened on the machine -- a GPU
        # trace, a watchdog message, a dropped ssh. If a
        # campaign has to be read back as a long test run, "which fold was
        # running at 03:12" is the question that gets asked first.
        "started_utc": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **_param_counts(best_model),
    }
    return fitted


# ---------------------------------------------------------------------------
# run_cv interface
# ---------------------------------------------------------------------------

def make_train_fn(component: str, classes: list, *, seed: int,
                  device: str | None = None, max_seq_len: int = MAX_SEQ_LEN,
                  max_epochs: int = MAX_EPOCHS,
                  grid: list[dict[str, Any]] | None = None,
                  condition: str = "none"):
    """Build a train_fn for harness.run_cv; everything nested happens inside it."""
    if component not in GRIDS:
        raise ValueError("Unknown encoder component: {!r}".format(component))

    def train_fn(train_df: pd.DataFrame) -> FittedEncoderComponent:
        return _fit_component(component, train_df, classes, seed=seed,
                              device=device, max_seq_len=max_seq_len,
                              max_epochs=max_epochs, grid=grid,
                              condition=condition)

    return train_fn


def predict_proba_fn(model: FittedEncoderComponent, df: pd.DataFrame, classes) -> np.ndarray:
    """predict_proba_fn for harness.run_cv; asserts the declared class ordering."""
    if list(classes) != list(model.classes_):
        raise ValueError(
            "classes_ ordering mismatch: run_cv passed {!r}, component holds {!r}.".format(
                list(classes), list(model.classes_)
            )
        )
    return model.predict_proba(df)
