"""
llm_components.py -- the generative arm, fine-tuned side (LLäMmlein 7B with LoRA).

Third component family, same protocol SHAPE as tml_components and
encoder_components: one three-way inner split per fold (fit / select /
calibrate), the grid resolved on the selection slice, the temperature fitted on
the calibration slice, and the fold's validation part touched by nothing but the
final prediction. What differs is only what a "fit" is and how a probability is
obtained.

WHAT A PROBABILITY IS HERE, AND WHY IT IS NOT WHAT 3.2 ORIGINALLY PROMISED
-------------------------------------------------------------------------
§3.2 promised that the probabilities the model "assigns to those label tokens
are read off and normalised". `src/verbaliser_probe.py` measured that a label is
NOT a token: on LLaeMmlein `Verherrlichung` -> `_Ver` + `herr` + `lichung` and
`Hetze` -> `_H` + `etze`, and the first tokens are generic German prefixes.
Reading P(`_Ver`) as P(glorification) charges that label with the mass of
*Verantwortung*, *Verbot*, *Versuch* -- i.e. it inflates the estimate on the
thinnest class in the study.

So each candidate label is scored as the **length-normalised log-likelihood of
its whole token sequence** under teacher forcing, and the label set is
softmax-normalised. That is independent of how any tokeniser splits a word and
returns exactly the (n_items, n_classes) row the interface specifies -- the contract is
unchanged, only its implementation is.

⚠️ Residual risk, stated rather than absorbed: surface-form competition
(`holtzman2021`). Likelihood still depends on the surface string. It is bounded
here -- the label set is small, fixed and identical for every item of a subtask,
so the competition acts as a constant per-class prior rather than per-item noise
-- and every component is temperature scaled on top. It is not eliminated.

WHERE THE CONSTANTS COME FROM
-----------------------------
From `e2_matrix.yaml`, read at import, never restated. `encoder_components`
keeps a `FIXED` dict beside the matrix and needs `tests/test_config_matches_matrix.py`
to hold the two together -- that duplication exists because it predates the
test and produced committed artefacts. There is no reason to repeat it in a
module written after the lesson.

Usage
-----
python -m src.llm_components --help
The runner drives it: python -m src.e2_runner --subtask dbo --component llm_llammlein
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.calibration import apply_temperature, fit_temperature
from src.component_store import load_matrix
from src.imbalance import balanced_weights, focal_gamma, oversample_indices, token_loss
from src.encoder_components import (diagnose_fit, resolve_device, _texts,
                                    _three_way_split, _trainable_snapshot)


# ⚠️ THE TEXT COMES THROUGH `encoder_components._texts`, WHICH IS
# `clean_encoder` -- the SAME profile the encoder arm uses, deliberately. The
# preprocessing module's contract is that differences between components reflect
# the MODELS rather than the text transformations, and exactly one flag may
# differ per component (`lowercase`, which is false for both). A generative arm
# that quietly saw raw text while the encoder saw cleaned text would make every
# arm-to-arm comparison in Chapter 4 confound the model with its input. The
# pool's column is `description`, not `text`; going through the shared helper is
# also what stops that from being rediscovered per module.
_MATRIX = load_matrix()
PROTOCOL = _MATRIX["llm_protocol"]
FIXED = dict(PROTOCOL["fixed_hyperparameters"])
EPOCHS = dict(PROTOCOL["epoch_selection"])

MAX_EPOCHS = int(EPOCHS["max_epochs"])          # v1.10: a CEILING, not the rule
PATIENCE = int(EPOCHS["patience"])
MAX_SEQ_LEN = int(FIXED["max_seq_len"])
VERBALISERS = _MATRIX["verbalisers"]

LLM_COMPONENTS = tuple(
    name for name, block in _MATRIX["components"].items()
    if block.get("family") == "llm_finetune"
)
MODEL_IDS = {name: _MATRIX["components"][name]["hf_id"] for name in LLM_COMPONENTS}


def _grid_from_matrix(component: str) -> list[dict[str, Any]]:
    """The declared grid, expanded to a list of cells.

    `lora_r` is deliberately NOT an axis (matrix v1.10): the encoder arm could
    not separate r=8 from r=16 -- selection is flagged `selection_noise` on all
    three subtasks -- so sweeping it on a ~7x more expensive backbone would
    re-ask a question this evaluation has been shown unable to answer.
    """
    grid = _MATRIX["components"][component]["grid"]
    if not isinstance(grid, dict):
        raise ValueError("{}: expected a grid mapping, got {!r}".format(component, grid))
    axes = sorted(grid)
    cells: list[dict[str, Any]] = [{}]
    for axis in axes:
        cells = [dict(c, **{axis: v}) for c in cells for v in grid[axis]]
    return cells


GRIDS = {c: _grid_from_matrix(c) for c in LLM_COMPONENTS}


# ---------------------------------------------------------------------------
# Verbalisers and the prompt
# ---------------------------------------------------------------------------
#
# One verbaliser set per subtask, German, and the SAME set for both LLM
# components: Dimension 2 requires that only the label set differ between
# subtasks, not between components. The organisers' English string is
# printed beside its German word in the prompt so the mapping an examiner reads
# is the mapping the model is scored against.

_VERBALISER_KEY = {"c2a": "c2a", "dbo": "dbo", "vio": "vio25"}

_INSTRUCTION = {
    "c2a": "Entscheide, ob der folgende Beitrag zu einer Handlung aufruft.",
    "dbo": "Ordne den folgenden Beitrag genau einer der vier Kategorien zu.",
    "vio": "Entscheide, ob der folgende Beitrag Gewalt aufruft oder befuerwortet.",
}


def verbalisers_for(subtask: str, edition: str = "2025") -> dict[str, str]:
    """class label -> German verbaliser, in no particular order.

    The 2026 six-way VIO scheme is a different label set and lives under
    `vio26`; it is reached only through E4c, never through E2.
    """
    key = _VERBALISER_KEY.get(subtask.lower())
    if key is None:
        raise ValueError("no verbaliser set declared for subtask {!r}".format(subtask))
    if subtask.lower() == "vio" and str(edition) == "2026":
        key = "vio26"
    return {str(k): str(v) for k, v in VERBALISERS[key].items()}


def build_prompt(text: str, subtask: str, classes: list) -> str:
    """The zero-shot instruction template. Fixed per subtask, reused across
    editions (3.2 subsec:llm), so a cross-edition difference can never be an
    artefact of a rewritten prompt.

    NO in-context examples: those belong to llm_fewshot_protocol and to a
    different component. Mixing them would make the two arms differ on two
    things at once.
    """
    verb = verbalisers_for(subtask)
    block = "\n".join("- {} ({})".format(verb[str(c)], c) for c in classes)
    return (
        "{}\n\nMoegliche Antworten:\n{}\n\nBeitrag: {}\n\nAntwort:".format(
            _INSTRUCTION[subtask.lower()], block, text)
    )


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _label_sequences(tokeniser, subtask: str, classes: list) -> list[list[int]]:
    """Token ids of each verbaliser, in the global classes_ order.

    A leading space is included so the continuation is tokenised the way it
    would appear after "Antwort:" -- scoring `Ja` where the model would emit
    ` Ja` measures a different string than the one the prompt asks for.
    """
    verb = verbalisers_for(subtask)
    out = []
    for c in classes:
        ids = tokeniser(" " + verb[str(c)], add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError("verbaliser for class {!r} tokenises to nothing".format(c))
        out.append(list(ids))
    return out


def _encode_prompts(tokeniser, texts: list[str], subtask: str, classes: list,
                    max_seq_len: int, longest_label: int, render=None,
                    slice_name: str | None = None):
    """Prompt token ids, with the ITEM TEXT truncated rather than the prompt.

    ⚠️ Truncating the assembled prompt from the right would cut the question,
    and from the left would cut the label block -- either way the model is asked
    something other than the protocol says. So the budget is spent on the text:
    the template is measured first, and the text is cut to whatever remains.
    The share of items that had to be cut is recorded, because it is a property
    of the protocol rather than of the data (llm_protocol.truncation).

    ⚠️ The recorded share is a property of ONE SLICE, and the slice is named in
    the stats rather than left to be inferred. `llm_protocol.truncation` used to
    say "the share of prompts exceeding max_seq_len", which is a quantity that
    is ZERO by construction here -- no prompt can exceed it, because the text is
    what gets cut. What is counted is the share of item TEXTS that had to be
    cut, recorded for the fit slice and, separately, for the validation slice
    that every reported probability comes from.

    `render` is the prompt shape, defaulting to the zero-shot template. The
    few-shot component passes its own, which carries the in-context block --
    the truncation rule is identical and belongs in ONE place: it is subtle
    (measure the skeleton, spend what is left on the text, count the cuts), and
    a second copy would drift.
    """
    render = render or (lambda body: build_prompt(body, subtask, classes))
    skeleton = render("")
    overhead = len(tokeniser(skeleton, add_special_tokens=True)["input_ids"])
    budget = max_seq_len - longest_label - overhead
    if budget <= 0:
        raise ValueError(
            "the {} template plus the longest verbaliser already fills "
            "max_seq_len={}; no room for any text".format(subtask, max_seq_len))

    ids, truncated = [], 0
    for text in texts:
        toks = tokeniser(str(text), add_special_tokens=False)["input_ids"]
        if len(toks) > budget:
            toks = toks[:budget]
            truncated += 1
        cut = tokeniser.decode(toks)
        ids.append(tokeniser(render(cut), add_special_tokens=True)["input_ids"])
    lengths = [len(x) for x in ids]
    stats = {
        "slice": slice_name,
        "n_items": len(ids),
        "text_truncated": int(truncated),
        "text_truncated_share": round(truncated / max(1, len(ids)), 4),
        "text_token_budget": int(budget),
        "template_overhead_tokens": int(overhead),
        "prompt_tokens_max": int(max(lengths)) if lengths else 0,
        "prompt_tokens_mean": round(float(np.mean(lengths)), 1) if lengths else 0.0,
    }
    return ids, stats


# ---------------------------------------------------------------------------
# Sequence scoring -- the probability interface
# ---------------------------------------------------------------------------

# How much GPU the logits of ONE forward pass may take. NOT a protocol value and
# deliberately not in the matrix: it changes no probability, and a matrix entry
# would move config_id for a decision about a graphics card. Measured
# need: Qwen2.5-14B has a 152 064 vocabulary, so ONE 2 048-token
# sequence's logits are 468 MB in bf16 -- the lm_head OOM'd asking for 7.53 GB
# with 7.19 GB free, after the fp32 copies had already been removed. LLaeMmlein's
# 32 000-vocabulary at 512 tokens is 33 MB, so the fine-tuned arm never chunks.
LOGIT_BUDGET_BYTES = 1 << 30          # 1 GiB


def _chunk_size(width: int, vocab: int, budget: int | None = None) -> int:
    """How many sequences of the given padded WIDTH fit the logit budget.

    Pulled out of `_score_batch` so a probe measuring this arithmetic against
    real prompts calls the one formula that
    actually runs, instead of a second copy of it.

    `budget` defaults to the CURRENT module-level `LOGIT_BUDGET_BYTES`, read at
    call time rather than bound as a default-argument value -- a default of
    `LOGIT_BUDGET_BYTES` would freeze the value from import time, and
    `tests/test_llm_cell.py` monkeypatches the module attribute to sweep chunk
    sizes without a real 15 GB tensor.
    """
    if budget is None:
        budget = LOGIT_BUDGET_BYTES
    per_seq_bytes = max(1, width * vocab * 2)
    return max(1, budget // per_seq_bytes)


def _score_batch(model, prompt_ids: list[list[int]], label_ids: list[list[int]],
                 pad_id: int, device: str):
    """Length-normalised log-likelihood of every label for every prompt.

    Returns (n_prompts, n_labels). One forward pass per (prompt, label) pair:
    the label tokens have to be attended to in order, so they cannot share a
    pass. Sequences are RIGHT-padded, which is correct under causal attention --
    a position never attends to a later one, so trailing pads cannot reach the
    logits being read.
    """
    import torch

    pairs = [(i, j) for i in range(len(prompt_ids)) for j in range(len(label_ids))]
    seqs = [prompt_ids[i] + label_ids[j] for i, j in pairs]
    width = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for k, s in enumerate(seqs):
        inp[k, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[k, :len(s)] = 1

    # ⚠️ ONLY THE POSITIONS THAT ARE READ ARE MATERIALISED IN fp32, and that is
    # not an optimisation -- it is what makes the few-shot arm runnable at all.
    # The obvious version of this function upcasts the
    # WHOLE logits tensor and log_softmaxes it: for Qwen2.5-14B that is
    # n_seq x width x 152 064 floats, so a 16-sequence batch of 2 048-token
    # few-shot prompts asks CUDA for 15 GB and dies -- after which it reads
    # three scalars per sequence. The fine-tuned arm never hit it because its
    # MAX_SEQ_LEN is 512, four times shorter, which is why a smoke test on
    # that arm cannot see it.
    #
    # log_softmax normalises over the VOCAB dimension, independently per
    # position, so restricting to the read positions before normalising is
    # arithmetically the same operation and not an approximation.
    # `tests/test_llm_cell.py` asserts the two agree exactly.
    width_needed = max(len(label_ids[j]) for _, j in pairs)
    pos = torch.zeros((len(seqs), width_needed), dtype=torch.long)
    tok = torch.zeros((len(seqs), width_needed), dtype=torch.long)
    keep = torch.zeros((len(seqs), width_needed), dtype=torch.bool)
    for k, (i, j) in enumerate(pairs):
        start, lab = len(prompt_ids[i]), label_ids[j]
        for t in range(len(lab)):
            # Token t of the label is predicted by the logits at position t-1.
            pos[k, t], tok[k, t], keep[k, t] = start + t - 1, lab[t], True

    # ⚠️ AND THE FORWARD PASS IS SPLIT SO ITS LOGITS FIT (measured).
    # Restricting to the read positions killed the two fp32 copies but not the
    # tensor the MODEL returns: `lm_head` still materialises n_seq x width x
    # vocab in bf16, which for Qwen2.5-14B is 7.53 GB against the 7.19 GB left
    # beside its own weights -- a second OOM, in `F.linear`, on a card the first
    # fix was supposed to have made room on. Chunking is arithmetically free:
    # each sub-batch is an independent forward pass, and `test_llm_cell` already
    # asserts a score does not depend on its batch neighbours' padding.
    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if not vocab:
        vocab = int(model.get_output_embeddings().out_features)
    chunk = _chunk_size(width, vocab)

    picked_parts = []
    with torch.no_grad():
        for lo in range(0, len(seqs), chunk):
            hi = min(lo + chunk, len(seqs))
            logits = model(input_ids=inp[lo:hi].to(device),
                           attention_mask=mask[lo:hi].to(device)).logits
            rows = torch.arange(hi - lo, device=logits.device).unsqueeze(1)
            selected = logits[rows, pos[lo:hi].to(logits.device)].float()
            logprobs = torch.log_softmax(selected, dim=-1)
            picked_parts.append(logprobs.gather(
                2, tok[lo:hi].to(logprobs.device).unsqueeze(-1)).squeeze(-1).cpu())
            del logits, selected, logprobs
    picked = torch.cat(picked_parts, dim=0)
    picked = picked.masked_fill(~keep, 0.0).double().numpy()

    out = np.zeros((len(prompt_ids), len(label_ids)), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        lab = label_ids[j]
        out[i, j] = float(picked[k].sum()) / len(lab)   # length normalisation
    return out


def _label_proba(model, prompt_ids: list[list[int]], label_ids: list[list[int]],
                 pad_id: int, device: str, batch_items: int) -> np.ndarray:
    """(n_items, n_classes), rows summing to 1, from the scores above."""
    from scipy.special import softmax

    chunks = []
    for s in range(0, len(prompt_ids), batch_items):
        chunks.append(_score_batch(model, prompt_ids[s:s + batch_items],
                                   label_ids, pad_id, device))
    scores = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(label_ids)))
    return softmax(scores, axis=1)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def _build_model(component: str, params: dict, seed: int, device: str):
    """A LoRA-adapted causal LM.

    No `modules_to_save`: unlike the encoder there is no randomly initialised
    head to train outright. The LM head already exists and is exactly what emits
    the verbaliser, which is the point of the sequence-scoring interface.

    Precision: bf16 on CUDA is a WEIGHT dtype here, not autocast. A 7B in fp32
    is 27 GB of VRAM before any activation; loading in bf16 halves it and is
    what `llm_protocol.precision` declares. The adapter parameters are kept in
    fp32, because that is the tensor the optimiser updates and bf16's 8-bit
    mantissa loses small updates outright.
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(MODEL_IDS[component], dtype=dtype)
    base.config.use_cache = False           # incompatible with gradient flow
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(FIXED["lora_r"]),
        lora_alpha=int(FIXED["lora_alpha"]),
        lora_dropout=float(FIXED["lora_dropout"]),
        target_modules=list(FIXED["lora_target_modules"]),
        bias="none",
    )
    model = get_peft_model(base, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    return model


def _param_counts(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    # ⚠️ COUNTED, NOT ASSUMED. PEFT matches target modules by NAME SUFFIX,
    # and a bare name can select more than it reads as -- on ModernBERT `Wo`
    # named both the attention and the MLP output projection, so the DECLARED
    # protocol described a model nobody ran. The Llama names below should not
    # collide (the MLP is gate/up/down_proj), but "should not" is exactly what
    # was disproved there, so the number is recorded in every artefact.
    names = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    # ⚠️ THE COUNT IS NOT THE CHECK. Mutating the targets to
    # ["gate_proj", "up_proj"] -- the MLP instead of attention -- leaves every
    # count-based check green, because `num_hidden_layers x len(target_modules)`
    # predicts the same number for any equinumerous target list. So the
    # SUFFIXES that were actually adapted are recorded in every artefact.
    suffixes = sorted({n.rsplit(".lora_A", 1)[0].split(".")[-1] for n in names})
    return {"trainable_params": int(trainable), "total_params": int(total),
            "adapted_modules": len(names),
            "adapted_suffixes": suffixes,
            "adapted_in_mlp": sorted(n for n in names if ".mlp." in n)[:4]}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _training_batches(prompt_ids, gold_label_ids, pad_id, batch_size, seed, shuffle=True,
                      item_class=None):
    """Right-padded batches whose loss is masked to the verbaliser tokens.

    `labels` is -100 everywhere except the label positions, which is how
    `llm_protocol.objective` is implemented: causal-LM cross-entropy on the
    verbaliser tokens only, prompt tokens masked out of the loss. HuggingFace
    shifts labels internally, so the ids are placed at their own positions.
    """
    import torch

    order = np.arange(len(prompt_ids))
    if shuffle:
        np.random.RandomState(seed).shuffle(order)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        seqs = [prompt_ids[i] + gold_label_ids[i] for i in idx]
        width = max(len(x) for x in seqs)
        inp = torch.full((len(idx), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        lab = torch.full((len(idx), width), -100, dtype=torch.long)
        for k, i in enumerate(idx):
            seq = prompt_ids[i] + gold_label_ids[i]
            inp[k, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[k, :len(seq)] = 1
            start = len(prompt_ids[i])
            lab[k, start:len(seq)] = torch.tensor(gold_label_ids[i], dtype=torch.long)
        batch = {"input_ids": inp, "attention_mask": mask, "labels": lab}
        # E4a class weighting needs each item's class to weight its label
        # tokens. It rides along only when asked for, and it is popped before
        # the forward pass; the shuffle above never sees it, so the batch ORDER
        # is identical with and without it.
        if item_class is not None:
            batch["item_class"] = torch.tensor([int(item_class[i]) for i in idx],
                                               dtype=torch.long)
        yield batch


def _load_trainable(model, state: dict | None) -> None:
    """Load a trainable-parameter snapshot, and REFUSE a partial load.

    `load_state_dict(..., strict=False)` treats an absent key as nothing at
    all, so a restore that quietly does no work looks exactly like one that
    worked -- and the model that then predicts is the LAST epoch's rather than
    the best one's, which no output would look odd about. A mutation test that
    removed this restore left every suite green, so it is asserted here instead
    of hoped for.
    """
    import torch

    if not state:
        raise ValueError("nothing to restore: the snapshot is empty, so the "
                         "best epoch was never captured")
    # `missing` is every FROZEN parameter and is expected: the snapshot is the
    # trainable set by construction. `unexpected` is the one that matters --
    # it is the shape a typo'd or stale key takes.
    _missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(
            "the snapshot carries {} parameter(s) this model does not have "
            "({}...) -- the restore would have been silently partial".format(
                len(unexpected), sorted(unexpected)[:3]))
    live = dict(model.named_parameters())
    for name, tensor in state.items():
        if name not in live:
            raise ValueError(
                "snapshot key {!r} is not a parameter of this model".format(name))
        if not torch.equal(live[name].detach().cpu(), tensor.detach().cpu()):
            raise ValueError(
                "parameter {!r} did not take the snapshot's value".format(name))


def _batch_loss(model, batch, condition, class_weights, gamma):
    """The training loss of one batch under an E4a condition (src/imbalance.py).

    `none` and `random_oversampling` return the model's own masked loss
    untouched, so the E2 path is the same call it always was. The other two
    recompute it on the label positions only (`imbalance.token_loss`), which is
    where llm_protocol.objective says the objective-level E4a conditions apply.
    """
    item_class = batch.pop("item_class", None)
    if condition in ("none", "random_oversampling"):
        return model(**batch).loss
    labels = batch.pop("labels")
    logits = model(**batch).logits
    return token_loss(
        logits, labels,
        item_weight=(class_weights[item_class]
                     if condition == "class_weighting" else None),
        gamma=gamma if condition == "focal_loss" else 0.0)


def _fit_one(component, params, fit_prompts, fit_gold, sel_prompts, y_sel,
             label_ids, pad_id, seed, device, max_epochs, n_classes,
             patience: int | None = None, condition: str = "none",
             class_weights=None, gamma: float = 0.0, y_fit=None):
    """Fine-tune one grid cell, keeping the best epoch by selection macro-F1.

    Identical stopping rule to the encoder since matrix v1.10: `max_epochs` is a
    CEILING and training stops after `patience` consecutive epochs with no new
    best. The per-epoch telemetry is the same four quantities, for the same
    reason -- if a paid campaign turns out unusable it should still be readable
    as a long real test run, and the selection score alone cannot tell "never
    learned" from "learned the wrong thing" from "the machine was throttling".
    """
    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    model = _build_model(component, params, seed, device).to(device)
    per_device = int(FIXED["per_device_batch_size"])
    accum = max(1, int(FIXED["effective_batch_size"]) // per_device)
    n_batches = -(-len(fit_prompts) // per_device)
    steps_per_epoch = max(1, -(-n_batches // accum))
    total_steps = steps_per_epoch * max_epochs

    optimiser = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=float(params["lr"]),
                      weight_decay=float(FIXED["weight_decay"]),
                      betas=tuple(FIXED["adam_betas"]), eps=float(FIXED["adam_eps"]))
    scheduler = get_linear_schedule_with_warmup(
        optimiser, int(float(FIXED["warmup_ratio"]) * total_steps), total_steps)

    if patience is None:
        patience = PATIENCE
    best = {"epoch": None, "score": -1.0, "state": None}
    trace, since_best, stopped_early = [], 0, False
    cw = (torch.tensor(class_weights, dtype=torch.float32, device=device)
          if class_weights is not None else None)
    if condition == "class_weighting" and y_fit is None:
        raise ValueError("class weighting needs y_fit to weight each item's tokens")

    for epoch in range(1, max_epochs + 1):
        t_epoch = time.time()
        model.train()
        optimiser.zero_grad(set_to_none=True)
        loss_sum = loss_n = 0
        loss_sum = 0.0
        gnorm_sum, gnorm_n = 0.0, 0
        for step, batch in enumerate(_training_batches(
                fit_prompts, fit_gold, pad_id, per_device, seed + epoch,
                item_class=y_fit if condition == "class_weighting" else None)):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = _batch_loss(model, batch, condition, cw, gamma)
            (loss / accum).backward()
            loss_sum += float(loss.detach()); loss_n += 1
            if (step + 1) % accum == 0 or (step + 1) == n_batches:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(FIXED["max_grad_norm"]))
                gnorm_sum += float(gnorm); gnorm_n += 1
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)

        model.eval()
        proba = _label_proba(model, sel_prompts, label_ids, pad_id, device,
                             max(1, int(FIXED["eval_batch_size"]) // max(1, n_classes)))
        pred = proba.argmax(axis=1)
        score = float(f1_score(y_sel, pred, average="macro", zero_division=0))
        trace.append({
            "epoch": epoch,
            "selection_macro_f1": score,
            "train_loss": round(loss_sum / max(1, loss_n), 6),
            "grad_norm": round(gnorm_sum / max(1, gnorm_n), 4),
            "pred_class_counts": np.bincount(pred, minlength=n_classes).tolist(),
            "lr": round(float(scheduler.get_last_lr()[0]), 9),
            "seconds": round(time.time() - t_epoch, 1),
        })
        if score > best["score"]:
            best = {"epoch": epoch, "score": score, "state": _trainable_snapshot(model)}
            since_best = 0
        else:
            since_best += 1
            if patience and since_best >= patience:
                stopped_early = True
                break

    _load_trainable(model, best["state"])
    peak_gb = reserved_gb = None
    if device == "cuda":
        try:
            # BOTH, and the second is the one a headroom claim must use.
            # `max_memory_allocated` excludes the caching allocator's
            # reserved-but-unallocated pool: on one cost measurement it read
            # 26.61 GB while nvidia-smi showed the card holding 41.85 GB of 48.
            # `peak_gpu_gb` keeps its meaning so the encoder artefacts stay
            # comparable; the reserved figure is added beside it.
            peak_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            reserved_gb = round(torch.cuda.max_memory_reserved() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            peak_gb = reserved_gb = None
    return model, best["epoch"], best["score"], trace, {
        "epochs_run": len(trace),
        "peak_gpu_gb": peak_gb,
        "peak_gpu_reserved_gb": reserved_gb,
        "stopped_early": stopped_early,
        "patience": patience,
        "ceiling": max_epochs,
        "ceiling_bound": (not stopped_early) and best["epoch"] == max_epochs,
    }


# ---------------------------------------------------------------------------
# The fitted component
# ---------------------------------------------------------------------------

class FittedLLMComponent:
    """An LLM component fitted on one fold training portion."""

    def __init__(self, component: str, classes: list, subtask: str,
                 record: dict[str, Any]):
        self.component = component
        self.classes_ = list(classes)
        self.subtask = subtask
        self.record = record
        self.model_ = None
        self.tokeniser_ = None
        self.label_ids_ = None
        self.device_ = "cpu"
        self.max_seq_len_ = MAX_SEQ_LEN
        self.temperature_: float = 1.0

    def _proba(self, df: pd.DataFrame) -> np.ndarray:
        longest = max(len(x) for x in self.label_ids_)
        prompts, stats = _encode_prompts(self.tokeniser_, _texts(df),
                                         self.subtask, self.classes_,
                                         self.max_seq_len_, longest,
                                         slice_name="validation")
        # The truncation figure for the slice that actually produces every
        # reported probability. `tokenisation` above describes the FIT slice,
        # which is not the same items. run_cv calls on_fold
        # AFTER predict_proba_fn, so this reaches the fold record.
        self.record["tokenisation_predict"] = stats
        self.model_.eval()          # lora_dropout is 0.05; a model left in
                                    # train mode makes predict_proba stochastic
        n_classes = max(1, len(self.classes_))
        return _label_proba(self.model_, prompts, self.label_ids_,
                            self.tokeniser_.pad_token_id, self.device_,
                            max(1, int(FIXED["eval_batch_size"]) // n_classes))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """(n_items, n_classes) in the global classes_ ordering, rows summing to 1."""
        return apply_temperature(self._proba(df), self.temperature_)


def _prepare_tokeniser(model_id: str):
    """Load the tokeniser and make sure it can pad.

    Llama-family tokenisers ship no pad token. Falling back to EOS is the
    standard move and is safe here because padding is always on the RIGHT and
    every padded position is masked out of both attention and the loss -- a pad
    token can therefore never be scored or trained on.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        raise ValueError(
            "{} has neither a pad nor an eos token, so batches cannot be "
            "padded".format(model_id))
    return tok


def _fit_component(component: str, train_df: pd.DataFrame, classes: list, *,
                   subtask: str, seed: int, device: str | None = None,
                   max_seq_len: int = MAX_SEQ_LEN,
                   max_epochs: int = MAX_EPOCHS,
                   grid: list[dict[str, Any]] | None = None,
                   condition: str = "none") -> FittedLLMComponent:
    """Fit one LLM component on a fold training portion under the nested protocol.

    `condition` is an E4a imbalance condition; it touches the FIT slice only, so
    epoch selection and temperature scaling still see the natural distribution.
    """
    import torch

    t0 = time.time()
    device = resolve_device(device)
    class_index = {str(c): i for i, c in enumerate(classes)}
    fit_df, sel_df, cal_df, stratified = _three_way_split(train_df, seed)

    tokeniser = _prepare_tokeniser(MODEL_IDS[component])
    label_ids = _label_sequences(tokeniser, subtask, classes)
    longest = max(len(x) for x in label_ids)

    fit_prompts, tok_stats = _encode_prompts(
        tokeniser, _texts(fit_df), subtask, classes,
        max_seq_len, longest, slice_name="fit")
    sel_prompts, _ = _encode_prompts(
        tokeniser, _texts(sel_df), subtask, classes,
        max_seq_len, longest)
    cal_prompts, _ = _encode_prompts(
        tokeniser, _texts(cal_df), subtask, classes,
        max_seq_len, longest)

    y_fit = np.array([class_index[str(v)] for v in fit_df["label"].values])
    y_sel = np.array([class_index[str(v)] for v in sel_df["label"].values])
    pad_id = tokeniser.pad_token_id

    # E4a. The NATURAL fit slice fixes the majority share the fit diagnosis
    # compares a collapse against, whatever the condition then trains on.
    majority_share = (float(np.bincount(y_fit).max()) / len(y_fit)
                      if len(y_fit) else None)
    class_weights, gamma = None, 0.0
    if condition == "class_weighting":
        class_weights = balanced_weights(y_fit, len(classes))
    elif condition == "focal_loss":
        gamma = focal_gamma(load_matrix())
    elif condition == "random_oversampling":
        idx = oversample_indices(y_fit, seed)
        fit_prompts = [fit_prompts[i] for i in idx]
        y_fit = y_fit[idx]
    elif condition != "none":
        raise ValueError("{} cannot train under {!r}".format(component, condition))
    fit_gold = [label_ids[i] for i in y_fit]

    # ⚠️ ONE BASE MODEL AT A TIME, AND THE WINNER IS CARRIED AS ITS ADAPTER
    # STATE RATHER THAN AS A LIVE MODEL. Measured by
    # counting live LlamaForCausalLM objects on this exact path: holding the
    # incumbent MODEL across the grid put THREE base models in memory at once
    # on the campaign shape -- the previous fold's (held by run_cv while it
    # builds the next), the incumbent grid cell's, and the challenger being
    # built. On LLaeMmlein 7B in bf16 that is ~40.5 GB of a 48 GB A40 before
    # activations and before the ~2 GB fp32 logit tensor the scoring pass
    # allocates. The adapter state is ~67 MB of LoRA tensors on the CPU, and
    # the surviving model is the LAST cell's -- its base weights are frozen and
    # byte-identical to every other cell's, so loading the winner's adapter
    # into it MAKES it the winner and costs no second `from_pretrained`.
    # Peak is now one model per grid cell and two across folds.
    selection: list[dict[str, Any]] = []
    best = None
    model = None
    for params in (grid if grid is not None else GRIDS[component]):
        model = None                       # free the previous cell BEFORE the
        if device == "cuda":               # next one is built, not after
            torch.cuda.empty_cache()
        model, cell_epoch, score, trace, stop_info = _fit_one(
            component, params, fit_prompts, fit_gold, sel_prompts, y_sel,
            label_ids, pad_id, seed, device, max_epochs, len(classes),
            condition=condition, class_weights=class_weights, gamma=gamma,
            y_fit=y_fit)
        selection.append({"params": dict(params), "selection_macro_f1": score,
                          "best_epoch": cell_epoch, "epoch_trace": trace,
                          "stopping": stop_info})
        if best is None or score > best[0]:
            best = (score, params, _trainable_snapshot(model), cell_epoch)

    best_score, best_params, best_state, best_epoch = best
    best_model = model
    _load_trainable(best_model, best_state)
    best_model.eval()
    fitted = FittedLLMComponent(component, classes, subtask, {})
    fitted.model_ = best_model
    fitted.tokeniser_ = tokeniser
    fitted.label_ids_ = label_ids
    fitted.device_ = device
    fitted.max_seq_len_ = max_seq_len

    raw_cal = _label_proba(best_model, cal_prompts, label_ids, pad_id, device,
                           max(1, int(FIXED["eval_batch_size"]) // max(1, len(classes))))
    temp = fit_temperature(raw_cal, [str(v) for v in cal_df["label"].values],
                           [str(c) for c in classes])
    fitted.temperature_ = temp["temperature"]

    winner = next((c for c in selection
                   if c["best_epoch"] == best_epoch
                   and c["selection_macro_f1"] == best_score), None)
    fitted.record = {
        "component": component,
        "grid_overridden": grid is not None,
        "model_id": MODEL_IDS[component],
        "subtask": subtask,
        "selected_params": dict(best_params),
        "selection_macro_f1": best_score,
        "selected_epoch": best_epoch,
        "fit_diagnosis": diagnose_fit(
            winner["epoch_trace"] if winner else None,
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
        "max_epochs": int(max_epochs),
        "patience": int(PATIENCE),
        "stopping": winner["stopping"] if winner else None,
        "epochs_run_per_cell": [c["stopping"]["epochs_run"] for c in selection],
        "effective_batch_size": int(FIXED["effective_batch_size"]),
        "per_device_batch_size": int(FIXED["per_device_batch_size"]),
        "gradient_accumulation_steps": max(
            1, int(FIXED["effective_batch_size"]) // int(FIXED["per_device_batch_size"])),
        "lora": {"r": int(FIXED["lora_r"]), "alpha": int(FIXED["lora_alpha"]),
                 "dropout": float(FIXED["lora_dropout"]),
                 "target_modules": list(FIXED["lora_target_modules"])},
        "tokenisation": tok_stats,
        "verbalisers": verbalisers_for(subtask),
        "verbaliser_token_lengths": [len(x) for x in label_ids],
        "probability_interface": "verbaliser_sequence_score",
        "runtime_seconds": float(time.time() - t0),
        "started_utc": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **_param_counts(best_model),
    }
    return fitted


# ---------------------------------------------------------------------------
# run_cv interface
# ---------------------------------------------------------------------------

def make_train_fn(component: str, classes: list, *, subtask: str, seed: int,
                  device: str | None = None, max_seq_len: int = MAX_SEQ_LEN,
                  max_epochs: int = MAX_EPOCHS,
                  grid: list[dict[str, Any]] | None = None,
                  condition: str = "none"):
    """Build a train_fn for harness.run_cv; everything nested happens inside it."""
    if component not in GRIDS:
        raise ValueError("Unknown LLM component: {!r}".format(component))

    def train_fn(train_df: pd.DataFrame) -> FittedLLMComponent:
        return _fit_component(component, train_df, classes, subtask=subtask,
                              seed=seed, device=device, max_seq_len=max_seq_len,
                              max_epochs=max_epochs, grid=grid,
                              condition=condition)

    return train_fn


def predict_proba_fn(model: FittedLLMComponent, df: pd.DataFrame, classes) -> np.ndarray:
    """predict_proba_fn for harness.run_cv; asserts the declared class ordering."""
    if [str(c) for c in classes] != [str(c) for c in model.classes_]:
        raise ValueError(
            "classes_ ordering mismatch: run_cv passed {!r}, component holds {!r}.".format(
                list(classes), list(model.classes_)))
    return model.predict_proba(df)


def _usage() -> str:
    lines = [
        "llm_components.py -- the fine-tuned generative arm.",
        "",
        "Not a CLI: the runner drives it.",
        "    python -m src.e2_runner --subtask dbo --component llm_llammlein",
        "    python -m src.e2_runner --arm llm            (all subtasks, both LLM components)",
        "",
        "Declared protocol, read from configs/e2_matrix.yaml at import:",
    ]
    for c in LLM_COMPONENTS:
        lines.append("  {:<16s} {}  grid {}".format(c, MODEL_IDS[c], GRIDS[c]))
    lines += [
        "  ceiling {} epochs, patience {} (v1.10)".format(MAX_EPOCHS, PATIENCE),
        "  max_seq_len {}, lora_r {}, target {}".format(
            MAX_SEQ_LEN, FIXED["lora_r"], FIXED["lora_target_modules"]),
        "  probability: length-normalised full-sequence log-likelihood,",
        "               softmax over the label set",
    ]
    for sub in ("c2a", "dbo", "vio"):
        lines.append("  verbalisers {:<4s} {}".format(sub, verbalisers_for(sub)))
    return "\n".join(lines)


if __name__ == "__main__":
    print(_usage())
