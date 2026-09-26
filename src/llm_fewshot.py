"""
llm_fewshot.py -- the generative arm's SECOND half: in-context, no training.

Qwen 2.5-14B prompted with twelve labelled examples. `--arm llm` of the runner
drives both LLM components.

WHAT IS DIFFERENT FROM llm_components, AND WHAT IS DELIBERATELY IDENTICAL
------------------------------------------------------------------------
Different: nothing is trained. There is no grid, no epoch, no LoRA adapter and
no optimiser -- `llm_fewshot_protocol.training` is `none`, and the component's
whole configuration is *which twelve examples go in the prompt*. So a "fit" here
is a draw plus a temperature.

Identical, and on purpose: the probability interface (each label scored as the
length-normalised log-likelihood of its full token sequence, softmax over the
label set), the verbaliser sets, the text profile (`clean_encoder`, through
the shared `_texts`), the three-way inner split, and the calibration slice. Those
are imported from `llm_components` rather than restated -- a second copy of the
truncation rule or the scoring loop would drift and silently make the two LLM
rows of Chapter 4 incomparable in ways no artefact records.

THE SELECTION SLICE HAS NO JOB HERE, AND THAT IS RECORDED RATHER THAN REUSED
---------------------------------------------------------------------------
The other three families spend the inner split as fit / select / calibrate.
Nothing is selected here, so the middle slice does no work. Two options, and the
choice matters for Chapter 4:

  (a) drop to a two-way split and give the examples a larger pool;
  (b) keep the identical three-way split and leave the selection slice unused.

**(b), and the reason is comparability rather than tidiness.** The temperature is
fitted on the calibrate slice and the ECE that
`llm_fewshot_protocol.preflight_before_b6` demands is read off it; if this
component calibrated on a slice built differently from every other component's,
its ECE would not be comparable with theirs, which is the one number the
combination layer's probability interface rests on. The examples cost twelve rows, so the
larger pool option buys nothing measurable. `n_unused_selection` is written into
every fold record so the cost is stated instead of hidden.

THE DRAW IS THE VARIANCE, NOT THE TRAINING SEED
-----------------------------------------------
`llm_fewshot_protocol.draws: 3` replaces the seed replicates: there is no
training to reseed, so what varies between replicates is which examples were
drawn. The runner's replicate machinery is reused unchanged -- seed 42 is the
primary draw, 43 and 44 are the replicates on fold 0 -- so the variance column
of Chapter 4 is built the same way for this component as for the encoder, and
`seed_semantics` in the artefact says what the seed actually moved.

THE ONE IMBALANCE LEVER
-----------------------
`imbalance_applicability` gives this family exactly one strategy,
`prompt_coverage`, and E2 is the reference cell (`none`). Both are implemented
here because the contrast is the whole point of the E4a row: `none` draws
proportionally to the natural distribution, which at DBO `subversive` = 0.8 %
almost never contains one -- that is the reference condition, not a defect --
while `prompt_coverage` draws equal-per-class at the SAME example count and adds
the rare-class definitions to the label block. Count held constant, composition
varied: otherwise the cell would measure budget rather than coverage.

Usage
-----
python -m src.llm_fewshot --help
The runner drives it: python -m src.e2_runner --subtask dbo --component llm_qwen_fewshot
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.calibration import apply_temperature, fit_temperature
from src.component_store import load_matrix
from src.encoder_components import resolve_device, _texts, _three_way_split
from src.llm_components import (_encode_prompts, _label_proba, _label_sequences,
                                _prepare_tokeniser, _INSTRUCTION, verbalisers_for)

_MATRIX = load_matrix()
PROTOCOL = _MATRIX["llm_fewshot_protocol"]

N_EXAMPLES = int(PROTOCOL["n_examples"])
DRAWS = int(PROTOCOL["draws"])
MAX_PROMPT_TOKENS = int(PROTOCOL["max_prompt_tokens"])
# The scoring batch is the fine-tuned arm's, for the same reason the inner
# split is: one difference between the two LLM components, not three.
EVAL_BATCH = int(_MATRIX["llm_protocol"]["fixed_hyperparameters"]["eval_batch_size"])

FEWSHOT_COMPONENTS = tuple(
    name for name, block in _MATRIX["components"].items()
    if block.get("family") == "llm_fewshot"
)
MODEL_IDS = {name: _MATRIX["components"][name]["hf_id"] for name in FEWSHOT_COMPONENTS}

STRATEGIES = tuple(PROTOCOL["example_selection"])       # ("none", "prompt_coverage")

# The rare-class definitions prompt_coverage adds to the label block. German,
# and drawn from the organisers' own annotation guidance rather than invented
# here -- an invented definition would make the E4a contrast a comparison between
# our paraphrase and nothing, instead of between coverage and no coverage.
_CLASS_DEFINITIONS = {
    "dbo": {
        "subversive": "untergraebt die demokratische Ordnung indirekt, ohne "
                      "offenen Aufruf -- etwa durch Delegitimierung von "
                      "Institutionen, Wahlen oder Medien",
        "extremist": "vertritt eine verfassungsfeindliche Position offen",
        "opposing": "widerspricht der demokratischen Ordnung in Teilen, bleibt "
                    "aber innerhalb des zulaessigen Meinungsspektrums",
    },
}


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------

def select_examples(train_df: pd.DataFrame, classes: list, *,
                    n_examples: int = N_EXAMPLES, strategy: str = "none",
                    seed: int = 42) -> pd.DataFrame:
    """The in-context examples, drawn from the fold training portion ONLY.

    `example_source` in the matrix is explicit that this never sees the
    validation part and never another fold. That property is not enforced here
    because it cannot be -- this function is handed a frame and cannot know what
    it is. It is enforced by `run_cv`, which hands `train_fn` the training
    portion and nothing else, and asserted in tests/test_llm_fewshot_cell.py
    against the real splitter.

    `none`         -- proportional to the natural class distribution. At DBO
                      subversive = 0.8 % a twelve-example draw almost never
                      contains one. That IS the reference condition.
    `prompt_coverage` -- equal per class, remainder to the largest classes, at
                      the identical count. Composition varies, budget does not.
    """
    if strategy not in STRATEGIES:
        raise ValueError("unknown example_selection {!r}; the matrix declares "
                         "{}".format(strategy, list(STRATEGIES)))
    if n_examples <= 0:
        raise ValueError("n_examples must be positive, got {}".format(n_examples))
    if len(train_df) < n_examples:
        raise ValueError(
            "the fold training portion holds {} rows and the prompt needs {} "
            "examples".format(len(train_df), n_examples))

    rng = np.random.default_rng(seed)
    labels = train_df["label"].astype(str)

    if strategy == "none":
        idx = rng.choice(len(train_df), size=n_examples, replace=False)
        return train_df.iloc[np.sort(idx)].copy().reset_index(drop=True)

    # prompt_coverage: equal per class first, then the remainder to the classes
    # that still have rows, largest first. A class absent from this fold's
    # training portion contributes nothing and is NOT invented -- the draw comes
    # from real rows or it is not a few-shot example.
    per = {str(c): int((labels == str(c)).sum()) for c in classes}
    present = [str(c) for c in classes if per[str(c)] > 0]
    if not present:
        raise ValueError("no class of {} occurs in the training portion".format(classes))

    base, extra = divmod(n_examples, len(present))
    want = {c: min(base, per[c]) for c in present}
    # hand out the remainder, and whatever the per-class ceilings left over,
    # to the classes with rows to spare -- largest first, deterministically.
    order = sorted(present, key=lambda c: (-per[c], c))
    short = n_examples - sum(want.values())
    while short > 0:
        moved = False
        for c in order:
            if short == 0:
                break
            if want[c] < per[c]:
                want[c] += 1
                short -= 1
                moved = True
        if not moved:                      # every class exhausted
            break
    del extra

    picks = []
    for c in order:
        if want[c] == 0:
            continue
        pool = np.flatnonzero((labels == c).values)
        picks.extend(rng.choice(pool, size=want[c], replace=False).tolist())
    return train_df.iloc[np.sort(np.array(picks))].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

def build_fewshot_prompt(text: str, subtask: str, classes: list,
                         examples: pd.DataFrame, *,
                         definitions: bool = False) -> str:
    """Instruction, label block, in-context examples, then the item.

    The shape mirrors `llm_components.build_prompt` exactly -- same instruction,
    same `verbaliser (organiser label)` block, same `Beitrag:` / `Antwort:`
    turns -- so the only difference between the two LLM components' prompts is
    the presence of the examples. Dimension 2 requires that a cross-component
    difference be attributable to the component; a second rewritten template
    would attribute it to the prompt instead.
    """
    sub = subtask.lower()
    verb = verbalisers_for(subtask)
    defs = _CLASS_DEFINITIONS.get(sub, {}) if definitions else {}

    lines = []
    for c in classes:
        c = str(c)
        entry = "- {} ({})".format(verb[c], c)
        if c in defs:
            entry += ": {}".format(defs[c])
        lines.append(entry)
    block = "\n".join(lines)

    shots = []
    for _, row in examples.iterrows():
        shots.append("Beitrag: {}\n\nAntwort: {}".format(
            _texts(pd.DataFrame([row]))[0], verb[str(row["label"])]))
    shot_block = ("\n\n".join(shots) + "\n\n") if shots else ""

    return "{}\n\nMoegliche Antworten:\n{}\n\n{}Beitrag: {}\n\nAntwort:".format(
        _INSTRUCTION[sub], block, shot_block, text)


# ---------------------------------------------------------------------------
# The component
# ---------------------------------------------------------------------------

class FittedFewShotComponent:
    """A few-shot component "fitted" on one fold training portion.

    Fitted in the interface's sense only: a draw and a temperature. No weight of the
    model was touched, which is why `trainable_params` is 0 in every record and
    why the artefact says `training: none` rather than leaving it to be inferred.
    """

    def __init__(self, component: str, classes: list, subtask: str,
                 record: dict[str, Any]):
        self.component = component
        self.classes_ = list(classes)
        self.subtask = subtask
        self.record = record
        self.model_ = None
        self.tokeniser_ = None
        self.label_ids_ = None
        self.examples_ = None
        self.definitions_ = False
        self.device_ = "cpu"
        self.max_prompt_tokens_ = MAX_PROMPT_TOKENS
        self.temperature_: float = 1.0

    def _render(self, body: str) -> str:
        return build_fewshot_prompt(body, self.subtask, self.classes_,
                                    self.examples_, definitions=self.definitions_)

    def _proba(self, df: pd.DataFrame) -> np.ndarray:
        longest = max(len(x) for x in self.label_ids_)
        prompts, stats = _encode_prompts(self.tokeniser_, _texts(df), self.subtask,
                                         self.classes_, self.max_prompt_tokens_,
                                         longest, render=self._render,
                                         slice_name="validation")
        # The truncation figure for the slice that produces every reported
        # probability. `tokenisation` in the record describes the CALIBRATE
        # slice -- a different, and much smaller, set of items.
        self.record["tokenisation_predict"] = stats
        n_classes = max(1, len(self.classes_))
        return _label_proba(self.model_, prompts, self.label_ids_,
                            self.tokeniser_.pad_token_id, self.device_,
                            max(1, int(EVAL_BATCH) // n_classes))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """(n_items, n_classes) in the global classes_ ordering, rows summing to 1."""
        return apply_temperature(self._proba(df), self.temperature_)



# ONE resident model per process, and exactly one. Measured on the first real
# few-shot cell (dbo / llm_qwen_fewshot): fold 0 scored
# fine at 34.4 GB, then `_fit_component` was called for the next fold, loaded a
# SECOND 34.4 GB copy beside the first, and died in `Module._apply` --
# "Tried to allocate 10.00 MiB ... 44.09 GiB is allocated by PyTorch" on a 44.43
# GiB A40. A cell is 5 folds + 2 replicates, so it asked for seven copies.
#
# Why no earlier check could see it: a cost measurement runs ONE fold, and the
# CPU suites run a ~1M-parameter random Llama where seven copies are free.
#
# Reuse is not an optimisation here, it is what the protocol already says: this arm
# touches NO weight (see FittedFewShotComponent's docstring, and `trainable_params`
# is 0 in every record), so the model is read-only and identical across folds.
# Nothing about a probability changes; `eval()` and `requires_grad_(False)` are
# idempotent and `_label_proba` runs under `torch.no_grad()`.
#
# The slot is SINGLE and evicts on a different key, because `--arm llm` runs six
# cells in one process across TWO components -- LLaeMmlein 27.0 GB plus Qwen
# 29.5 GB would reproduce the same OOM one level up.
#
# The key carries the resolved model ID, not the component name: the CPU suites
# point `MODEL_IDS[...]` at a tiny local directory, and a name-keyed slot would
# hand them a stale model and quietly pass.
_RESIDENT: dict[str, Any] = {}


def _evict_resident() -> None:
    """Drop the resident model and give the memory back to the allocator."""
    import gc

    model = _RESIDENT.pop("model", None)
    _RESIDENT.pop("key", None)
    if model is None:
        return
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_model(component: str, device: str):
    """The base model, in eval mode. No adapter, no optimiser, no gradients.

    bf16 on CUDA is what `llm_fewshot_protocol.serving` declares, rather than a
    Q5_K_M quantisation -- so this arm's confidences reach the combination layer
    through the same precision as the fine-tuned arm's, and the calibration
    difference between the two rows is the model rather than the quantisation.

    Returns the RESIDENT model when one matching this key is already loaded --
    see the note above `_RESIDENT`.
    """
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    key = "{}|{}|{}".format(MODEL_IDS[component], device, dtype)
    if _RESIDENT.get("key") == key and _RESIDENT.get("model") is not None:
        return _RESIDENT["model"]

    _evict_resident()
    model = AutoModelForCausalLM.from_pretrained(MODEL_IDS[component], dtype=dtype)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _RESIDENT["key"] = key
    _RESIDENT["model"] = model
    return model


def _fit_component(component: str, train_df: pd.DataFrame, classes: list, *,
                   subtask: str, seed: int, device: str | None = None,
                   max_prompt_tokens: int = MAX_PROMPT_TOKENS,
                   n_examples: int = N_EXAMPLES,
                   strategy: str = "none") -> FittedFewShotComponent:
    """Draw the examples, load the model, fit the temperature. Nothing trains."""
    t0 = time.time()
    device = resolve_device(device)
    fit_df, sel_df, cal_df, stratified = _three_way_split(train_df, seed)

    examples = select_examples(fit_df, classes, n_examples=n_examples,
                               strategy=strategy, seed=seed)

    tokeniser = _prepare_tokeniser(MODEL_IDS[component])
    label_ids = _label_sequences(tokeniser, subtask, classes)
    longest = max(len(x) for x in label_ids)

    fitted = FittedFewShotComponent(component, classes, subtask, {})
    fitted.tokeniser_ = tokeniser
    fitted.label_ids_ = label_ids
    fitted.examples_ = examples
    fitted.definitions_ = (strategy == "prompt_coverage")
    fitted.device_ = device
    fitted.max_prompt_tokens_ = max_prompt_tokens
    fitted.model_ = _load_model(component, device)

    # The prompt is measured on the calibration slice while it is being encoded
    # anyway. The example block eats the budget the item text would otherwise
    # get, so the truncated share here is a different number from the fine-tuned
    # arm's and Chapter 4 has to be able to say so.
    cal_prompts, tok_stats = _encode_prompts(
        tokeniser, _texts(cal_df), subtask, classes, max_prompt_tokens,
        longest, render=fitted._render, slice_name="calibrate")
    raw_cal = _label_proba(fitted.model_, cal_prompts, label_ids,
                           tokeniser.pad_token_id, device,
                           max(1, int(EVAL_BATCH) // max(1, len(classes))))
    temp = fit_temperature(raw_cal, [str(v) for v in cal_df["label"].values],
                           [str(c) for c in classes])
    fitted.temperature_ = temp["temperature"]

    ex_labels = [str(v) for v in examples["label"].values]
    example_tokens = len(tokeniser(fitted._render(""),
                                   add_special_tokens=True)["input_ids"])
    fitted.record = {
        "component": component,
        "model_id": MODEL_IDS[component],
        "subtask": subtask,
        "training": "none",
        # Stated as null rather than omitted. `run_component` reads
        # `selected_params` off every fold record, and the aggregation that
        # feeds `diagnose_selection` needs the key to exist -- but the reason
        # to write it is not the KeyError: an artefact that simply lacks the
        # field leaves a reader to guess whether nothing was selected or the
        # selection was lost. Nothing is selected here, and it says so.
        "selected_params": None,
        "selected_epoch": None,
        "selection_grid": [],
        "example_selection": strategy,
        "n_examples": int(len(examples)),
        "example_class_counts": {str(c): int(ex_labels.count(str(c)))
                                 for c in classes},
        "example_ids": [str(v) for v in examples.get("id", pd.Series(dtype=str)).values],
        "example_source": "the fit part of this fold's training portion only",
        "rare_class_definitions_in_prompt": bool(fitted.definitions_),
        "prompt_skeleton_tokens": int(example_tokens),
        "temperature": temp,
        "n_fit": int(len(fit_df)),
        "n_unused_selection": int(len(sel_df)),
        "n_calibrate": int(len(cal_df)),
        "inner_split_stratified": bool(stratified),
        "selection_slice_note": (
            "kept and left unused: nothing is selected for this component, and "
            "an identical calibration slice is what makes its ECE comparable "
            "with the other three families' (llm_fewshot module docstring)"),
        "classes_present_in_examples": sorted(set(ex_labels)),
        "device": device,
        "precision": "bf16" if device == "cuda" else "fp32",
        "max_prompt_tokens": int(max_prompt_tokens),
        "tokenisation": tok_stats,
        "verbalisers": verbalisers_for(subtask),
        "verbaliser_token_lengths": [len(x) for x in label_ids],
        "probability_interface": "verbaliser_sequence_score",
        "trainable_params": 0,
        "adapted_modules": 0,
        "runtime_seconds": float(time.time() - t0),
        "started_utc": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return fitted


# ---------------------------------------------------------------------------
# run_cv interface
# ---------------------------------------------------------------------------

def make_train_fn(component: str, classes: list, *, subtask: str, seed: int,
                  device: str | None = None,
                  max_prompt_tokens: int = MAX_PROMPT_TOKENS,
                  n_examples: int = N_EXAMPLES, strategy: str = "none"):
    """Build a train_fn for harness.run_cv.

    The seed reaches both the inner split and the draw, which is what makes a
    replicate a different DRAW rather than a different training run
    (`llm_fewshot_protocol.draws: 3`).
    """
    if component not in MODEL_IDS:
        raise ValueError(
            "Unknown few-shot component: {!r}. The matrix declares {}".format(
                component, list(FEWSHOT_COMPONENTS)))

    def train_fn(train_df: pd.DataFrame) -> FittedFewShotComponent:
        return _fit_component(component, train_df, classes, subtask=subtask,
                              seed=seed, device=device,
                              max_prompt_tokens=max_prompt_tokens,
                              n_examples=n_examples, strategy=strategy)

    return train_fn


def predict_proba_fn(model: FittedFewShotComponent, df: pd.DataFrame,
                     classes) -> np.ndarray:
    """predict_proba_fn for harness.run_cv; asserts the declared class ordering."""
    if [str(c) for c in classes] != [str(c) for c in model.classes_]:
        raise ValueError(
            "classes_ ordering mismatch: run_cv passed {!r}, component holds {!r}.".format(
                list(classes), list(model.classes_)))
    return model.predict_proba(df)


def _usage() -> str:
    lines = [
        "llm_fewshot.py -- the in-context generative arm.",
        "",
        "Not a CLI: the runner drives it.",
        "    python -m src.e2_runner --subtask dbo --component llm_qwen_fewshot",
        "",
        "Declared protocol, read from configs/e2_matrix.yaml at import:",
    ]
    for c in FEWSHOT_COMPONENTS:
        lines.append("  {:<18s} {}".format(c, MODEL_IDS[c]))
    lines += [
        "  training none; {} examples per prompt; {} draws (variance)".format(
            N_EXAMPLES, DRAWS),
        "  max_prompt_tokens {}, eval batch {} (item,label) pairs".format(
            MAX_PROMPT_TOKENS, EVAL_BATCH),
        "  example_selection: {}".format(", ".join(STRATEGIES)),
        "  probability: length-normalised full-sequence log-likelihood,",
        "               softmax over the label set, then temperature",
    ]
    for sub in ("c2a", "dbo", "vio"):
        lines.append("  verbalisers {:<4s} {}".format(sub, verbalisers_for(sub)))
    return "\n".join(lines)


if __name__ == "__main__":
    print(_usage())
