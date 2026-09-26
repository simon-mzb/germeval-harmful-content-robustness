"""
e2_runner.py -- driver for the E2 component experiments, and for E4a/E4b.

Runs one component family against one subtask/edition under the protocol fixed
in `configs/e2_matrix.yaml`, and writes
two artefacts per run: a results JSON with everything the thesis reports, and a
component-store entry with the pooled out-of-fold probabilities that E3 will
consume without retraining.

Protocol, in one place so it can be checked against the matrix
--------------------------------------------------------------
* Pool: the training split of the given edition, after the contradiction rule
  and within-edition deduplication. The train-test overlap rule is deliberately
  **off**: it exists to protect evaluation on the official test set, which E2
  never touches, and switching it on would discard **123** labelled DBO pool
  rows including **37 of the 224** `agitation` instances. This also makes the
  pool identical to the one E1 measured the canonical baselines on, and
  reproduces the documented class counts exactly. (Both figures describe the
  POOL, after the contradiction rule, not the raw train split.)
* Repeats: three stratified 5-fold partitions (seeds 42, 43, 44). For the
  classical models the estimators are deterministic, so a seed that only
  reseeded the model would measure nothing; the seed varies the partition,
  which is what three seeds mean for a CPU family.
* Aggregation: pooled out-of-fold **within** a repeat, so every item is scored
  exactly once. The point estimate is the mean over the three repeats;
  fold-to-fold and seed-to-seed spread are reported separately.
* Bootstrap CIs are computed on the primary repeat (seed 42). Bootstrapping a
  vector that contained each item three times would understate the interval.

Usage
-----
python -m src.e2_runner --arm tml
python -m src.e2_runner --subtask dbo --edition 2025 --component tml_xgboost
python -m src.e2_runner --arm tml --dry-run
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.component_store import (ComponentStore, compute_config_id,
                                 compute_data_rules_id, run_key)
from src.data_loading import load_split
from src import e4_config
from src import encoder_components as enc
from src import imbalance
from src import llm_components as llm
from src import llm_fewshot as fewshot
from src import tml_components as tml
from src.harness import (
    apply_data_quality_rules,
    env_pin,
    evaluate_predictions,
    load_fold_checkpoint,
    make_cv_splits,
    pooled_oof_report,
    resolve_classes,
    run_cv,
    save_fold_checkpoint,
    save_results,
)
from src.tml_components import FEATURE_SPEC, GRIDS, INNER_SPLIT, TML_COMPONENTS

_ROOT = Path(__file__).parent.parent
MATRIX_PATH = _ROOT / "configs" / "e2_matrix.yaml"
RESULTS_DIR = _ROOT / "results" / "e2"
# Fold checkpoints. A cell that finishes deletes its own; what survives a run
# is therefore exactly the work an interruption left unfinished.
CKPT_ROOT = RESULTS_DIR / ".checkpoints"

SEEDS = (42, 43, 44)
PRIMARY_SEED = 42

# The protocol gives the two families different run shapes, and the difference is not
# cosmetic. The classical estimators are deterministic, so a seed that only
# reseeded the model would measure nothing and the seed varies the *partition*
# instead. The fine-tune is stochastic, so there the partition is held fixed
# and the seed varies *training*, which is what seeds buy there: an
# estimate of training variance. Encoder replicates therefore run on one fixed
# fold of the primary partition and are reported as a dispersion statistic,
# never pooled into the point estimate.
ENCODER_REPLICATE_SEEDS = (43, 44)
ENCODER_FIXED_FOLD = 0


def family_of(component: str) -> str:
    if component in TML_COMPONENTS:
        return "tml"
    if component in enc.ENCODER_COMPONENTS:
        return "encoder"
    if component in llm.LLM_COMPONENTS:
        return "llm_finetune"
    if component in fewshot.FEWSHOT_COMPONENTS:
        return "llm_fewshot"
    raise ValueError("Unknown component: {!r}".format(component))


def component_api(component: str, *, n_jobs: int = -1, device: str | None = None,
                  max_epochs: int | None = None,
                  subtask: str | None = None,
                  condition: str = "none",
                  grid: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Everything run_component needs to stay agnostic about the family.

    `subtask` is required for the LLM family and ignored by the other two: the
    prompt template and the verbaliser set are per subtask, so an LLM train_fn
    cannot be built without knowing which one it is. It is a parameter rather
    than something the component looks up, because run_component already knows
    it and a second lookup is a second place to be wrong.

    `condition` and `grid` carry an E4 cell (an imbalance condition and the
    frozen one-cell configuration); both default to the E2 path."""
    fam = family_of(component)
    if fam == "llm_fewshot" and (condition != "none" or grid is not None):
        raise ValueError(
            "{} trains nothing, so it has no loss- or data-level condition and no "
            "grid to fix; the protocol's only lever for it is prompt_coverage, which is a "
            "descriptive cell this runner does not implement".format(component))
    if fam == "tml":
        return {
            "family": fam,
            "seeds": SEEDS,
            "replicate_seeds": (),
            "make_train_fn": lambda classes, seed: tml.make_train_fn(
                component, classes, seed=seed, n_jobs=n_jobs,
                condition=condition, grid=grid),
            "predict_proba_fn": tml.predict_proba_fn,
            "meta": {
                "feature_spec": dict(FEATURE_SPEC),
                "inner_split": dict(INNER_SPLIT),
                "grid": GRIDS[component],
                "grid_deviation": (
                    "n_estimators replaced by early stopping on the selection "
                    "slice (cap 1000, patience 30); see THESIS_LOG session B3"),
                "seed_semantics": "the seed varies the CV partition",
            },
        }
    if fam == "llm_fewshot":
        if not subtask:
            raise ValueError(
                "component_api({!r}) needs subtask=: the prompt template, the "
                "verbaliser set and the example draw are per subtask "
                "(llm_fewshot_protocol.prompt)".format(component))
        # Variance for this family is `draws: 3`, not three training seeds -- nothing
        # trains. The replicate machinery is reused unchanged so the variance
        # column is built the same way as the encoder's; what the seed moves is
        # recorded in seed_semantics rather than left to be inferred.
        return {
            "family": fam,
            "seeds": (PRIMARY_SEED,),
            "replicate_seeds": ENCODER_REPLICATE_SEEDS,
            "make_train_fn": lambda classes, seed: fewshot.make_train_fn(
                component, classes, subtask=subtask, seed=seed, device=device),
            "predict_proba_fn": fewshot.predict_proba_fn,
            "meta": {
                "model_id": fewshot.MODEL_IDS[component],
                "inner_split": dict(INNER_SPLIT),
                "grid": None,
                "grid_deviation": (
                    "no grid: nothing is tuned. The configuration is the draw, "
                    "and `draws: {}` replaces D3's seed replicates".format(
                        fewshot.DRAWS)),
                "training": "none",
                "n_examples": fewshot.N_EXAMPLES,
                "example_selection": "none (D4 reference; prompt_coverage is the E4a cell)",
                "max_prompt_tokens": fewshot.MAX_PROMPT_TOKENS,
                "verbalisers": llm.verbalisers_for(subtask),
                "probability_interface": "verbaliser_sequence_score",
                "seed_semantics": (
                    "the partition is fixed at the primary seed; the seed varies "
                    "the EXAMPLE DRAW, and replicates run on one fold only (D3)"),
            },
        }
    if fam == "llm_finetune":
        if not subtask:
            raise ValueError(
                "component_api({!r}) needs subtask=: the prompt template and the "
                "verbaliser set are per subtask (llm_protocol.prompt)".format(component))
        # The fine-tuned LLM gets the SAME variance treatment as the encoder
        # -- one partition seed, two training-seed replicates on fold 0 -- so the
        # replicate constants are shared rather than restated. A second copy is
        # a second thing to drift.
        return {
            "family": fam,
            "seeds": (PRIMARY_SEED,),
            "replicate_seeds": ENCODER_REPLICATE_SEEDS,
            "make_train_fn": lambda classes, seed: llm.make_train_fn(
                component, classes, subtask=subtask, seed=seed, device=device,
                max_epochs=max_epochs or llm.MAX_EPOCHS,
                grid=grid, condition=condition),
            "predict_proba_fn": llm.predict_proba_fn,
            "meta": {
                "model_id": llm.MODEL_IDS[component],
                "inner_split": dict(INNER_SPLIT),
                "grid": llm.GRIDS[component],
                "fixed_hyperparameters": {
                    k: (list(v) if isinstance(v, (list, tuple)) else v)
                    for k, v in llm.FIXED.items()},
                "max_seq_len": llm.MAX_SEQ_LEN,
                "max_epochs": max_epochs or llm.MAX_EPOCHS,
                "patience": llm.PATIENCE,
                "verbalisers": llm.verbalisers_for(subtask),
                "probability_interface": "verbaliser_sequence_score",
                "grid_deviation": (
                    "lora_r is FIXED at {} rather than swept, decided against the "
                    "completed encoder arm (matrix v1.10, THESIS_LOG #100d); the "
                    "epoch axis is a ceiling of {} closed by early stopping at "
                    "patience {}, not a fixed cap. {} grid cells on lr.".format(
                        llm.FIXED["lora_r"], max_epochs or llm.MAX_EPOCHS,
                        llm.PATIENCE, len(llm.GRIDS[component]))),
                "seed_semantics": (
                    "the partition is fixed at the primary seed; the seed varies "
                    "training, and replicates run on one fold only (D3)"),
            },
        }

    return {
        "family": fam,
        "seeds": (PRIMARY_SEED,),
        "replicate_seeds": ENCODER_REPLICATE_SEEDS,
        "make_train_fn": lambda classes, seed: enc.make_train_fn(
            component, classes, seed=seed, device=device,
            max_epochs=max_epochs or enc.MAX_EPOCHS,
            grid=grid, condition=condition),
        "predict_proba_fn": enc.predict_proba_fn,
        "meta": {
            "model_id": enc.MODEL_IDS[component],
            "inner_split": dict(INNER_SPLIT),
            "grid": enc.GRIDS[component],
            "tied": {"lora_alpha": "2 * lora_r"},
            "fixed_hyperparameters": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                                      for k, v in enc.FIXED.items()},
            "max_seq_len": enc.MAX_SEQ_LEN,
            "max_epochs": max_epochs or enc.MAX_EPOCHS,
            # ⚠️ THE CELL COUNT IS DERIVED, NOT SPELLED OUT: a number about the
            # matrix maintained by hand beside the matrix that decides it goes
            # stale the first time the grid changes.
            "grid_deviation": (
                "epochs replaced by best-epoch selection on the selection slice "
                "(cap {}); lora_alpha tied to 2*lora_r. The v1.1 16-cell grid was "
                "re-cut to 4 in v1.2 and widened to {} in the current matrix; see "
                "THESIS_LOG Next Steps #40 and e2_matrix.yaml".format(
                    max_epochs or enc.MAX_EPOCHS, len(enc.GRIDS[component]))),
            "seed_semantics": (
                "the partition is fixed at the primary seed; the seed varies "
                "training, and replicates run on one fold only (D3)"),
        },
    }


def run_seed_replicate(df, component: str, classes: list, *, api: dict,
                       model_seed: int, partition_seed: int = PRIMARY_SEED,
                       fold: int = ENCODER_FIXED_FOLD,
                       n_splits: int = 5,
                       ckpt_dir=None, fingerprint: str | None = None) -> dict[str, Any]:
    """One fold of the primary partition, retrained under a different seed.

    Checkpointed on the same terms as the folds in `run_cv`, because on the
    encoder each replicate is another full fine-tune. NOTE the trap this had
    to avoid: a replicate uses the SAME fold index and the SAME validation
    indices as the main run's fold, so nothing in the checkpoint's own
    consistency checks would catch a mix-up. The model seed therefore goes into
    both the directory name and the fingerprint."""
    splits = list(make_cv_splits(df, n_splits=n_splits, seed=partition_seed))
    train_idx, val_idx = splits[fold]
    val_fold = df.iloc[val_idx].copy().reset_index(drop=True)

    restored = None
    if ckpt_dir is not None:
        restored = load_fold_checkpoint(
            ckpt_dir, fold, fingerprint=fingerprint, seed=partition_seed,
            n_splits=n_splits, classes=classes, val_idx=val_idx)

    t0 = time.time()
    if restored is not None:
        proba, payload = restored
        record = payload or {}
        runtime = float(record.get("runtime_seconds", 0.0))
        resumed = True
    else:
        train_fold = df.iloc[train_idx].copy().reset_index(drop=True)
        train_fold, _ = apply_data_quality_rules(
            train_fold, val_fold, dedup=True, drop_overlap=True)
        model = api["make_train_fn"](classes, model_seed)(train_fold)
        proba = np.asarray(api["predict_proba_fn"](model, val_fold, classes), dtype=np.float64)
        runtime = float(time.time() - t0)
        record = {
            "selected_params": model.record.get("selected_params"),
            "selected_epoch": model.record.get("selected_epoch"),
            "fit_diagnosis": model.record.get("fit_diagnosis"),
            # The cost diagnostics, lifted here so they reach BOTH the checkpoint
            # payload and the replicate record -- see the note at the return.
            **{k: model.record[k] for k in ("tokenisation", "tokenisation_predict",
                                            "prompt_skeleton_tokens",
                                            "max_prompt_tokens",
                                            "verbaliser_token_lengths")
               if k in model.record},
            "runtime_seconds": runtime,
        }
        resumed = False
        if ckpt_dir is not None:
            save_fold_checkpoint(
                ckpt_dir, fold, fingerprint=fingerprint, seed=partition_seed,
                n_splits=n_splits, classes=classes, val_idx=val_idx,
                fold_proba=proba, payload=record)

    y_pred = np.asarray([classes[i] for i in proba.argmax(axis=1)])
    metrics = evaluate_predictions(
        val_fold["label"].values, y_pred, labels=classes, bootstrap=False)
    return {
        "model_seed": model_seed,
        "partition_seed": partition_seed,
        "fold": fold,
        "macro_f1": metrics["macro_f1"],
        "selected_params": record.get("selected_params"),
        "selected_epoch": record.get("selected_epoch"),
        # Carried into the replicate record so _report_fit_diagnoses sees a
        # verdict for replicates too, not only for the primary folds.
        "fit_diagnosis": record.get("fit_diagnosis"),
        # The same for the cost diagnostics. On the few-shot component two
        # replicates of the IDENTICAL work on fold 0 ran 1215 s and 4371 s; a
        # different `model_seed` draws different in-context examples, so prompt
        # length is the obvious explanation, and it can only be checked if the
        # replicate records carry `tokenisation_predict` like the fold records.
        #
        # Carried by key rather than by family: TML and encoder records simply do
        # not have these, and `.get` leaves them absent instead of writing nulls.
        **{k: record[k] for k in ("tokenisation", "tokenisation_predict",
                                  "prompt_skeleton_tokens", "max_prompt_tokens",
                                  "verbaliser_token_lengths")
           if k in record},
        "runtime_seconds": runtime,
        "resumed": resumed,
    }


# ---------------------------------------------------------------------------
# Environment and configuration
# ---------------------------------------------------------------------------

def load_matrix() -> dict[str, Any]:
    with open(MATRIX_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pool(subtask: str, edition: str):
    """
    The E2 evaluation pool: the training split, deduplicated within edition.

    See the module docstring for why the train-test overlap rule stays off.
    """
    df = load_split(subtask, edition, "train")
    df, _ = apply_data_quality_rules(df, dedup=True, drop_overlap=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# One component run
# ---------------------------------------------------------------------------


def _git_commit() -> str | None:
    """The commit the code was at, or None outside a repo.

    Recorded so a result can be traced to the code that made it -- the bootstrap
    already warns on a dirty tree for the same reason. Never fatal: a missing
    commit must not stop a campaign that is otherwise fine.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, cwd=str(Path(__file__).resolve().parent.parent))
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def run_component(
    subtask: str,
    edition: str,
    component: str,
    *,
    seeds=None,
    n_splits: int = 5,
    n_jobs: int = -1,
    device: str | None = None,
    max_epochs: int | None = None,
    bootstrap_n: int = 1000,
    ci: float = 0.95,
    store: ComponentStore | None = None,
    out_dir: Path = RESULTS_DIR,
    force: bool = False,
    ckpt_root: Path | None = CKPT_ROOT,
    experiment: str = "E2",
    condition: str = "none",
) -> dict[str, Any]:
    """Run all repeats of one component on one subtask/edition and persist them.

    `experiment` E4a / E4b runs the same component under an imbalance condition
    (E4a) or across subtasks (E4b), at the configuration frozen by
    `src/e4_config.py`, on the primary seed only and with no seed replicates --
    E4 is sized as conditions x folds x subtasks and nothing more. Every E4
    artefact, store key and checkpoint carries its own name, so an E4 run can
    neither overwrite nor resume from the E2 run it is compared against.
    """
    exp = str(experiment).upper()
    if exp not in ("E2", "E4A", "E4B"):
        raise ValueError("unknown experiment {!r}; E2, E4a or E4b".format(experiment))
    e4 = fixed_grid = variant = None
    if exp == "E2":
        if condition != "none":
            raise ValueError("an imbalance condition belongs to E4a, not E2 "
                             "(E2 is the reference cell)")
    else:
        matrix_now = load_matrix()
        block = imbalance.protocol(matrix_now)
        status = str(block.get("status", ""))
        if not status.startswith("confirmed"):
            raise RuntimeError(
                "REFUSING {} for {}/{}: e4_protocol.status is {!r}. The E4 "
                "conditions and configuration rule are a pre-registration; nothing "
                "runs under them until they are confirmed.".format(
                    exp, subtask, component, status))
        fam = family_of(component)
        if exp == "E4B" and condition != "none":
            raise ValueError("E4b holds the condition at none (e4_protocol.e4b)")
        imbalance.check_applicable(condition, component, fam, matrix_now)
        folds_key = "cv_folds_e4a" if exp == "E4A" else "cv_folds_e2"
        expected_splits = int(matrix_now["protocol"][folds_key])
        if n_splits != expected_splits:
            raise ValueError(
                "{} runs at {} = {} folds and this call asks for {}".format(
                    exp, folds_key, expected_splits, n_splits))
        fixed = e4_config.fixed_params_for(component, subtask, edition, exp,
                                           matrix=matrix_now)
        fixed_grid = [fixed]
        e4 = {"experiment": exp, "condition": condition, "fixed_params": fixed}
        variant = "e4a-{}".format(condition) if exp == "E4A" else "e4b"
        if Path(out_dir) == RESULTS_DIR:
            out_dir = RESULTS_DIR.parent / exp.lower()
    exp_label = {"E2": "E2", "E4A": "E4a", "E4B": "E4b"}[exp]
    cell_stem = "{}_{}_{}{}".format(subtask, edition, component,
                                    "__" + variant if variant else "")
    if exp == "E2":
        out_path = Path(out_dir) / "e2_{}_{}_{}.json".format(subtask, edition, component)
    else:
        out_path = Path(out_dir) / "{}_{}_{}_{}{}.json".format(
            exp.lower(), subtask, edition, component,
            "__" + condition if exp == "E4A" else "")
    if out_path.exists() and not force:
        # ⚠️ A SKIP MUST VERIFY WHAT IT IS SKIPPING. Skipping on filename
        # existence alone lets a 2-fold measurement run written under the
        # campaign's filename stand in for a genuine 5-fold run, with nothing
        # to tell them apart. The artefact is committed, so a clone brings it
        # back; the guard therefore has to live here, not in a tidy working
        # tree.
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        meta = existing.get("meta") or {}
        prior_splits = meta.get("n_splits")
        prior_cfg = meta.get("config_id")
        now_cfg = compute_config_id(component, e4=e4)
        mismatch = []
        if prior_splits is not None and prior_splits != n_splits:
            mismatch.append("it has {} folds, this run asks for {}".format(
                prior_splits, n_splits))
        if prior_cfg is not None and prior_cfg != now_cfg:
            mismatch.append("its config_id is {}, the matrix now gives {}".format(
                prior_cfg, now_cfg))
        if mismatch:
            raise RuntimeError(
                "REFUSING to skip {}: {}.\n"
                "  Skipping it would silently drop {}/{} from this campaign and "
                "leave an incomparable artefact behind.\n"
                "  Either move it aside (results/measurements/ is where "
                "measurement runs belong -- see its README), or pass --force to "
                "overwrite it deliberately.".format(
                    out_path.name, "; and ".join(mismatch), subtask, component))
        print("  [skip] {} exists and matches ({} folds, config {}); "
              "use --force to redo".format(
                  out_path.name, prior_splits,
                  prior_cfg if prior_cfg is not None
                  else "not recorded -- pre-2026-08-27 artefact, only folds checked"))
        return existing

    api = component_api(component, n_jobs=n_jobs, device=device,
                        max_epochs=max_epochs, subtask=subtask,
                        condition=condition, grid=fixed_grid)
    if seeds is None:
        seeds = (PRIMARY_SEED,) if e4 else tuple(api["seeds"])
    seeds = tuple(seeds)

    df = load_pool(subtask, edition)
    classes = resolve_classes(df["label"].values)
    class_counts = {str(c): int((df["label"].values == c).sum()) for c in classes}
    print("\n=== {} / {} / {} ===".format(subtask.upper(), edition, component))
    print("  pool n={}  classes={}  counts={}".format(len(df), classes, class_counts))

    store = store if store is not None else ComponentStore()
    repeats: list[dict[str, Any]] = []
    primary: dict[str, Any] | None = None
    t_start = time.time()

    # Every store entry names the matrix
    # generation that produced it, so E3 can assert a single generation
    # instead of trusting a `created` timestamp.
    matrix_meta = load_matrix()["meta"]
    config_id = compute_config_id(component, e4=e4)
    data_rules_id = compute_data_rules_id()
    # Everything that would make two checkpoints incomparable. Seed, split
    # count, class ordering and the fold's validation indices are checked
    # separately by the checkpoint itself, so they are deliberately not here.
    fingerprint = "|".join([component, subtask, str(edition), config_id,
                            data_rules_id, str(matrix_meta["version"])])

    for seed in seeds:
        t_seed = time.time()
        fold_of_item = np.full(len(df), -1, dtype=np.int16)

        # The record is RETURNED rather than appended to a closure list: run_cv
        # collects it into fold_payloads and checkpoints it with the fold, so a
        # restored fold carries its selected params instead of a hole.
        def _on_fold(fold_idx, model, val_df, fold_proba, _seed=seed):
            return dict(model.record, fold=fold_idx)

        # Fold membership is recomputed here rather than returned by run_cv, so
        # that the component store can say which fold produced each row.
        from src.harness import make_cv_splits

        for f, (_, val_idx) in enumerate(make_cv_splits(df, n_splits=n_splits, seed=seed)):
            fold_of_item[val_idx] = f

        seed_ckpt = (Path(ckpt_root) / "{}_seed{}".format(
            cell_stem, seed)) if ckpt_root else None

        res = run_cv(
            df,
            api["make_train_fn"](classes, seed),
            predict_proba_fn=api["predict_proba_fn"],
            classes=classes,
            n_splits=n_splits,
            seed=seed,
            bootstrap=(seed == PRIMARY_SEED),
            bootstrap_n=bootstrap_n,
            ci=ci,
            on_fold=_on_fold,
            checkpoint_dir=seed_ckpt,
            checkpoint_fingerprint=fingerprint if seed_ckpt else None,
            verbose=False,
        )
        fold_records = res["fold_payloads"]
        # Not filtered: a missing payload would shorten this list and silently
        # misalign selected_params_per_fold / temperature_per_fold against the
        # folds they are printed beside.
        assert all(r is not None for r in fold_records) and len(fold_records) == n_splits, \
            "fold payloads {} do not line up with {} folds".format(fold_records, n_splits)

        pooled = res["pooled"]
        elapsed = time.time() - t_seed
        print("  seed {}: pooled macro-F1 {:.4f}  folds {}  ({:.0f}s)".format(
            seed, pooled["macro_f1"],
            [round(f["macro_f1"], 4) for f in res["folds"]], elapsed))

        key = run_key(subtask, edition, component, seed, variant=variant)
        store.save(
            key,
            proba=res["oof_proba"],
            y_true=res["oof_true"],
            ids=res["oof_index"],
            classes=classes,
            fold=fold_of_item,
            meta={
                "subtask": subtask, "edition": edition, "component": component,
                "seed": seed, "n_splits": n_splits,
                "aggregation": "pooled_out_of_fold",
                "uncalibrated": res["uncalibrated"],
                "pooled_macro_f1": pooled["macro_f1"],
                "matrix_version": matrix_meta["version"],
                "config_id": config_id,
                "data_rules_id": data_rules_id,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                # E2 sidecars stay exactly the shape they were; an E4 sidecar
                # carries the identity load_many must re-derive it under.
                **({"experiment": exp_label, "e4": e4} if e4 else {}),
            },
        )

        repeat = {
            "seed": seed,
            "pooled_macro_f1": pooled["macro_f1"],
            "pooled_weighted_f1": pooled["weighted_f1"],
            "per_fold_macro_f1": [f["macro_f1"] for f in res["folds"]],
            "fold_sd_macro_f1": res["std_macro_f1"],
            # A class with zero support in a fold's validation part still enters
            # that fold's macro-F1 as 0.0, which deflates the fold score and
            # inflates fold_sd above. Never non-empty on the 2025 pools (minimum
            # observed support over all 45 folds is 12, DBO subversive), but it
            # will fire on thin E4a/E4c cells, and a silently deflated fold_sd is
            # worse than a loud one.
            "zero_support_classes_per_fold": [f["zero_support_classes"] for f in res["folds"]],
            "per_class_f1": {c: r["f1"] for c, r in pooled["per_class"].items()},
            "ece": pooled["calibration"]["ece"],
            "brier": pooled["calibration"]["brier"],
            "selected_params_per_fold": [r["selected_params"] for r in fold_records],
            "temperature_per_fold": [r["temperature"]["temperature"] for r in fold_records],
            "folds_detail": fold_records,
            "component_store_key": key,
            # WALL CLOCK of this process. On a resumed run it measures only the
            # folds that were retrained, so it is NOT the cost of the cell --
            # use fold_runtime_seconds_sum for that. The distinction matters
            # because the GPU arms were budgeted on measured unit cost.
            "runtime_seconds": elapsed,
            # Sum of the per-fold training times, restored folds included, so a
            # resumed cell still reports the work it represents. None when the
            # component does not record a per-fold runtime (the TML families).
            "fold_runtime_seconds_sum": (
                float(sum(r["runtime_seconds"] for r in fold_records))
                if all("runtime_seconds" in r for r in fold_records) else None),
            # Provenance, not bookkeeping: on the encoder a resumed cell mixes
            # folds trained in different processes, and the artefact must name
            # where a number came from.
            "resumed_folds": res["resumed_folds"],
        }
        repeats.append(repeat)
        if seed == PRIMARY_SEED:
            primary = pooled

    # Seed replicates: same partition, same fold, different training seed.
    replicates: list[dict[str, Any]] = []
    for model_seed in (() if e4 else api["replicate_seeds"]):
        rep_ckpt = (Path(ckpt_root) / "{}_{}_{}_replicate{}".format(
            subtask, edition, component, model_seed)) if ckpt_root else None
        rep = run_seed_replicate(df, component, classes, api=api,
                                 model_seed=model_seed, n_splits=n_splits,
                                 ckpt_dir=rep_ckpt,
                                 fingerprint=("{}|replicate{}".format(fingerprint, model_seed)
                                              if rep_ckpt else None))
        print("  replicate seed {} on fold {}: macro-F1 {:.4f}  ({:.0f}s)".format(
            model_seed, rep["fold"], rep["macro_f1"], rep["runtime_seconds"]))
        replicates.append(rep)

    seed_scores = [r["pooled_macro_f1"] for r in repeats]
    matrix = load_matrix()

    result: dict[str, Any] = {
        "experiment": exp_label,
        # The work package an artefact belongs to, so the summary can group
        # runs by the campaign that produced them: B3 classical, B4 encoder,
        # B6 LLMs, B8 E4a, B9 E4b.
        "package": ({"tml": "B3", "encoder": "B4"}.get(api["family"], "B6") if not e4
                    else {"E4A": "B8", "E4B": "B9"}[exp]),
        # `meta` carries env_pin and n_splits; the commit is the one thing that
        # ties a number to the code that produced it.
        "git_commit": _git_commit(),
        "subtask": subtask,
        "edition": edition,
        "component": component,
        "meta": {
            "design_lock": matrix["meta"]["design_lock"],
            "matrix_version": matrix["meta"]["version"],
            # As in the store sidecars, so the summary layer can tell which rule
            # generation a reported figure belongs to.
            "data_rules_id": data_rules_id,
            "primary_metric": "macro_f1",
            "aggregation": "pooled_out_of_fold",
            "imbalance_condition": ("none (D4 reference condition; E4a varies this)"
                                    if not e4 else condition),
            "data_rules": {
                "within_edition_dedup": True,
                "drop_train_test_overlap": False,
                "note": ("the overlap rule protects evaluation on the official test set, "
                         "which E2 does not use; leaving it off keeps the pool identical "
                         "to the E1 baseline pool and preserves thin-class instances"),
            },
            "pool_n": int(len(load_pool(subtask, edition))),
            "classes": [str(c) for c in resolve_classes(load_pool(subtask, edition)["label"].values)],
            "n_splits": n_splits,
            # The skip guard compares `meta.config_id`, so the run file must
            # carry it: without it a matrix change after the campaign -- another
            # lr window, say -- would be skipped in silence.
            "config_id": config_id,
            "seeds": list(seeds),
            "primary_seed": PRIMARY_SEED,
            "family": api["family"],
            **api["meta"],
            # An E4 cell ran ONE frozen configuration, and the meta says so --
            # otherwise it would carry the E2 grid it never searched, and
            # selection_diagnosis would read a grid that was not the one run.
            **({"grid": fixed_grid,
                "grid_deviation": ("no search: the configuration frozen in "
                                   "results/e4_fixed_configurations.json under "
                                   "e4_protocol.{}.configuration".format(exp.lower())),
                "e4": e4,
                "replicate_seeds_skipped": "E4 has no D3 replicates (D5)"}
               if e4 else {}),
            "bootstrap_n": bootstrap_n,
            "ci_level": ci,
            "env_pin": env_pin(),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "primary": primary,
        "repeats": repeats,
        "seed_replicates": replicates,
        "dispersion": {
            "mean_pooled_macro_f1": float(np.mean(seed_scores)),
            "seed_sd_macro_f1": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0,
            "fold_sd_macro_f1_primary": repeats[0]["fold_sd_macro_f1"],
            "per_seed_pooled_macro_f1": seed_scores,
            # For the encoder the seed spread comes from the fold replicates and
            # is a single-fold quantity; it must not be read as the spread of a
            # pooled out-of-fold estimate.
            "replicate_fold_macro_f1": [r["macro_f1"] for r in replicates],
            "replicate_sd_macro_f1": (
                float(np.std([repeats[0]["per_fold_macro_f1"][ENCODER_FIXED_FOLD]]
                             + [r["macro_f1"] for r in replicates], ddof=1))
                if replicates else 0.0),
        },
        "runtime_seconds": time.time() - t_start,
        # True when any fold in this cell came off a checkpoint. Anything
        # deriving a cost from this artefact must read fold_runtime_seconds_sum
        # rather than runtime_seconds when it is set.
        "resumed": any(r["resumed_folds"] for r in repeats),
    }
    if result["resumed"]:
        print("  NOTE resumed from checkpoints -- runtime_seconds measures only "
              "the retrained folds. Cost: read fold_runtime_seconds_sum.")

    # The grid-edge and selection-stability verdict travels WITH the artefact,
    # so the summary and the thesis read a recorded judgement rather than
    # re-deriving one -- and so a reader of the artefact can see that the
    # question was asked at all.
    result["selection_diagnosis"] = selection_diagnosis_for(result)
    # Evidence for the epoch-stopping rule itself, from the traces this run
    # produced. Written into the artefact so the rule is argued from a measurement.
    result["stopping_evidence"] = stopping_report(result)
    save_results(result, out_path)

    # The cell is durable now: its results JSON exists, and that alone makes a
    # rerun skip it. Keeping the fold checkpoints past this point would only
    # leave stale copies of a finished computation lying around.
    if ckpt_root:
        # Matched on the two suffixes this cell actually creates rather than on
        # the bare prefix: a plain "{cell}_*" glob would also sweep a component
        # whose name merely starts with this one's.
        stem = cell_stem
        for pattern in ("{}_seed*".format(stem), "{}_replicate*".format(stem)):
            for d in Path(ckpt_root).glob(pattern):
                shutil.rmtree(d, ignore_errors=True)

    print("  mean pooled macro-F1 = {:.4f} (seed SD {:.4f})".format(
        result["dispersion"]["mean_pooled_macro_f1"],
        result["dispersion"]["seed_sd_macro_f1"]))
    _report_fit_diagnoses(result)
    _report_selection(result)
    _report_stopping(result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """The CLI, separable from main so a test can read its defaults.

    Those defaults are not cosmetic: a campaign launch passes neither --folds
    nor --bootstrap-n, so whatever stands here IS the protocol for a campaign
    launch. tests/test_config_matches_matrix.py checks them
    against configs/e2_matrix.yaml for exactly that reason.
    """
    parser = argparse.ArgumentParser(description="Run the E2 component experiments.")
    parser.add_argument("--arm", choices=["tml", "encoder", "llm"], default=None,
                        help="run a whole component family across all subtasks")
    parser.add_argument("--subtask", default=None)
    parser.add_argument("--edition", default="2025")
    parser.add_argument("--component", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="override the family default (tml 42/43/44, encoder 42)")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu; auto-detected when omitted")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="encoder only; overrides the compute CEILING, which "
                             "since matrix v1.7 is read from e2_matrix.yaml "
                             "(currently {}) and is not the stopping criterion -- "
                             "early stopping with patience {} is".format(
                                 enc.MAX_EPOCHS, enc.PATIENCE))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--checkpoint-dir", default=str(CKPT_ROOT),
                        help="where finished folds are parked so an interrupted "
                             "run resumes instead of retraining (default: %(default)s)")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="disable fold checkpointing entirely")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the runs that would happen, then exit")
    parser.add_argument("--experiment", choices=["e2", "e4a", "e4b"], default="e2",
                        help="e4a: imbalance conditions at 3 folds; e4b: the frozen "
                             "configuration across subtasks at 5 folds. Both refuse "
                             "until e4_protocol.status is confirmed")
    parser.add_argument("--condition", choices=list(imbalance.CONDITIONS) + ["all"],
                        nargs="+", default=["none"],
                        help="e4a only; one or more conditions, or 'all' = every "
                             "condition the protocol defines for the component. Several are "
                             "accepted because a subtask whose four cells exceed "
                             "one machine's wall clock is split across runs")
    return parser


def main(argv=None) -> int:
    # ⚠️ The parser is BOUND. This once read
    # `args = build_parser().parse_args(argv)` and the error branch below then
    # called `parser.error(...)` on a name that does not exist -- so the one
    # invocation the branch is written for (`python -m src.e2_runner` with
    # neither --arm nor --subtask/--component) died with a bare NameError
    # instead of the usage message. Harmless in money, but it is a refusal that
    # cannot deliver its refusal, which is the pattern this project keeps
    # finding, and it fires on the most likely typo of the campaign command.
    parser = build_parser()
    args = parser.parse_args(argv)

    matrix = load_matrix()
    if args.arm:
        subtasks = matrix["subtasks"]
        if args.arm == "tml":
            components = list(TML_COMPONENTS)
        elif args.arm == "encoder":
            components = [c for c in enc.ENCODER_COMPONENTS if c == "encoder_1b"]
        else:
            # `--arm llm` runs BOTH LLM components. The completeness check
            # stays, because a clean exit looks identical either way: a
            # component declared in the matrix and absent from this list must
            # still say so out loud.
            components = list(llm.LLM_COMPONENTS) + list(fewshot.FEWSHOT_COMPONENTS)
            declared = [c for c, b in matrix["components"].items()
                        if str(b.get("family", "")).startswith("llm")]
            missing = [c for c in declared if c not in components]
            if missing:
                print("  NOTE: --arm llm runs {} and NOT {} -- declared in the "
                      "matrix with no module behind it.".format(
                          components, missing))
        jobs = [(st, args.edition, c) for st in subtasks for c in components]
    else:
        if not (args.subtask and args.component):
            parser.error("give --arm, or both --subtask and --component")
        jobs = [(args.subtask, args.edition, args.component)]

    exp = args.experiment.upper()
    if exp == "E2":
        if args.condition != ["none"]:
            parser.error("--condition belongs to --experiment e4a")
        n_splits = args.folds
        jobs = [(st, ed, comp, "none") for st, ed, comp in jobs]
    else:
        folds_key = "cv_folds_e4a" if exp == "E4A" else "cv_folds_e2"
        n_splits = int(matrix["protocol"][folds_key])
        if args.folds not in (5, n_splits):      # 5 is the parser default
            parser.error("{} runs at {} = {} folds; --folds {} contradicts it".format(
                exp, folds_key, n_splits, args.folds))
        if exp == "E4B" and args.condition != ["none"]:
            parser.error("E4b holds the condition at none")
        if args.arm:
            comps = [c for c in dict.fromkeys(c for _, _, c in jobs)
                     if family_of(c) != "llm_fewshot"]
            cells = ([tuple(x) for x in matrix["e4_protocol"]["e4a"]["cells"]]
                     if exp == "E4A" else [(st, "2025") for st in matrix["subtasks"]])
            base = [(st, ed, c) for st, ed in cells for c in comps]
        else:
            base = jobs
        jobs = []
        for st, ed, comp in base:
            if "all" in args.condition:
                key = imbalance.family_key(comp, family_of(comp))
                conds = [k for k in imbalance.CONDITIONS
                         if key in matrix["imbalance_applicability"].get(k, [])]
            else:
                conds = list(dict.fromkeys(args.condition))
            # Refused HERE, before a dry run can list it as runnable: a run
            # launched on a cell the protocol does not define would die at its
            # first fit.
            for k in conds:
                try:
                    imbalance.check_applicable(k, comp, family_of(comp), matrix)
                except ValueError as e:
                    parser.error(str(e))
            jobs += [(st, ed, comp, k) for k in conds]

    print("{} runner: {} run(s), seeds {}, {} folds".format(
        exp.replace("A", "a").replace("B", "b"), len(jobs), args.seeds, n_splits))
    for st, ed, comp, cond in jobs:
        print("  - {} / {} / {}{}".format(st, ed, comp,
                                          "" if exp == "E2" else " / " + cond))
    if args.dry_run:
        return 0

    # LightGBM warns that a sparse matrix carries no feature names; it is
    # emitted once per prediction call and says nothing about correctness.
    warnings.filterwarnings(
        "ignore", message="X does not have valid feature names", category=UserWarning
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ComponentStore()

    t0 = time.time()
    for st, ed, comp, cond in jobs:
        run_component(
            st, ed, comp,
            experiment=exp,
            condition=cond,
            seeds=tuple(args.seeds) if args.seeds else None,
            n_splits=n_splits,
            n_jobs=args.n_jobs,
            device=args.device,
            max_epochs=args.max_epochs,
            bootstrap_n=args.bootstrap_n,
            store=store,
            out_dir=out_dir,
            force=args.force,
            ckpt_root=None if args.no_checkpoints else Path(args.checkpoint_dir),
        )
    print("\nAll runs done in {:.1f} min.".format((time.time() - t0) / 60))
    return 0



def _report_fit_diagnoses(result):
    """Print the over/underfit verdict where a human will actually see it.

    Hours of fine-tuning that turn out to over- or underfit are thrown-away
    time, and the result is attackable. `diagnose_fit` writes a
    verdict into every fold record, but a verdict inside a JSON nobody opens is
    not a safeguard. This surfaces it next to the headline macro-F1 -- and only
    when something is wrong, so it keeps meaning something.

    It reports; it does not act. Changing max_epochs mid-run would make the
    folds incomparable, which is a worse defect than the one it would fix.
    """
    counts, examples = {}, {}
    n_folds = n_flagged = 0
    for rec in _walk_fold_records(result):
        diag = rec.get("fit_diagnosis")
        if not diag:
            continue
        n_folds += 1
        flags = diag.get("flags", [])
        if flags:
            n_flagged += 1
        notes = diag.get("notes", [])
        for i, flag in enumerate(flags):
            counts[flag] = counts.get(flag, 0) + 1
            examples.setdefault(flag, notes[i] if i < len(notes) else "")
    if not n_folds or not counts:
        return
    print("  " + "!" * 60)
    print("  FIT DIAGNOSIS: {} of {} folds raised a flag.".format(n_flagged, n_folds))
    for flag, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("    [{}] {}x -- {}".format(flag, n, examples.get(flag, "")))
    if "budget_binding" in counts:
        print("    -> UNDERFITTING. Raise max_epochs in configs/e2_matrix.yaml and")
        print("       re-run the WHOLE arm; a per-fold change makes folds incomparable.")
    if "collapsed_early" in counts:
        print("    -> OVERFITTING inside one pass. Lower lr, or lora_r.")
    print("  " + "!" * 60)


# ---------------------------------------------------------------------------
# Selection diagnosis: is the grid deciding, or is the data?
# ---------------------------------------------------------------------------

# ⚠️ WHY THIS EXISTS. A grid whose winner sits on its edge is deciding instead
# of the data: the SVM's C axis showed it (the winner sat on the upper edge in
# 14 of 15 folds), and so did the encoder's lr axis. Both were found by hand. The campaign already records everything needed to
# find it automatically -- `selected_params_per_fold` and the full per-cell
# `selection_grid` -- so it is wired in rather than left to whoever remembers.
#
# It reports; it never acts. Re-cutting a grid mid-arm would make the folds
# incomparable, which is the rule `diagnose_fit` follows one function down.

def _axis_values(grid: list[dict[str, Any]]) -> dict[str, list]:
    """The distinct values each grid axis actually offers, in sorted order."""
    axes: dict[str, list] = {}
    for cell in grid:
        for k, v in cell.items():
            axes.setdefault(k, [])
            if v not in axes[k]:
                axes[k].append(v)
    return {k: sorted(vs, key=lambda x: (isinstance(x, str), x))
            for k, vs in axes.items() if len(vs) > 1}


def diagnose_selection(selected_per_fold, grid, selection_grids=None,
                       fold_scores=None) -> dict[str, Any]:
    """Say whether the search grid, rather than the data, chose the winner.

    Two independent failure modes, and the artefact carries the evidence for
    both:

      grid_edge:<axis>   every fit selected the SAME extreme value of an axis.
                         The optimum is then not interior and may lie outside
                         the window entirely. One fit
                         on an edge means nothing; all of them on the same edge
                         means the window is placed wrong.
      selection_noise    the margin between the winning cell and the runner-up
                         is, in the median, smaller than the fold-to-fold
                         spread of the final scores. The grid is then choosing
                         between configurations the evaluation cannot tell
                         apart, so `selected_params_per_fold` is a coin toss
                         and has to be reported as one.
      single_cell_grid   there is nothing to select. Recorded rather than
                         passing silently, because a reduced grid is a
                         legitimate thing to run and an unlabelled "no edge
                         found" would read as reassurance.

    Everything it decides on is also returned as numbers: a flag is not
    reportable on its own, and 3.3 needs the counts.
    """
    selected = [dict(p) for p in (selected_per_fold or []) if p]
    axes = _axis_values(list(grid or []))
    flags: list[str] = []
    notes: list[str] = []

    edge: dict[str, Any] = {}
    for axis, values in axes.items():
        chosen = [p[axis] for p in selected if axis in p]
        if not chosen:
            continue
        lo, hi = values[0], values[-1]
        at_lo = sum(1 for v in chosen if v == lo)
        at_hi = sum(1 for v in chosen if v == hi)
        edge[axis] = {
            "values": list(values),
            "selected_counts": {str(v): sum(1 for c in chosen if c == v) for v in values},
            "n_fits": len(chosen),
            "share_at_lower_edge": at_lo / len(chosen),
            "share_at_upper_edge": at_hi / len(chosen),
        }
        if at_lo == len(chosen) or at_hi == len(chosen):
            which = "lower" if at_lo == len(chosen) else "upper"
            flags.append("grid_edge:{}".format(axis))
            notes.append(
                "every one of {} fits selected {} = {!r}, the {} end of {}. The "
                "optimum is not interior, so the window may be placed wrong and "
                "the grid, not the data, is deciding (#31a). Widen it past that "
                "end and re-run the WHOLE arm, or state in 3.3 that the axis was "
                "not resolved.".format(len(chosen), axis,
                                       lo if which == "lower" else hi, which, values))

    margins: list[float] = []
    for cells in (selection_grids or []):
        scores = sorted((float(c["selection_macro_f1"]) for c in (cells or [])
                         if "selection_macro_f1" in c), reverse=True)
        if len(scores) >= 2:
            margins.append(scores[0] - scores[1])
    margin_stats = None
    if margins:
        margin_stats = {
            "median": float(np.median(margins)),
            "mean": float(np.mean(margins)),
            "min": float(np.min(margins)),
            "n_fits": len(margins),
        }
        scores_list = list(fold_scores) if fold_scores is not None else []
        spread = float(np.std(scores_list, ddof=1)) if len(scores_list) > 1 else None
        margin_stats["fold_sd"] = spread
    distinct = {tuple(sorted(p.items(), key=lambda kv: str(kv[0]))) for p in selected}

    # ⚠️ THE SECOND CONDITION IS NOT COSMETIC. Measured over the nine
    # classical-component artefacts plus an encoder measurement, the median winner-to-runner-up
    # margin is below the fold-to-fold spread in NINE of ten cells. That is a
    # real property of this project rather than ten defects -- the inner
    # selection separates configurations by less than the partition separates
    # folds -- but a flag that fires on nine of ten cells is not an alarm, and
    # this project's own rule (test_fit_diagnosis) is that an alarm which always
    # fires stops meaning anything. So the flag additionally requires the
    # instability to be REALISED: if every fold picked the same configuration,
    # nothing arbitrary happened, whatever the margin. On the committed set that
    # takes it from nine cells to eight, which is still the honest answer and is
    # why the margin numbers are reported unconditionally below and the
    # printout for this flag is one line rather than a banner.
    if margin_stats is not None and margin_stats.get("fold_sd") is not None \
            and len(distinct) > 1 and margin_stats["median"] < margin_stats["fold_sd"]:
        flags.append("selection_noise")
        notes.append(
            "{} different configurations won across {} fits, and the median "
            "winner-to-runner-up margin on the selection slice is {:.4f} -- "
            "below the {:.4f} fold-to-fold spread of the final scores. The grid "
            "is separating configurations the evaluation cannot, so "
            "selected_params_per_fold must be reported as UNRESOLVED rather "
            "than as a finding about the model.".format(
                len(distinct), len(selected), margin_stats["median"],
                margin_stats["fold_sd"]))

    if not axes:
        flags.append("single_cell_grid")
        notes.append("the grid has no axis with more than one value -- nothing "
                     "was selected, so no edge check is possible.")

    return {
        "flags": flags,
        "notes": notes,
        "axes": edge,
        "n_distinct_winners": len(distinct),
        "n_fits": len(selected),
        "margin": margin_stats,
        "ok": not [f for f in flags if f != "single_cell_grid"],
    }


def selection_diagnosis_for(result: dict[str, Any]) -> dict[str, Any] | None:
    """Run `diagnose_selection` over a finished cell's primary repeat.

    The seed replicates are folded in on purpose: they are further independent
    fits of the same grid, and an edge that holds across seven fits is a
    stronger statement than one across five.
    """
    repeats = result.get("repeats") or []
    if not repeats:
        return None
    primary = next((r for r in repeats if r.get("seed") == PRIMARY_SEED), repeats[0])
    detail = primary.get("folds_detail") or []
    replicates = result.get("seed_replicates") or []
    selected = list(primary.get("selected_params_per_fold") or [])
    selected += [r.get("selected_params") for r in replicates]
    return diagnose_selection(
        selected,
        (result.get("meta") or {}).get("grid") or [],
        selection_grids=[d.get("selection_grid") for d in detail],
        fold_scores=primary.get("per_fold_macro_f1"),
    )


# ---------------------------------------------------------------------------
# Stopping evidence: was `patience` right, measured rather than argued?
# ---------------------------------------------------------------------------

# ⚠️ WHY THIS EXISTS. The epoch axis
# is closed by early stopping with patience 2 under a ceiling of 8, and the
# patience was chosen from the 14 epoch traces this project had produced: the
# longest non-improving run still followed by a new best was 1, never 2. That
# is a defensible derivation and a thin sample, and the protocol has to answer
# the obvious objection -- "you stopped too early and missed the optimum".
#
# The campaign records every per-epoch selection score of every grid cell, so
# the argument does not have to stay an argument. Two numbers settle it:
#
#   * how often a fit stopped early WHILE ITS TRACE WAS STILL CLIMBING back
#     (last epoch above the one before it, though both below the best). That is
#     the only shape in which patience 2 can plausibly have cut off a recovery.
#     Rare -> the rule is right. Common -> patience was too tight, and the
#     thesis must say so rather than assert the opposite.
#   * how much the tail actually cost, i.e. the best score minus the last one.
#     A large gap means the fits decay fast after their peak, which is the
#     regime early stopping is FOR.
#
# It answers the other half too, at no extra cost: how often the CEILING bound,
# which is the opposite failure and the one v1.7 was written to remove.
#
# Reports only. Changing `patience` or the ceiling mid-arm would make the folds
# incomparable, exactly as with `diagnose_fit` and `diagnose_selection`.

def stopping_report(result: dict[str, Any]) -> dict[str, Any] | None:
    """Evidence for and against the epoch-stopping rule, from the traces.

    Covers every grid cell of every fold of the primary repeat -- that is the
    whole population of fine-tunes the rule was applied to, not just the
    winners, which matters because a rule has to be right on the cells it
    discards as well as on the one it keeps. The seed replicates are left out:
    their record keeps the selected epoch but not the per-cell traces.
    """
    repeats = result.get("repeats") or []
    if not repeats:
        return None
    primary = next((r for r in repeats if r.get("seed") == PRIMARY_SEED), repeats[0])
    cells = [c for d in (primary.get("folds_detail") or [])
             for c in (d.get("selection_grid") or [])]
    cells = [c for c in cells if c.get("stopping") and c.get("epoch_trace")]
    if not cells:
        return None

    n = len(cells)
    early, at_ceiling, rising_at_stop, near_miss = 0, 0, 0, 0
    best_epochs, headroom, tail_gap = [], [], []
    for c in cells:
        stop = c["stopping"]
        scores = [float(e["selection_macro_f1"]) for e in c["epoch_trace"]]
        best_epoch = int(c["best_epoch"])
        best_epochs.append(best_epoch)
        headroom.append(int(stop["epochs_run"]) - best_epoch)
        tail_gap.append(max(scores) - scores[-1])
        if stop["stopped_early"]:
            early += 1
            # The one shape in which patience can have cut off a recovery: the
            # trace turned upwards again on the very epoch the rule stopped on.
            if len(scores) >= 2 and scores[-1] > scores[-2]:
                rising_at_stop += 1
            # And the near miss: the last epoch came within 1% of the best.
            if max(scores) > 0 and (max(scores) - scores[-1]) / max(scores) < 0.01:
                near_miss += 1
        if stop.get("ceiling_bound"):
            at_ceiling += 1

    ceiling = int(cells[0]["stopping"]["ceiling"])
    patience = int(cells[0]["stopping"].get("patience") or enc.PATIENCE)
    return {
        "n_fits": n,
        "ceiling": ceiling,
        "patience": patience,
        "n_stopped_early": early,
        "n_ceiling_bound": at_ceiling,
        "best_epoch_counts": {str(e): best_epochs.count(e)
                              for e in sorted(set(best_epochs))},
        "mean_epochs_run": float(np.mean([c["stopping"]["epochs_run"] for c in cells])),
        # Compute the rule costs beyond the peak, as a share of the peak itself.
        # An oracle that stopped AT the best epoch would have run this much less.
        "overrun_ratio": (float(np.sum([c["stopping"]["epochs_run"] for c in cells]))
                          / max(1.0, float(np.sum(best_epochs)))),
        "mean_tail_gap": float(np.mean(tail_gap)),
        "rising_at_stop": rising_at_stop,
        "rising_at_stop_share": (rising_at_stop / early) if early else None,
        "near_miss_within_1pct": near_miss,
        "verdict": _stopping_verdict(early, at_ceiling, rising_at_stop, n),
    }


def _stopping_verdict(early, at_ceiling, rising, n) -> str:
    """One sentence, and it must be TRUE OF THE COUNTS rather than of the case
    it was written for. The first branch said "the ceiling bound every fit" and
    fired whenever `at_ceiling` was merely non-zero; on a 4-fit cell with 2
    bound it would have overstated by half. That is the stale-claim pattern in
    a sentence generated from data, which is the one place it has no excuse."""
    if early == 0 and at_ceiling:
        return ("no fit stopped early and {} of {} were still improving at the "
                "ceiling: the BUDGET, not the data, decided where training "
                "ended".format(at_ceiling, n))
    if early and rising / early > 0.5:
        return ("{} of {} early stops happened while the trace was climbing "
                "again -- patience is too tight to trust the epoch choice"
                .format(rising, early))
    if early and rising == 0:
        return ("{} of {} fits stopped early and none of them on a rising trace: "
                "patience was never the binding constraint on the epoch choice"
                .format(early, n))
    return ("{} of {} fits stopped early, {} on a rising trace, {} were still "
            "improving at the ceiling".format(early, n, rising, at_ceiling))


def _report_stopping(result):
    """One line, always, because this is evidence for the stopping rule rather than an alarm."""
    rep = result.get("stopping_evidence") or stopping_report(result)
    if not rep:
        return
    print("  stopping: {}/{} fits stopped early (ceiling {} bound {}), best epoch "
          "{}, mean {:.1f} epochs run, tail gap {:.4f}".format(
              rep["n_stopped_early"], rep["n_fits"], rep["ceiling"],
              rep["n_ceiling_bound"], rep["best_epoch_counts"],
              rep["mean_epochs_run"], rep["mean_tail_gap"]))
    print("            {}".format(rep["verdict"]))


def _report_selection(result):
    """Print the selection verdict beside the fit verdict, on the same terms:
    only when something is wrong, so that it keeps meaning something."""
    diag = result.get("selection_diagnosis") or selection_diagnosis_for(result)
    if not diag:
        return
    pairs = list(zip(diag["flags"], diag["notes"]))
    edges = [(f, t) for f, t in pairs if f.startswith("grid_edge:")]
    # One line, always, when the selection was unresolved: it is a reporting
    # instruction, not an operational alarm, and it is true of most cells here
    # (see diagnose_selection).
    for f, t in pairs:
        if f == "selection_noise":
            print("  NOTE selection unresolved -- {}".format(t))
    if not edges:
        return
    print("  " + "!" * 60)
    print("  SELECTION DIAGNOSIS: {} fits, {} distinct winner(s).".format(
        diag["n_fits"], diag["n_distinct_winners"]))
    for flag, text in edges:
        print("    [{}] {}".format(flag, text))
    print("    -> this does NOT invalidate the numbers; it says the SEARCH did")
    print("       not resolve. Re-cutting a grid mid-arm makes folds incomparable,")
    print("       so the decision belongs to the NEXT arm, not to this one.")
    print("  " + "!" * 60)


def _walk_fold_records(node):
    """Yield every fold record in a result tree, whatever its nesting."""
    if isinstance(node, dict):
        if "fit_diagnosis" in node or "selected_epoch" in node:
            yield node
        for v in node.values():
            yield from _walk_fold_records(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_fold_records(v)

if __name__ == "__main__":
    raise SystemExit(main())
