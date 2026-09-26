# How this repository was assembled

This repository is a snapshot of the pipeline code of a larger development repository,
which also held the thesis text, working notes and the scripts that rented and monitored
the GPU machine. Only the code the experiments need was carried over, with a fresh
history. This file records what changed on the way, so that the code here can be trusted
to be the code that produced the results.

## The rule

Code was copied, not rewritten. Comments and docstrings were edited to remove references
to the development repository's working notes (numbered log entries, work-package codes,
internal file paths) and to describe the reason instead. Nothing else changed, with the
exceptions listed below.

**How this is verified.** For every module, the abstract syntax tree of the shipped file
was compared with that of the original, with docstrings removed (comments are not part of
the tree). A module passes as *identical* when the trees are equal. Where they differ, the
only admissible difference is the text of a string literal, and every such string is
listed below. Two further invariants were checked:

* `config_id` of all seven components, `data_rules_id`, and the `config_id` of all 46
  stored E4 variants recompute to exactly the values stored with the results.
* Every result record is byte-identical to the original except for the one field
  described under *Result records*.

Strings that are written **into** result records (notes, anchors, protocol descriptions)
were deliberately left unchanged, so that a re-run writes the same fields as the shipped
records. The `PROTOCOL` block of `e3_combination` is additionally hashed into its artefact.

## Modules

**Identical up to comments and docstrings (25):** `__init__`, `b12_ablation`, `calibration`, `data_figures`, `data_loading`, `data_manifest`, `data_profile`, `e1_machine_baseline`, `e1_platform_check`, `e4c_pool`, `e4c_runner`, `e5_error_analysis`, `e5_examples`, `e5_irreducible`, `e5_label_analysis`, `e5_label_sample`, `encoder_components`, `lr_probe`, `multiplicity_by_experiment`, `plotstyle`, `preprocessing`, `significance`, `significance_e4`, `verbaliser_probe`, `verify_data_defects`.

**Changed string literals (16 modules).** Error, help and usage messages that named a log entry, a design-note code or a development-only script. Each change, old text first:

* `baselines`
  * `'SentenceBERT cannot be loaded after harness.env_pin() in the same process: env_pin imports torch and lightgbm, and a second OpenMP runtime segfaults the interp ...`  
    -> `'SentenceBERT cannot be loaded after harness.env_pin() in the same process: env_pin imports torch and lightgbm, and a second OpenMP runtime segfaults the interp ...`
* `combination_headroom`
  * `'combination headroom -- what is there for B7/E3 to win?\n'`  
    -> `'combination headroom -- what is there for E3 to win?\n'`
* `component_store`
  * `'component store: this set spans several machines ({}). Expected once the encoder arm runs on a pod -- but every cross-component claim built on it must say so.'`  
    -> `'component store: this set spans several machines ({}). Expected when the GPU components ran on another machine -- but every cross-component claim built on it m ...`
  * `'{}: sidecar carries no config_id (pre-#61 generation, re-stamp or re-run)'`  
    -> `'{}: sidecar carries no config_id (older generation, re-stamp or re-run)'`
* `dimension2_selection`
  * `'D8 -- the fixed configuration for Dimension 2'`  
    -> `'The fixed configuration for Dimension 2 (rule D8)'`
  * `'D8 ranks across all three subtasks and {} cell(s) are missing: {}. A mean rank over an incomplete matrix would silently favour whichever component skipped the  ...`  
    -> `'The rule ranks across all three subtasks and {} cell(s) are missing: {}. A mean rank over an incomplete matrix would silently favour whichever component skippe ...`
* `e2_runner`
  * `"e4a only; one or more conditions, or 'all' = every condition D4 defines for the component. Several are accepted because a subtask whose four cells exceed one p ...`  
    -> `"e4a only; one or more conditions, or 'all' = every condition the protocol defines for the component. Several are accepted because a subtask whose four cells ex ...`
  * `'an imbalance condition belongs to E4a, not E2 (D4: E2 is the reference cell)'`  
    -> `'an imbalance condition belongs to E4a, not E2 (E2 is the reference cell)'`
  * `"{} trains nothing, so it has no loss- or data-level condition and no grid to fix; D4's only lever for it is prompt_coverage, which is a descriptive cell this r ...`  
    -> `"{} trains nothing, so it has no loss- or data-level condition and no grid to fix; the protocol's only lever for it is prompt_coverage, which is a descriptive c ...`
* `e2_summary`
  * `'E2 summary: results span several machines ({}). That is expected once the encoder arm runs on a pod, but every cross-component comparison in a chapter must the ...`  
    -> `'E2 summary: results span several machines ({}). That is expected when the GPU components ran on another machine, but every cross-component comparison in a chap ...`
* `e3_combination`
  * `'{} / {} is flagged uncalibrated; E3 consumes calibrated probabilities only (D7)'`  
    -> `'{} / {} is flagged uncalibrated; E3 consumes calibrated probabilities only'`
* `e3_combination_2026`
  * `'REFUSING: the E3 primary set is {}, h22 pre-registered {}'`  
    -> `'REFUSING: the E3 primary set is {}, the pre-registration names {}'`
  * `'{} / {} 2026 is flagged uncalibrated (D7)'`  
    -> `'{} / {} 2026 is flagged uncalibrated'`
* `e4_config`
  * `'\nNOT DERIVED: e4_protocol.status is {!r}. The rule is applied to the E2 selections only once it is confirmed, so that nobody ...`  
    -> `'\nNOT DERIVED: e4_protocol.status is {!r}. The rule is applied to the E2 selections only once it is confirmed, so that nobody has seen what it selects before f ...`
* `gwdg_smoke`
  * `'\nPick a judge from a different model family than the LLM component (D6), then rerun with --model <id>.'`  
    -> `'\nPick a judge from a different model family than the LLM component, then rerun with --model <id>.'`
* `harness`
  * `'Fold checkpointing is implemented for probability mode only. Hard-label mode is the pre-D7 path and is deliberately left byte-for-byte as it was.'`  
    -> `'Fold checkpointing is implemented for probability mode only. Hard-label mode is the original path and is deliberately left byte-for-byte as it was.'`
* `imbalance`
  * `'unknown imbalance condition {!r}; D4 defines {}'`  
    -> `'unknown imbalance condition {!r}; the protocol defines {}'`
  * `'D4 does not define {!r} for {} (applicability key {!r}); it is defined for {}. The comparison runs where each strategy is defined, not as a crossed grid (3.2 s ...`  
    -> `'the protocol does not define {!r} for {} (applicability key {!r}); it is defined for {}. The comparison runs where each strategy is defined, not as a crossed g ...`
* `llm_components`
  * `'llm_components.py -- the fine-tuned generative arm (B5/B6).'`  
    -> `'llm_components.py -- the fine-tuned generative arm.'`
  * `'    python -m src.llm_smoke --component llm_llammlein --n 120   (B5)'`  
    -> `'    python -m src.e2_runner --arm llm            (all subtasks, both LLM components)'`
  * `'               softmax over the label set (#54)'`  
    -> `'               softmax over the label set'`
* `llm_fewshot`
  * `'llm_fewshot.py -- the in-context generative arm (B6, second half).'`  
    -> `'llm_fewshot.py -- the in-context generative arm.'`
  * `'               softmax over the label set (#54), then temperature'`  
    -> `'               softmax over the label set, then temperature'`
  * `'  training none; {} examples per prompt; {} draws (D3 variance)'`  
    -> `'  training none; {} examples per prompt; {} draws (variance)'`
* `tml_components`
  * `'{} cannot train under {!r}: D4 defines class weighting and oversampling for the classical models and nothing else'`  
    -> `'{} cannot train under {!r}: the protocol defines class weighting and oversampling for the classical models and nothing else'`
* `gwdg_client`
  * `'GWDG credentials not found. Copy experiments/.env.example to experiments/.env and fill in OPENAI_API_KEY and OPENAI_BASE_URL. The file is gitignored; do not pa ...`  
    -> `'GWDG credentials not found. Copy .env.example to .env and fill in OPENAI_API_KEY and OPENAI_BASE_URL. The file is gitignored; do not paste the key anywhere els ...`

**Other changes.**

* `gwdg_client`: the method `GWDGClient.complete()`, a long-form text generation helper
  that no module, test or experiment of this repository calls, was removed. Verified by
  removing the same method from the original and comparing as above.
* `reliability_figure` is new here: the generator of the appendix reliability figure
  lived outside the pipeline code in the development repository. Only its paths changed.
  The figure files in `figures/e2/` are the ones printed in the thesis. Re-running the
  generator reproduces the same curves from the same bins; in a fresh environment the
  rendering came out one pixel wider. Its input-digest sidecar is not shipped, because the
  18 E2 records it hashes differ in the field described under *Result records*.
* `configs/e2_matrix.yaml`: two header comments, `meta.design_lock` and three
  `meta.changelog` entries named paths of the development repository; they were changed.
  None of these is hashed (see the `config_id` check above). Everything else, including
  the notes inside the hashed protocol blocks, is unchanged; its changelog still refers to
  numbered entries of the development log.
* `pyproject.toml`: comments shortened; the TOML data are identical, and `uv lock --check`
  passes against the unchanged `uv.lock`.

## Modules not carried over

Maintenance and development-only modules: one-off provenance backfills, GPU cost and disk
probes, smoke tests of the rented machine, a probe of batch-width chunking, and the
proxy 2026 baselines and training-dynamics figure, which the thesis does not report.

## Tests

The CPU suites were carried over with comment edits as above, except:

* `test_data_integrity.py` keeps the 21 checks of the data, the code and the stored
  artefacts; the checks of the thesis text, the writing workflow and the GPU-rental
  tooling were removed. Its table of protected modules lists only modules shipped here.
  In `check_r`, the half that inspected a module not carried over was removed, and the
  refusal message is now matched on `OpenMP` rather than on a log reference.
* `test_llm_cell.py`: two blocks that exercised a smoke-test script not carried over were
  removed.
* `test_e4_conditions.py`: one refusal check matches the reworded message.
* `run_all.sh` is new and runs exactly the suites in `tests/`.

Comments in the tests may still name numbered entries of the development log where a
reference could not be removed without rewriting the explanation.

## Result records

All records were copied unchanged except one field: in 104 run records,
`meta.design_lock` named a working-notes path of the development repository and was
rewritten to `notes/e2_design_lock.md`, the value `configs/e2_matrix.yaml` now carries.
A JSON comparison confirmed that no other field of any record changed. The records keep
their `git_commit` fields, which name commits of the development repository.
