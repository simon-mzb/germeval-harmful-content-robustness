# German Harmful Content Detection: Robustness of Classical, Encoder and Generative Models

Code, configuration and result records of the bachelor thesis *Combining Classical,
Encoder-Based and Generative Models for Robust German Harmful Content Detection*
(Simon Manzenberger, University of Regensburg, 2026).

The thesis builds a modular pipeline on the GermEval 2025 and 2026 shared tasks on harmful
content detection: classical TF-IDF models, a fine-tuned German encoder, a fine-tuned German
LLM and a multilingual LLM prompted with few-shot examples. It combines them by soft voting,
a confidence cascade and an LLM judge, and tests how robust each is to class imbalance,
to a change of subtask and to the change from the 2025 to the 2026 edition. Every
component writes calibrated out-of-fold probabilities under one shared protocol, so every
comparison runs on identical items. This README describes how to run the code; it
deliberately reports no results. Those are in the thesis and in `results/`.

## Repository map

```
configs/e2_matrix.yaml   the declared protocol: grids, fixed settings, seeds, folds, verbalisers
src/                     the pipeline; every experiment is a module run with `python -m src.<name>`
tests/                   CPU test suites (run_all.sh runs all of them)
notebooks/               01_baseline_repro.ipynb: the organiser baseline reproduction (E1)
results/                 the result records every table and figure of the thesis is read from
figures/                 the figures printed in the thesis, generated from results/
data/                    where the GermEval data go after download (not redistributed)
PORTING.md               how this repository was assembled from the development repository
```

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/). The lock file pins every package,
including PyTorch 2.12.1 per platform (CUDA 12.6 build on Linux, CUDA 13.0 on Windows,
the PyPI build on macOS).

```bash
uv sync --frozen
uv run python -c "import nltk; nltk.download('punkt_tab')"   # the C2A baseline tokenises through NLTK
```

On macOS, LightGBM needs OpenMP (`brew install libomp`) if `import lightgbm` fails.

## Data

The GermEval 2025 and 2026 data are **not redistributed** with this repository. They are
published by the organisers under the GPL-3.0 in
[Communication-Forensics-Lab/harmful-content-detection](https://github.com/Communication-Forensics-Lab/harmful-content-detection)
(Felser, Spranger and Siegel, 2025, *Overview of the GermEval 2025 Shared Task on Harmful
Content Detection*, KONVENS 2025). [`data/README.md`](data/README.md) gives the download
command, the upstream commit the results were produced from and the expected paths. The
loader checks all 21 files against `results/data_manifest.json` and refuses on any mismatch.

## Reproducing the results

After `uv sync --frozen` and the data download,
`bash tests/run_all.sh` checks the pipeline on CPU and rebuilds every reported component
figure from the stored per-item probabilities; the commands below re-run each stage.

The analysis stages (combination, significance, ablation, error analysis) read the stored
component outputs in `results/component_store/` and need neither a GPU nor the data text.
The measuring modules refuse to overwrite a committed artefact: a bare run prints `SKIP`,
and `--force` (or `--out <path>`) produces a fresh one to compare against.

| Stage | Command | Needs |
|---|---|---|
| Data profile (3.1) | `python -m src.data_profile`, then `python -m src.data_figures` | data |
| Data defects, independent re-check (3.1) | `python -m src.verify_data_defects` | data |
| E1: organiser baselines | `notebooks/01_baseline_repro.ipynb`; per machine `python -m src.e1_machine_baseline` | data; VIO also an Ollama server with `qwen2.5:32b` (`--subtask vio --ollama-url ...`) |
| E2: classical components | `python -m src.e2_runner --arm tml` | data, CPU |
| E2: encoder | `python -m src.e2_runner --subtask dbo --component encoder_1b --device cuda` (per subtask) | data, CUDA GPU |
| E2: both LLM components | `python -m src.e2_runner --arm llm --device cuda` | data, CUDA GPU with 48 GB |
| E3: combination layer | `python -m src.e3_combination --offline` (voting and cascade); `--go` runs the judge | stored outputs; the judge also needs the data and a GWDG key |
| E3 on 2026 | `python -m src.e3_combination_2026 --plan` / `--go` | stored outputs |
| E4a/E4b: imbalance, fixed configuration | `python -m src.e4_config --verify`, then `python -m src.e2_runner --experiment e4a --subtask dbo --component encoder_1b --condition all` (`--experiment e4b` likewise) | data; GPU for the neural components |
| E4c: 2025 to 2026 | `python -m src.e4c_pool --verify`, then `python -m src.e4c_runner --go --component tml_svm --subtask dbo` | data; GPU for the neural components |
| Significance | `python -m src.significance`, `python -m src.significance_e4`, `python -m src.multiplicity_by_experiment` | stored outputs |
| Ablation | `python -m src.b12_ablation` | stored outputs |
| E5: error analysis | `python -m src.e5_error_analysis`, `python -m src.e5_irreducible` | stored outputs |
| E5: manual-reading sample | `python -m src.e5_label_sample` (draw), `python -m src.e5_label_analysis`, `python -m src.e5_examples` | data; the coded files are not included |
| Judge selection | `python -m src.gwdg_smoke --list-models`, `--model <id>` | GWDG key |
| Reliability figure | `python -m src.reliability_figure --force` (a bare run keeps the committed figure) | stored outputs |

Every module answers `--help`.

### What the records do not cover

Four statements of the thesis rest on material that is not in this repository:

* **The Windows side of the platform probe.** The probe compared macOS (arm64, Python
  3.13.15) against Windows (AMD64, Python 3.13.3) at identical versions of scikit-learn and
  XGBoost. The figures measured on the Windows machine are recorded only in the history of
  the development repository; `results/e1_platform_check.json` carries its baseline values
  as constants. In that record, `platform_effect.machines.this_machine` names the machine that
  wrote the record (Linux), while the pair it holds under the earlier data rules was
  measured on macOS. The Linux rerun of the support vector machine is included, in
  `results/platform_probe_svm/`.
* **One exploratory count in 4.5** (45 of the 72 shared errors, and the coder's agreement
  on 30 of them) was made from the coded files of the manual reading, which are not
  included; no record carries it.
* **The generated tables of the appendix section *Hyperparameters, Selections and
  Prompts*** were produced by a tool of the development repository from
  `configs/e2_matrix.yaml`, `results/e2/`, `results/e4_fixed_configurations.json` and the
  prompt builders in `src/llm_components.py` and `src/llm_fewshot.py`. Every value in them
  is in those files, but no command here rebuilds the tables.
* **The data-defect record** `results/data_defects_verification.json` is not included,
  because it quotes posts; `python -m src.verify_data_defects` regenerates it from the data.

## Thesis section to module

| Thesis | Modules |
|---|---|
| 3.1 Data | `data_loading`, `data_manifest`, `data_profile`, `data_figures`, `verify_data_defects`, `preprocessing` |
| 3.2 Pipeline Design | `tml_components`, `encoder_components`, `llm_components`, `llm_fewshot`, `calibration`, `imbalance`, `component_store`, `e3_combination` |
| 3.3 Experimental Setup | `harness`, `e2_runner`, `configs/e2_matrix.yaml`, `e1_platform_check`, `e1_machine_baseline`, `lr_probe`, `verbaliser_probe`, `gwdg_smoke` |
| 4.1 Individual Component Performance | `e2_runner`, `e2_summary`, `significance`, `baselines`, `notebooks/01_baseline_repro.ipynb` |
| 4.2 Combination Layer Results | `e3_combination`, `combination_headroom` |
| 4.3 Robustness Analysis | `e4_config`, `dimension2_selection`, `e4c_pool`, `e4c_runner`, `e3_combination_2026`, `significance_e4`, `multiplicity_by_experiment` |
| 4.4 Ablation Study | `b12_ablation` |
| 4.5 Error Analysis | `e5_error_analysis`, `e5_irreducible`, `e5_label_sample`, `e5_label_analysis`, `e5_examples` |
| Appendix: Pre-Registration of the Manual Reading | `e5_label_sample`, `results/e5_sample/dbo_sample_key.json` |
| Appendix: Supplementary Results | `e3_combination` (drift probe of the judge), `reliability_figure` |
| Appendix: Protocol Detail | `configs/e2_matrix.yaml`, `e4_config`, `results/e4_fixed_configurations.json` |

## Hardware

The classical components ran on a laptop CPU (MacBook Air, Apple M2, macOS). The encoder
and both LLM components ran on a rented NVIDIA A40 (48 GB) under Linux; the orchestration
scripts for renting and monitoring that machine are not part of this repository, and the
modules above run unchanged on any CUDA machine whose driver supports CUDA 12.6. The LLM
judge ran on the hosted GWDG ChatAI service (`mistral-medium-3.5-128b`), which needs an API
key that is not included: copy `.env.example` to `.env` and fill it in. Because a result
can depend on the platform (the thesis documents where), every record names the machine
that produced it, and no delta is formed across machines.

## What is and is not included

Included: all code of the pipeline, the protocol, the CPU test suites, and the result
records: summary artefacts, per-run records, the component store (item ids, gold labels,
fold indices and calibrated probabilities per component; no post text), the judge's answer
cache, and the sidecar of the manual-reading sample. No file contains the text of a post.

Not included: the data themselves, trained model weights and adapters, the coded files of
the manual reading, and the development infrastructure (GPU rental, working notes). Two
development notebooks are not included either: they called `e2_summary` to write
`results/e2_tml_summary.json` and `results/g2_gate_assessment.json`, which are included,
and the thesis reads no figure from either record.

Records carry provenance written at run time. Their `git_commit` fields name commits of
the development repository, which is not published; codes such as `B4` (a work package),
`G2` (a decision gate) or `D8` (a design rule) and references to a numbered project log
are identifiers from that repository. They are kept as recorded rather than rewritten.

## Licence

GPL-3.0 (`LICENSE`), the licence of the organisers' baselines this code reimplements and
of the data.
