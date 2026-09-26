#!/usr/bin/env bash
# run_all.sh -- every check that runs on a CPU, without a GPU and without network
# access beyond the Hugging Face tokeniser/model downloads noted below.
#
# Needs the GermEval data under data/codabench/ (data/README.md). The suites are
# standalone scripts rather than pytest, so that the locked environment needs no
# extra dependency. Takes about ten minutes; the three "end to end" suites
# dominate.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
FAILED=0

run () {
  echo; echo "=== $1 ==="
  shift
  if "$@"; then :; else FAILED=$((FAILED+1)); echo "  ^^ FAILED"; fi
}

# Protocol: the code reads, and agrees with, the declared matrix
run "grid: code vs matrix"                      "$PY" -m tests.test_grid_matches_matrix
run "constants: code vs matrix"                 "$PY" -m tests.test_config_matches_matrix

# The fold loop: checkpoints, resume, over/underfit and selection diagnostics
run "fold checkpoint / resume"                  "$PY" -m tests.test_fold_resume
run "resume with the real components"           "$PY" -m tests.test_resume_integration
run "over/underfit alarm"                       "$PY" -m tests.test_fit_diagnosis
run "grid edge + selection + stopping"          "$PY" -m tests.test_selection_diagnosis
run "component-store generation guard"          "$PY" -m tests.test_store_generation

# Data: pools, folds, leakage, and the stored artefacts
run "data integrity (pools, folds, leakage)"    "$PY" -m tests.test_data_integrity
run "reported figures vs raw probabilities"     "$PY" -m tests.test_reported_numbers_rebuild

# Analysis layers over the stored outputs
run "significance layer (McNemar + paired CI)"  "$PY" -m tests.test_significance
run "E3 combination layer"                      "$PY" -m tests.test_e3_combination
run "E4 conditions + cells (CPU)"               "$PY" -m tests.test_e4_conditions

# The three component families end to end on CPU. The encoder suite downloads
# ModernGBERT-134M; the two LLM suites use a tiny random Llama with the real
# tokeniser, so they download tokenisers but no weights.
run "encoder cell end to end (CPU, slow)"       "$PY" -m tests.test_encoder_cell
run "LLM cell end to end (CPU, no weights)"     "$PY" -m tests.test_llm_cell
run "few-shot cell end to end (CPU, no weights)" "$PY" -m tests.test_llm_fewshot_cell

echo
echo "########################################"
if [ "$FAILED" -eq 0 ]; then
  echo "  all suites passed"
else
  echo "  $FAILED suite(s) FAILED"
fi
echo "########################################"
exit "$FAILED"
