"""Do the CONSTANTS the code runs match the values the matrix DECLARES?

tests/test_grid_matches_matrix.py asks this of the grid. Review 4
found the same defect one field over, and this time the two did NOT agree:

  `encoder_protocol.fixed_hyperparameters.lora_target_modules` declared
  ["Wqkv", "Wo"] while the code ran ["attn.Wqkv", "attn.Wo"].

PEFT matches target modules by NAME SUFFIX, and ModernBERT calls both the
attention output projection and the MLP output projection `Wo` -- so the bare
form selects 66 modules including the MLP, against the 44 attention modules the
line is commented as meaning (measured against ModernGBERT_134M). The code was
right and the matrix was wrong, which is the worse direction: the matrix is what
3.3 is written from and what `config_id` hashes, so a stored run carried the
identity of a protocol nobody ran, and a reader reproducing from the matrix
would have fine-tuned a different model.

That is the #84 pattern exactly -- a value restated in Python beside the matrix
that decides it -- and #84's own fix only covered the grid. So this covers the
rest of the surface: the fixed hyperparameters, the epoch-stopping rule, the
inner split, the fold and seed structure, and the CLI defaults that decide how
a campaign is actually launched.

The duplication stays (the constants are used in hot paths and some produced
committed B3 artefacts). This test is what makes it safe: it fails the moment
the two drift apart, in either direction.

Usage: .venv/bin/python -m tests.test_config_matches_matrix
"""
import sys

import inspect

from src import e2_runner as R
from src import encoder_components as enc
from src import harness as H
from src import tml_components as tml
from src.component_store import load_matrix
from src.data_loading import SUBTASKS

PASS = FAIL = 0


def _sig(fn, name):
    """A callable's default for one keyword -- the value a campaign gets when
    the runbook's command does not pass the flag."""
    return inspect.signature(fn).parameters[name].default


def check(name, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print("  PASS  {}".format(name))
    else:
        FAIL += 1
        print("  FAIL  {}\n          declared: {!r}\n          running : {!r}".format(
            name, want, got))


matrix = load_matrix()
enc_proto = matrix["encoder_protocol"]
tml_proto = matrix["tml_protocol"]
proto = matrix["protocol"]
fixed = enc_proto["fixed_hyperparameters"]

print("\n=== encoder fixed hyperparameters: matrix vs encoder_components.FIXED ===")
check("effective_batch_size", enc.FIXED["effective_batch_size"], fixed["effective_batch_size"])
check("warmup_ratio", float(enc.FIXED["warmup_ratio"]), float(fixed["warmup_ratio"]))
check("weight_decay", float(enc.FIXED["weight_decay"]), float(fixed["weight_decay"]))
check("max_grad_norm", float(enc.FIXED["max_grad_norm"]), float(fixed["max_grad_norm"]))
check("lora_dropout", float(enc.FIXED["lora_dropout"]), float(fixed["lora_dropout"]))
check("max_seq_len", enc.MAX_SEQ_LEN, fixed["max_seq_len"])
# The one that was actually wrong. Qualified names only: a bare suffix would
# also match ModernBERT's mlp.Wo and adapt 50% more modules than declared.
check("lora_target_modules", list(enc.FIXED["lora_target_modules"]),
      list(fixed["lora_target_modules"]))
check("  ...and every target is qualified, so no suffix collides",
      all("." in t for t in fixed["lora_target_modules"]), True)

print("\n=== the epoch axis (v1.7): derived, and asserted anyway ===")
check("max_epochs ceiling", enc.MAX_EPOCHS, enc_proto["epoch_selection"]["max_epochs"])
check("patience", enc.PATIENCE, enc_proto["epoch_selection"]["patience"])
check("  ...and the ceiling can actually exercise the patience rule",
      enc.MAX_EPOCHS > enc.PATIENCE + 1, True)

print("\n=== the inner split is one value for both families ===")
check("encoder inner_split", {k: float(v) for k, v in tml.INNER_SPLIT.items()},
      {k: float(v) for k, v in enc_proto["inner_split"].items()})
check("tml inner_split", {k: float(v) for k, v in tml.INNER_SPLIT.items()},
      {k: float(v) for k, v in tml_proto["inner_split"].items()})
check("  ...and it sums to 1", round(sum(tml.INNER_SPLIT.values()), 9), 1.0)

print("\n=== tml early stopping (v1.1) ===")
check("max_rounds", tml.MAX_BOOSTING_ROUNDS, tml_proto["early_stopping"]["max_rounds"])
check("patience", tml.EARLY_STOPPING_ROUNDS, tml_proto["early_stopping"]["patience"])

print("\n=== the fold and seed structure ===")
check("primary seed", R.PRIMARY_SEED, proto["seed_base"])
check("tml seeds", len(R.SEEDS), matrix["variance"]["tml"]["seeds"])
check("encoder replicate seeds",
      len(R.ENCODER_REPLICATE_SEEDS), matrix["variance"]["encoder"]["extra_seeds_on_fold"])
check("encoder fixed fold", R.ENCODER_FIXED_FOLD, matrix["variance"]["encoder"]["fixed_fold"])
check("encoder fixed fold (encoder_protocol says it too)",
      R.ENCODER_FIXED_FOLD, enc_proto["fixed_fold_for_extra_seeds"])
check("encoder runs one partition seed",
      len(R.component_api("encoder_1b")["seeds"]), matrix["variance"]["encoder"]["seeds"])

print("\n=== the model the code loads is the model the matrix declares ===")
# ⚠️ NOT COVERED UNTIL REVIEW 5, and it is #92's field exactly one
# more time: `MODEL_IDS` is a Python dict beside a matrix that declares `hf_id`,
# `config_id` hashes the matrix, and only the Python one is ever loaded. A
# divergence here would fine-tune one model and stamp every artefact with the
# identity of another -- the single most expensive silent defect available in
# this tree, because nothing downstream could see it.
for comp in ("encoder_1b", "encoder_134m"):
    check("{} hf_id".format(comp), enc.MODEL_IDS[comp],
          matrix["components"][comp]["hf_id"])
check("encoder_1b max_seq_len (component block, not just the protocol)",
      enc.MAX_SEQ_LEN, matrix["components"]["encoder_1b"]["max_seq_len"])
check("subtasks", list(matrix["subtasks"]), list(SUBTASKS))

print("\n=== the reporting tiers decide what Chapter 4 may CLAIM ===")
# harness.REPORTING_TIERS carries the comment "Mirrors configs/e2_matrix.yaml",
# which is the #84 sentence written out in full: a restated constant with a note
# saying so. The wording of the rules differs deliberately between the two files;
# the THRESHOLDS must not. tier_for_n is checked at each boundary rather than
# through the table, because it is the function that actually runs.
rt = matrix["reporting_tiers"]
check("interpret min_n", H.REPORTING_TIERS["interpret"]["min_n"],
      rt["interpret_normally"]["min_n"])
check("ci_gated min_n", H.REPORTING_TIERS["ci_gated"]["min_n"], rt["ci_gated"]["min_n"])
check("ci_gated max_n", H.REPORTING_TIERS["ci_gated"]["max_n"], rt["ci_gated"]["max_n"])
check("report_only max_n", H.REPORTING_TIERS["report_only"]["max_n"],
      rt["report_uninterpreted"]["max_n"])
check("tier_for_n at the boundaries",
      [H.tier_for_n(n) for n in (rt["report_uninterpreted"]["max_n"],
                                 rt["ci_gated"]["min_n"],
                                 rt["ci_gated"]["max_n"],
                                 rt["interpret_normally"]["min_n"])],
      ["report_only", "ci_gated", "ci_gated", "interpret"])

print("\n=== the shared feature space (3.2) ===")
fs = matrix["tml_protocol"]["feature_space"]
check("min_df", tml.FEATURE_SPEC["min_df"], fs["min_df"])
check("sublinear_tf", tml.FEATURE_SPEC["sublinear_tf"], fs["sublinear_tf"])
check("char_analyzer", tml.FEATURE_SPEC["char_analyzer"], fs["char_analyzer"])
# The n-gram ranges live in the COMPONENT blocks, and all three must agree with
# each other as well as with the code -- 3.2 promises ONE shared space.
for comp in ("tml_svm", "tml_xgboost", "tml_lightgbm"):
    feats = matrix["components"][comp]["features"]
    check("{} word ngrams".format(comp), list(tml.FEATURE_SPEC["word_ngram_range"]),
          list(feats["tfidf_word_ngrams"]))
    check("{} char ngrams".format(comp), list(tml.FEATURE_SPEC["char_ngram_range"]),
          list(feats["tfidf_char_ngrams"]))
    check("{} lowercase".format(comp), tml.FEATURE_SPEC["lowercase"], feats["lowercase"])

print("\n=== the fixed estimator settings the matrix declares ===")
# These are literals inside _fit_estimator, declared a second time in the matrix
# under components.tml_xgboost.fixed. Read out of the SOURCE rather than from a
# constant, because there is no constant -- which is precisely the problem.
import inspect as _inspect
_src = _inspect.getsource(tml._fit_estimator)
for key, value in matrix["components"]["tml_xgboost"]["fixed"].items():
    literal = '{}={!r}'.format(key, value).replace("'", '"')
    check("xgboost {}".format(key), literal in _src.replace("'", '"'), True)

print("\n=== the protocol-wide numbers, in the harness that uses them ===")
check("harness DEFAULT_N_SPLITS", H.DEFAULT_N_SPLITS, proto["cv_folds_e2"])
check("harness DEFAULT_SEED", H.DEFAULT_SEED, proto["seed_base"])
check("harness BOOTSTRAP_N", H.BOOTSTRAP_N, proto["bootstrap_n"])
check("harness N_SEEDS", H.N_SEEDS, matrix["variance"]["tml"]["seeds"])
# Added 2026-09-07. The significance layer runs at its OWN resample count,
# because there an interval decides a verdict rather than describing a run --
# at 1000 the vio encoder-vs-LLM verdict flipped with the resampling seed. The
# second check is the point of the first: if the two ever converge, the reason
# this key exists is gone and the verdicts are back on 1000.
import src.significance as SIG
check("significance SIG_BOOTSTRAP_N", SIG.SIG_BOOTSTRAP_N,
      proto["significance_bootstrap_n"])
check("significance resamples exceed the artefact-side bootstrap_n",
      SIG.SIG_BOOTSTRAP_N > proto["bootstrap_n"], True)
check("compare() defaults to the significance resample count",
      _sig(SIG.compare, "n_bootstrap"), proto["significance_bootstrap_n"])
check("run_cv ci default", float(_sig(H.run_cv, "ci")), float(proto["ci_level"]))
check("run_component ci default", float(_sig(R.run_component, "ci")), float(proto["ci_level"]))
check("run_component n_splits default", _sig(R.run_component, "n_splits"),
      proto["cv_folds_e2"])
check("run_component bootstrap_n default", _sig(R.run_component, "bootstrap_n"),
      proto["bootstrap_n"])
check("variance rows agree on the fold count",
      {matrix["variance"]["tml"]["folds"], matrix["variance"]["encoder"]["folds"]},
      {proto["cv_folds_e2"]})

print("\n=== the pool rules load_pool actually applies ===")
# Verified by executing load_pool's source rather than by trusting the matrix to
# describe it; tests/test_data_integrity.py then proves the consequence on data.
_pool_src = _inspect.getsource(R.load_pool)
for family in ("tml_protocol", "encoder_protocol"):
    rules = matrix[family]["pool_rules"]
    check("{} within_edition_dedup".format(family), "dedup=True" in _pool_src,
          rules["within_edition_dedup"])
    check("{} drop_train_test_overlap".format(family),
          "drop_overlap=True" in _pool_src, rules["drop_train_test_overlap"])

print("\n=== the CLI defaults are how a campaign is actually launched ===")
# Not cosmetic: the runbook's campaign command passes neither --folds nor
# --bootstrap-n, so whatever stands in the parser IS the protocol for a launch.
d = {a.dest: a.default for a in R.build_parser()._actions}
check("--folds default", d.get("folds"), proto["cv_folds_e2"])
check("--bootstrap-n default", d.get("bootstrap_n"), proto["bootstrap_n"])

print("\n" + "-" * 40)
print("  {} passed, {} failed".format(PASS, FAIL))
print("-" * 40)
sys.exit(1 if FAIL else 0)
