"""test_data_integrity.py -- is the machinery FED correctly?

Every other suite in this directory asks whether the code does what it says.
This one asks the question that survives a correct codebase: whether the data
going into it is the data the protocol claims. That class of defect is silent
-- every test passes, every guard fires, and the number is wrong. A leaked
value is not a crash; it is a chapter that has to be withdrawn.

Ten properties, all verified by EXECUTING the real code against the real pools
(`src.e2_runner.load_pool`) rather than by reading the code that claims them:

  (a) out-of-fold construction: each pool item scored exactly once, ids unique,
      y_true is the pool's own label, rows sum to 1, `fold` reproduces -- and
      e2_runner's SEPARATE fold_of_item loop agrees with the placement run_cv
      actually used, which nothing had ever checked.
  (b) train/eval separation inside a fold, by the key the rule really uses,
      plus the residual under a STRICTER key so a near-duplicate the rule
      misses cannot hide.
  (c) the three-way inner split: disjoint, exhaustive, stratified, and the
      calibration slice unseen by the grid, the epoch choice and the fold's
      validation part.
  (d) no official test row inside an E2 pool, and the stated reason for the
      overlap rule being off is re-measured rather than quoted.
  (e) cross-edition contamination: drop_cross_edition is off everywhere but
      E4c, and the frozen E4c pool really is separated.
  (f) the seed replicate runs on fold 0's exact validation indices.
  (g) resume cannot launder a different configuration: config_id,
      data_rules_id and matrix version each individually refused.
  (h) the TML vectoriser is fitted on the fold's fit slice alone.
  (i) the partition is a deterministic function of the pool, and the pool is a
      deterministic function of the files.
  (j) class ordering is the same everywhere, and the store's `classes` match
      the column order of `proba` in every artefact.

Written as a standalone script for the same reason as its siblings: pytest is
not a project dependency and adding one would rewrite uv.lock, which the GPU
machine consumes with `uv sync --frozen`.

This file holds the checks of the data, code and stored artefacts. Checks of
the thesis text and of the development infrastructure are not part of this
repository.

Run: .venv/bin/python -m tests.test_data_integrity
"""
from __future__ import annotations

import ast
import inspect
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import e2_runner, harness, tml_components
from src.component_store import (ComponentStore, compute_config_id,
                                 compute_data_rules_id, load_matrix)
from src.data_loading import load_split
from src.e2_runner import load_pool, run_component, run_seed_replicate
from src.harness import (apply_data_quality_rules, evaluate_predictions,
                         load_fold_checkpoint, make_cv_splits, resolve_classes,
                         save_fold_checkpoint)

SUBTASKS = ("c2a", "dbo", "vio")
EDITION = "2025"
N_SPLITS = 5
SEEDS = (42, 43, 44)

_PASS, _FAIL = 0, 0
_NOTES: list[str] = []


def ok(msg):
    global _PASS
    _PASS += 1
    print("  \033[32mPASS\033[0m  " + msg)


def bad(msg):
    global _FAIL
    _FAIL += 1
    print("  \033[31mFAIL\033[0m  " + msg)


def note(msg):
    _NOTES.append(msg)
    print("        " + msg)


# The key the rule itself uses (harness._drop_within_edition_duplicates and
# _drop_train_test_overlap both normalise this way, and nothing else does).
def rule_key(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower()


# Deliberately STRICTER than the rule: internal whitespace collapsed as well.
# A pair the rule misses and this one catches is a genuine near-duplicate that
# can sit in a fold's training and validation part at the same time.
def strict_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


_POOLS: dict[str, pd.DataFrame] = {}


def pool(subtask: str) -> pd.DataFrame:
    if subtask not in _POOLS:
        _POOLS[subtask] = load_pool(subtask, EDITION)
    return _POOLS[subtask]


# ---------------------------------------------------------------------------
# (i) determinism of the pool and of the partition
# ---------------------------------------------------------------------------

def check_i_pool_and_partition_are_deterministic():
    """The campaign runs on Linux, every committed artefact was made on Darwin,
    and E3 pools them -- so the partition has to be a function of the data and
    nothing else.

    StratifiedKFold is NOT invariant to row order, and it does not have to be:
    what must hold is that the pool's row order is itself a deterministic
    function of the files. Both halves are checked, because only the pair gives
    the property. The row order is exercised explicitly rather than assumed:
    a shuffled frame MUST produce a different partition, so a future rule that
    reorders rows (a groupby, a sort, a merge) can never pass unnoticed."""
    for st in SUBTASKS:
        df = pool(st)
        again = load_pool(st, EDITION)
        assert df["id"].tolist() == again["id"].tolist(), \
            "{}: load_pool is not deterministic in row order".format(st)
        assert df["label"].astype(str).tolist() == again["label"].astype(str).tolist(), st

        for seed in SEEDS:
            a = make_cv_splits(df, N_SPLITS, seed)
            b = make_cv_splits(again, N_SPLITS, seed)
            for (ta, va), (tb, vb) in zip(a, b):
                assert np.array_equal(ta, tb) and np.array_equal(va, vb), \
                    "{} seed {}: the partition moved between two calls".format(st, seed)

        shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        by_id_ordered = [set(df["id"].values[v]) for _, v in make_cv_splits(df, N_SPLITS, 42)]
        by_id_shuffled = [set(shuffled["id"].values[v])
                          for _, v in make_cv_splits(shuffled, N_SPLITS, 42)]
        assert by_id_ordered != by_id_shuffled, (
            "{}: a shuffled frame produced the SAME partition. StratifiedKFold "
            "is order-dependent, so this means the test is not exercising what "
            "it claims -- or load_pool has started sorting.".format(st))
    ok("pool row order and the partition are deterministic (and order-sensitive, as expected)")


# ---------------------------------------------------------------------------
# (a) out-of-fold construction
# ---------------------------------------------------------------------------

def check_a_stored_entries_are_well_formed():
    store = ComponentStore()
    keys = store.list_runs()
    assert keys, "the component store is empty -- nothing to verify"
    from src.component_store import load_matrix, parse_run_key
    for key in keys:
        rec = store.load(key)
        meta = rec["meta"]
        e4c = parse_run_key(key)["variant"] == "e4c"
        if e4c:
            # A 2025-trained model scored ONCE on the frozen 2026 pool: its ids are
            # that pool in order, its gold labels are mapped into the 2025 label
            # set, and there are no folds -- every item carries -1.
            from src.e4c_runner import e4c_protocol, eval_pool, map_labels
            df, _ = eval_pool(meta["subtask"])
            classes25 = resolve_classes(load_pool(meta["subtask"], "2025")["label"].values)
            df = df.assign(label=[str(v) for v in map_labels(
                meta["subtask"], df["label"].values, classes25, e4c_protocol(load_matrix()))])
        else:
            df = pool(meta["subtask"]) if meta["edition"] == EDITION else load_pool(
                meta["subtask"], meta["edition"])
        ids = np.asarray(rec["ids"])
        pool_ids = np.asarray(df["id"].values, dtype=np.int64)

        assert len(set(ids.tolist())) == len(ids), "{}: duplicate ids".format(key)
        assert np.array_equal(ids, pool_ids), \
            "{}: stored ids are not the pool, in pool order".format(key)

        want = {int(i): str(l) for i, l in zip(pool_ids, df["label"].values)}
        assert all(want[int(i)] == y for i, y in zip(ids, rec["y_true"])), \
            "{}: y_true does not match the pool's label for that id".format(key)

        rows = rec["proba"].sum(axis=1)
        assert np.allclose(rows, 1.0, atol=1e-5), \
            "{}: probability rows do not sum to 1 (min {:.6f})".format(key, rows.min())

        fold = np.asarray(rec["fold"])
        if e4c:
            assert (fold == -1).all(), "{}: an E4c entry claims a fold".format(key)
            continue
        assert (fold >= 0).all(), "{}: fold holds -1, i.e. unscored items".format(key)
        expected = np.full(len(df), -1, dtype=np.int16)
        for f, (_, val_idx) in enumerate(make_cv_splits(df, meta["n_splits"], meta["seed"])):
            expected[val_idx] = f
        assert np.array_equal(fold, expected), \
            "{}: the stored fold column does not reproduce make_cv_splits".format(key)
        counts = np.bincount(fold, minlength=meta["n_splits"])
        assert (counts > 0).all() and counts.sum() == len(df), key
    ok("{} stored entries: ids, labels, rows, folds all reproduce from the pool".format(len(keys)))


class _FoldStampComponent:
    """A stand-in whose probabilities ENCODE the fold that produced them.

    This is the only way to answer the question that matters here. `e2_runner`
    computes `fold_of_item` in a loop of its own, separate from the one
    `run_cv` uses to place rows into the out-of-fold matrix. Both call
    `make_cv_splits`, so they agree -- but nothing checked it, and a divergence
    would be invisible: the shapes match, the metrics are computed from the
    matrix, and only the store's `fold` column would be wrong. Chapter 4's
    per-fold reading and any fold-aware E3 analysis rest on it.

    So the stand-in writes the fold index into the probability itself, and the
    check reads it back out of the STORE and compares it with the store's own
    fold column."""

    def __init__(self, fold_idx, classes):
        self.fold_idx = fold_idx
        self.classes_ = list(classes)
        self.record = {"selected_params": {"fold_stamp": fold_idx},
                       "temperature": {"temperature": 1.0},
                       "runtime_seconds": 0.0}


def check_a_fold_of_item_agrees_with_the_placement_run_cv_used(tmp: Path):
    df = pool("dbo").head(300).copy()
    df["label"] = df["label"].astype(str)
    classes = resolve_classes(df["label"].values)
    n_splits = 5
    state = {"fold": -1}

    def make_train_fn(cls, seed):
        def train_fn(train_df):
            state["fold"] += 1
            return _FoldStampComponent(state["fold"], cls)
        return train_fn

    def predict_proba_fn(model, val_df, cls):
        # column 0 carries (fold + 1) / 100 -- recoverable to the exact integer
        p = np.zeros((len(val_df), len(cls)))
        p[:, 0] = (model.fold_idx + 1) / 100.0
        p[:, 1:] = (1.0 - p[:, 0:1]) / (len(cls) - 1)
        return p

    stub = {
        "family": "tml", "seeds": (42,), "replicate_seeds": (),
        "make_train_fn": make_train_fn, "predict_proba_fn": predict_proba_fn,
        "meta": {"grid": [{"fold_stamp": True}], "inner_split": {},
                 "seed_semantics": "stand-in"},
    }
    original_pool, original_api = e2_runner.load_pool, e2_runner.component_api
    e2_runner.load_pool = lambda st, ed: df
    e2_runner.component_api = lambda component, **kw: stub
    try:
        store = ComponentStore(root=tmp / "store")
        run_component("dbo", EDITION, "tml_svm", seeds=(42,), n_splits=n_splits,
                      bootstrap_n=20, store=store, out_dir=tmp / "out",
                      ckpt_root=tmp / "ck")
    finally:
        e2_runner.load_pool, e2_runner.component_api = original_pool, original_api

    rec = store.load("dbo_{}/tml_svm__seed42".format(EDITION))
    recovered = np.rint(rec["proba"][:, 0] * 100).astype(int) - 1
    stored_fold = np.asarray(rec["fold"]).astype(int)
    assert np.array_equal(recovered, stored_fold), (
        "the fold that PRODUCED each row and the fold the store RECORDS for it "
        "disagree on {} of {} items".format(
            int((recovered != stored_fold).sum()), len(stored_fold)))
    assert set(stored_fold.tolist()) == set(range(n_splits))
    ok("e2_runner's fold_of_item loop matches the placement run_cv actually used")


# ---------------------------------------------------------------------------
# (b) train/eval separation inside a fold
# ---------------------------------------------------------------------------

def check_b_no_training_text_is_in_its_own_folds_validation_part():
    """`run_cv` applies the overlap rule per fold with drop_overlap=True.

    Two numbers are reported and only one is asserted tightly. Under the rule's
    OWN key the residual must be exactly zero -- and it is, for a reason worth
    stating: the pool is already deduplicated by that same key, so the per-fold
    rule drops nothing and is a belt-and-braces check rather than a working
    filter. Under a stricter key (internal whitespace collapsed) a handful of
    pairs survive, and those are the leak the rule cannot see. The bound is
    generous but finite, so a data change that introduces real near-duplicate
    leakage fails here instead of appearing in a chapter."""
    worst = 0.0
    for st in SUBTASKS:
        df = pool(st)
        for seed in SEEDS:
            for f, (tr, va) in enumerate(make_cv_splits(df, N_SPLITS, seed)):
                train_fold = df.iloc[tr].copy().reset_index(drop=True)
                val_fold = df.iloc[va].copy().reset_index(drop=True)
                cleaned, _ = apply_data_quality_rules(
                    train_fold, val_fold, dedup=True, drop_overlap=True)
                assert set(cleaned["id"]).isdisjoint(set(val_fold["id"])), \
                    "{} seed {} fold {}: an id is in both parts".format(st, seed, f)
                residual = len(set(rule_key(cleaned["description"]))
                               & set(rule_key(val_fold["description"])))
                assert residual == 0, (
                    "{} seed {} fold {}: {} texts are in the training AND the "
                    "validation part under the rule's own key".format(
                        st, seed, f, residual))
                strict = len({strict_key(t) for t in cleaned["description"]}
                             & {strict_key(t) for t in val_fold["description"]})
                worst = max(worst, strict / len(val_fold))

    # The per-fold figure above is a consequence, not the invariant. What
    # bounds every fold at once is the number of near-duplicate texts the POOL
    # still holds under the stricter key -- a pair that survives global
    # deduplication is exactly what can land on both sides of a split. Measured:
    # c2a 1, dbo 2, vio 1 extra rows, i.e. 0.01-0.03% of a pool.
    # The bound is 0.1%, three to seven times the measurement, and it is a
    # statement about the DATA rather than about a fold's luck.
    for st in SUBTASKS:
        df = pool(st)
        keys = [strict_key(t) for t in df["description"]]
        extra = len(keys) - len(set(keys))
        share = extra / len(df)
        assert share < 0.001, (
            "{}: {} rows ({:.3%} of the pool) are near-duplicates that the "
            "deduplication rule's own key does not see. Above 0.1% this stops "
            "being a handful of stray double spaces and becomes leakage the "
            "rule was written to prevent.".format(st, extra, share))
        note("{}: {} near-duplicate rows survive the rule's key ({:.4%} of the pool)"
             .format(st, extra, share))
    note("realised per-fold overlap: 0 under the rule's key in all {} folds; "
         "worst residual under a stricter key {:.4%} of a validation part"
         .format(len(SUBTASKS) * len(SEEDS) * N_SPLITS, worst))
    assert worst < 0.005, "per-fold near-duplicate residual reached {:.3%}".format(worst)
    ok("no training text sits in its own fold's validation part")


# ---------------------------------------------------------------------------
# (c) the three-way inner split
# ---------------------------------------------------------------------------

def check_c_inner_split_is_disjoint_exhaustive_and_stratified():
    inner = tml_components.INNER_SPLIT
    for st in SUBTASKS:
        df = pool(st)
        for f, (tr, va) in enumerate(make_cv_splits(df, N_SPLITS, 42)):
            train_fold = df.iloc[tr].copy().reset_index(drop=True)
            val_fold = df.iloc[va].copy().reset_index(drop=True)
            train_fold, _ = apply_data_quality_rules(
                train_fold, val_fold, dedup=True, drop_overlap=True)
            fit, sel, cal, stratified = tml_components._three_way_split(train_fold, 42)
            s_fit, s_sel, s_cal = (set(x["id"].tolist()) for x in (fit, sel, cal))
            n = len(train_fold)

            assert not (s_fit & s_sel) and not (s_fit & s_cal) and not (s_sel & s_cal), \
                "{} fold {}: the inner slices overlap".format(st, f)
            assert s_fit | s_sel | s_cal == set(train_fold["id"].tolist()), \
                "{} fold {}: the inner slices do not cover the training portion".format(st, f)
            assert len(fit) + len(sel) + len(cal) == n, "{} fold {}".format(st, f)
            for name, part, want in (("fit", fit, inner["fit"]),
                                     ("select", sel, inner["select"]),
                                     ("calibrate", cal, inner["calibrate"])):
                got = len(part) / n
                assert abs(got - want) < 0.002, \
                    "{} fold {}: {} slice is {:.4f} of the training portion, matrix says {}".format(
                        st, f, name, got, want)
            assert stratified, (
                "{} fold {}: the inner split fell back to UNSTRATIFIED. On the "
                "campaign pools it must not -- the thinnest class (DBO "
                "subversive) has enough members. inner_split_stratified would "
                "record it, but a campaign should never reach that state.".format(st, f))
            # every class present in the fold's training portion must reach
            # every slice, or the stratification claim is empty
            for part_name, part in (("fit", fit), ("select", sel), ("calibrate", cal)):
                missing = set(train_fold["label"].astype(str)) - set(part["label"].astype(str))
                assert not missing, "{} fold {}: {} slice misses class(es) {}".format(
                    st, f, part_name, sorted(missing))
            assert set(cal["id"]).isdisjoint(set(val_fold["id"])), \
                "{} fold {}: the calibration slice touches the validation part".format(st, f)
    ok("inner split: disjoint, exhaustive, 80/10/10, stratified, calibration unseen")


def check_c_the_unstratifiable_case_is_recorded_rather_than_hidden():
    """What happens when stratification CANNOT hold, which is the half that a
    campaign never reaches and a thin E4a/E4c cell will."""
    df = pd.DataFrame({
        "id": list(range(30)),
        "description": ["text {}".format(i) for i in range(30)],
        "label": ["a"] * 27 + ["b", "b", "c"],       # 'c' has one member
    })
    fit, sel, cal, stratified = tml_components._three_way_split(df, 42)
    assert not stratified, "a class with one member must disable stratification"
    assert len(fit) + len(sel) + len(cal) == len(df)
    assert set(fit["id"]) | set(sel["id"]) | set(cal["id"]) == set(df["id"])
    ok("an unstratifiable fold falls back cleanly and says so (inner_split_stratified)")


# ---------------------------------------------------------------------------
# (d) no test data anywhere in E2
# ---------------------------------------------------------------------------

def check_d_no_official_test_row_is_in_an_e2_pool():
    """The pool is built with drop_overlap=False ON PURPOSE, and the reason is
    that E2 never scores anything on the official test split. That reason is
    re-measured here rather than quoted: no test ITEM (id) is in a pool, while
    a measured number of test TEXTS are -- which is exactly what the rule being
    off means, and what §3.3 has to state."""
    matrix_rules = load_matrix()["encoder_protocol"]["pool_rules"]
    assert matrix_rules["drop_train_test_overlap"] is False, \
        "the matrix now asks for the overlap rule; load_pool does not apply it"
    assert matrix_rules["within_edition_dedup"] is True

    src = inspect.getsource(load_pool)
    assert "drop_overlap=False" in src, \
        "load_pool no longer states drop_overlap=False -- the pool changed"

    for st in SUBTASKS:
        p = pool(st)
        test25 = load_split(st, EDITION, "test")
        shared_ids = set(p["id"].tolist()) & set(test25["id"].tolist())
        assert not shared_ids, (
            "{}: {} rows of the 2025 TEST split are in the E2 pool by id. E2 "
            "must never evaluate on the official test set.".format(
                st, len(shared_ids)))
        overlap = int(rule_key(p["description"]).isin(
            set(rule_key(test25["description"]))).sum())
        note("{}: {} pool rows share a TEXT with 2025-test ({:.2f}% of the pool) "
             "-- kept deliberately, the rule is off".format(
                 st, overlap, 100 * overlap / len(p)))
    ok("no official 2025 test row is inside any E2 pool (ids disjoint, all subtasks)")


def check_d_the_2026_test_split_is_not_an_evaluation_source_anywhere():
    """⚠️ MEASURED, and it is not zero. Ids are stable across
    editions and identify the same text, so the 2026 test split can be checked
    against the 2025 pool by id -- and for VIO it overlaps heavily: a large
    share of VIO-2026-test consists of items that are in our 2025 VIO training
    pool, re-annotated into the six-way scheme. C2A and DBO are clean.

    That is harmless only as long as nothing evaluates on 2026-test, which is
    the design (dimension3 evaluates on a fresh 2026-TRAIN holdout, and E4c
    freezes its pool from 2026-train minus 2025-train). It is the mirror of
    the cross-edition overlap finding and it is asserted rather than remembered, because the day
    someone reaches for `load_split(st, "2026", "test")` as the cross-edition
    evaluation set, the number would be inflated for the worst possible reason
    and nothing else in the tree would notice."""
    import ast

    matrix = load_matrix()
    assert matrix["dimension3"]["vio_eval_source"] == "fresh 2026-train holdout", \
        "dimension3.vio_eval_source changed -- re-check the contamination below"
    assert matrix["dimension3"]["retraining_on_2026"] is False

    for st in SUBTASKS:
        p = pool(st)
        test26 = load_split(st, "2026", "test")
        shared = set(p["id"].tolist()) & set(test26["id"].tolist())
        note("{}: {} of {} 2026-test items ({:.1f}%) are in the 2025 E2 training "
             "pool by id".format(st, len(shared), len(test26),
                                 100 * len(shared) / len(test26)))

    # nothing in src/ may read the 2026 test split
    root = Path(harness.__file__).resolve().parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", getattr(node.func, "id", "")) == "load_split"):
                continue
            args = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if "2026" in args and "test" in args:
                offenders.append("{}:{}".format(path.name, node.lineno))
    assert not offenders, (
        "src/ reads the 2026 TEST split at {}. For VIO that split overlaps the "
        "2025 training pool heavily, so any evaluation on it is contaminated. "
        "Dimension 3 evaluates on a fresh 2026-TRAIN holdout.".format(offenders))
    ok("the 2026 test split is read nowhere in src/, and the design does not use it")


# ---------------------------------------------------------------------------
# (e) cross-edition contamination
# ---------------------------------------------------------------------------

def check_e_cross_edition_rule_is_off_outside_e4c():
    """A silent True would move every E1/E2/E3 number, so this asks the source
    tree rather than one call site."""
    sig = inspect.signature(apply_data_quality_rules)
    assert sig.parameters["drop_cross_edition"].default is False, \
        "apply_data_quality_rules now defaults drop_cross_edition to True"

    root = Path(harness.__file__).resolve().parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords or []:
                if kw.arg == "drop_cross_edition" and not (
                        isinstance(kw.value, ast.Constant) and kw.value.value is False):
                    offenders.append(path.name)
    assert offenders == ["e4c_pool.py"], (
        "drop_cross_edition is switched on outside src/e4c_pool.py: {}. It is "
        "the Dimension-3 rule and belongs to E4c alone.".format(
            sorted(set(offenders))))
    ok("drop_cross_edition is True in src/e4c_pool.py and nowhere else")


def check_e_the_frozen_e4c_pool_really_is_separated():
    import json

    from src import e4c_pool

    artefact = json.loads(e4c_pool.ARTEFACT.read_text(encoding="utf-8"))
    for stored in artefact["pools"]:
        st = stored["subtask"]
        rebuilt = e4c_pool.build_pool(st)
        assert rebuilt["ids_sha256"] == stored["ids_sha256"], (
            "{}: the E4c pool no longer re-derives to the frozen hash -- it was "
            "frozen so that no later choice could be made with knowledge of "
            "it".format(st))
        train26, _ = apply_data_quality_rules(
            load_split(st, "2026", "train"), dedup=True, drop_overlap=False)
        keep = train26[train26["id"].astype(str).isin(set(stored["ids"]))]
        residual = len(set(rule_key(keep["description"]))
                       & set(rule_key(pool(st)["description"])))
        assert residual == 0, (
            "{}: {} texts of the frozen E4c evaluation pool are still inside "
            "the 2025 training pool".format(st, residual))
    ok("the frozen E4c pool re-derives exactly and holds no 2025 training text")


# ---------------------------------------------------------------------------
# (f) the seed replicate
# ---------------------------------------------------------------------------

def check_f_the_replicate_runs_on_fold_zeros_own_validation_indices():
    """Nothing in the checkpoint's consistency checks can catch a mix-up here,
    because the validation indices are identical BY DESIGN -- which is why the
    model seed goes into the fingerprint instead. So the identity is proved by
    executing both paths and comparing the ids they actually scored."""
    df = pool("c2a").head(400).copy()
    df["label"] = df["label"].astype(str)
    classes = resolve_classes(df["label"].values)
    seen = {}

    def make_train_fn(cls, seed):
        def train_fn(train_df):
            seen["train_ids"] = set(train_df["id"].tolist())
            seen["model_seed"] = seed
            m = _FoldStampComponent(0, cls)
            m.record = {"selected_params": {"seed": seed}, "selected_epoch": 1,
                        "fit_diagnosis": None, "runtime_seconds": 0.0,
                        "temperature": {"temperature": 1.0}}
            return m
        return train_fn

    def predict_proba_fn(model, val_df, cls):
        seen["val_ids"] = val_df["id"].tolist()
        return np.full((len(val_df), len(cls)), 1.0 / len(cls))

    api = {"make_train_fn": make_train_fn, "predict_proba_fn": predict_proba_fn}
    rep = run_seed_replicate(df, "encoder_1b", classes, api=api, model_seed=44,
                             n_splits=N_SPLITS)

    train_idx, val_idx = make_cv_splits(df, N_SPLITS, e2_runner.PRIMARY_SEED)[
        e2_runner.ENCODER_FIXED_FOLD]
    assert rep["partition_seed"] == e2_runner.PRIMARY_SEED, rep["partition_seed"]
    assert rep["fold"] == e2_runner.ENCODER_FIXED_FOLD, rep["fold"]
    assert rep["model_seed"] == 44 and seen["model_seed"] == 44, \
        "only the TRAINING seed may differ; the replicate trained under {}".format(
            seen["model_seed"])
    assert seen["val_ids"] == df["id"].values[val_idx].tolist(), (
        "the replicate scored different items than fold {} of the main run".format(
            e2_runner.ENCODER_FIXED_FOLD))
    # the training portion may only differ from fold 0's by the per-fold rules
    fold0_train = df.iloc[train_idx].copy().reset_index(drop=True)
    fold0_train, _ = apply_data_quality_rules(
        fold0_train, df.iloc[val_idx].copy().reset_index(drop=True),
        dedup=True, drop_overlap=True)
    assert seen["train_ids"] == set(fold0_train["id"].tolist()), \
        "the replicate trained on a different training portion than fold 0"
    ok("the D3 replicate is fold 0 of the primary partition with only the training seed changed")


# ---------------------------------------------------------------------------
# (g) resume cannot launder a different configuration
# ---------------------------------------------------------------------------

def check_g_each_identity_field_is_refused_on_its_own(tmp: Path):
    """tests/test_fold_resume.py covers seed, split count, class ordering and
    the validation indices. The three fields folded into the fingerprint --
    config_id, data_rules_id and the matrix version -- were only ever tested as
    one opaque string, so a fingerprint that silently stopped carrying one of
    them would have passed. Each is varied on its own here, built the way
    e2_runner builds it."""
    component, subtask = "encoder_1b", "dbo"
    fields = {
        "component": component, "subtask": subtask, "edition": EDITION,
        "config_id": compute_config_id(component),
        "data_rules_id": compute_data_rules_id(),
        "matrix_version": str(load_matrix()["meta"]["version"]),
    }
    order = ["component", "subtask", "edition", "config_id", "data_rules_id",
             "matrix_version"]
    real = "|".join(fields[k] for k in order)

    classes = ["agitation", "criticism", "nothing", "subversive"]
    val_idx = np.arange(10)
    proba = np.full((10, 4), 0.25)
    ck = tmp / "ck"
    save_fold_checkpoint(ck, 0, fingerprint=real, seed=42, n_splits=5,
                         classes=classes, val_idx=val_idx, fold_proba=proba,
                         payload={"ok": True})

    restored = load_fold_checkpoint(ck, 0, fingerprint=real, seed=42, n_splits=5,
                                    classes=classes, val_idx=val_idx)
    assert restored is not None and np.array_equal(restored[0], proba), \
        "the legitimate resume did not restore"

    def refused(**overrides):
        f = dict(fields)
        f.update({k: v for k, v in overrides.items() if k in fields})
        fp = "|".join(f[k] for k in order)
        kw = dict(fingerprint=fp, seed=42, n_splits=5, classes=classes,
                  val_idx=val_idx)
        kw.update({k: v for k, v in overrides.items() if k not in fields})
        try:
            load_fold_checkpoint(ck, 0, **kw)
        except RuntimeError:
            return True
        return False

    cases = {
        "config_id": dict(config_id="deadbeefcafe"),
        "data_rules_id": dict(data_rules_id="0" * 12),
        "matrix version": dict(matrix_version="9.9"),
        "seed": dict(seed=43),
        "split count": dict(n_splits=3),
        "class ordering": dict(classes=["criticism", "agitation", "nothing", "subversive"]),
        "validation indices": dict(val_idx=np.arange(1, 11)),
    }
    for name, override in cases.items():
        assert refused(**override), \
            "a checkpoint with a different {} was REUSED instead of refused".format(name)

    # and the legitimate one still restores after all that
    again = load_fold_checkpoint(ck, 0, fingerprint=real, seed=42, n_splits=5,
                                 classes=classes, val_idx=val_idx)
    assert again is not None and np.array_equal(again[0], proba)
    ok("all seven identity mismatches are refused; a legitimate resume still restores")


def check_g_the_fingerprint_e2_runner_builds_carries_all_three_identities():
    """The refusals above are worth nothing if the string the runner actually
    builds has stopped containing one of the fields. Read from the source of
    run_component rather than reconstructed here, so the two cannot drift."""
    src = inspect.getsource(run_component)
    body = src[src.index("fingerprint = "):src.index("for seed in seeds:")]
    for token in ("component", "subtask", "config_id", "data_rules_id",
                  'matrix_meta["version"]'):
        assert token in body, (
            "run_component's checkpoint fingerprint no longer carries {} -- a "
            "resume could then reuse folds from another configuration".format(token))
    ok("run_component's fingerprint still carries component, subtask, config_id, "
       "data_rules_id and the matrix version")


# ---------------------------------------------------------------------------
# (h) feature fitting
# ---------------------------------------------------------------------------

def check_h_the_vectoriser_is_fitted_on_the_fit_slice_alone():
    """Verified by execution, not by reading build_vectoriser's call site: a
    marker token is planted in the select, calibrate and validation parts and
    must be absent from the fitted vocabulary."""
    df = pool("c2a").head(600).copy()
    df["label"] = df["label"].astype(str)
    classes = resolve_classes(df["label"].values)
    train_idx, val_idx = make_cv_splits(df, N_SPLITS, 42)[0]
    train_fold = df.iloc[train_idx].copy().reset_index(drop=True)
    val_fold = df.iloc[val_idx].copy().reset_index(drop=True)
    train_fold, _ = apply_data_quality_rules(train_fold, val_fold, dedup=True,
                                             drop_overlap=True)
    fit, sel, cal, _ = tml_components._three_way_split(train_fold, 42)

    marker_sel, marker_cal, marker_val = "zzselectmarker", "zzcalibmarker", "zzvalmarker"
    tagged = train_fold.copy()
    tag = {i: marker_sel for i in sel["id"]}
    tag.update({i: marker_cal for i in cal["id"]})
    tagged["description"] = [
        "{} {}".format(t, tag[i]) if i in tag else t
        for t, i in zip(tagged["description"], tagged["id"])]
    val_tagged = val_fold.copy()
    val_tagged["description"] = val_tagged["description"] + " " + marker_val

    model = tml_components.make_train_fn("tml_svm", classes, seed=42, n_jobs=1)(tagged)
    vocab = set()
    for _, sub in model.vectoriser_.transformer_list:
        vocab |= set(sub.vocabulary_)
    for marker in (marker_sel, marker_cal, marker_val):
        assert marker not in vocab, (
            "the TF-IDF vocabulary contains {!r} -- the vectoriser saw a slice "
            "it must never be fitted on".format(marker))
    # ... and it does contain fit-slice vocabulary, so the check is not vacuous
    fit_words = set()
    for text in fit["description"].head(50):
        fit_words |= {w.lower() for w in str(text).split() if len(w) > 6}
    assert fit_words & vocab, "the vocabulary shares nothing with the fit slice"
    proba = model.predict_proba(val_tagged)
    assert proba.shape == (len(val_tagged), len(classes))
    assert np.allclose(proba.sum(axis=1), 1.0)
    ok("the TF-IDF vectoriser is fitted on the fold's fit slice only")


# ---------------------------------------------------------------------------
# (j) class ordering
# ---------------------------------------------------------------------------

def check_j_class_ordering_is_the_same_everywhere():
    """A transposed class order is silent, survives every shape check and
    inverts a per-class F1 table."""
    for st in SUBTASKS:
        df = pool(st)
        classes = resolve_classes(df["label"].values)
        assert classes == sorted(set(df["label"].values), key=str)
        assert [str(c) for c in classes] == sorted({str(c) for c in df["label"].values}), \
            "{}: resolve_classes is not sorted by string form".format(st)
        # order must not depend on row order or on the fold
        assert resolve_classes(df.sample(frac=1.0, random_state=3)["label"].values) == classes
        for _, val_idx in make_cv_splits(df, N_SPLITS, 42):
            sub = resolve_classes(df["label"].values[val_idx])
            assert sub == [c for c in classes if c in set(sub)], \
                "{}: a fold's class ordering is not a subsequence of the global one".format(st)

    store = ComponentStore()
    per_subtask: dict[str, list] = {}
    for key in store.list_runs():
        rec = store.load(key)
        meta = rec["meta"]
        from src.component_store import parse_run_key
        if parse_run_key(key)["variant"] == "e4c":
            # An E4c entry is a 2025-trained model scored on 2026: its classes
            # are the 2025 label set (VIO mapped to binary), not 2026's.
            df = load_pool(meta["subtask"], "2025")
        else:
            df = pool(meta["subtask"]) if meta["edition"] == EDITION else load_pool(
                meta["subtask"], meta["edition"])
        want = [str(c) for c in resolve_classes(df["label"].values)]
        assert rec["classes"] == want, \
            "{}: stored classes {} != {}".format(key, rec["classes"], want)
        assert rec["meta"]["classes"] == want, \
            "{}: the sidecar's classes disagree with the .npz".format(key)
        assert rec["proba"].shape[1] == len(want), key
        # the column order IS the class order: argmax must reproduce the
        # prediction the artefact's own metrics were computed from
        pred = np.asarray([want[i] for i in rec["proba"].argmax(axis=1)])
        m = evaluate_predictions(np.asarray(rec["y_true"]), pred, labels=want,
                                 bootstrap=False)
        stored_score = rec["meta"].get("pooled_macro_f1")
        if stored_score is not None:
            assert abs(m["macro_f1"] - stored_score) < 1e-4, (
                "{}: recomputing macro-F1 from proba's column order gives {:.6f}, "
                "the sidecar says {:.6f} -- the columns are not in `classes` "
                "order".format(key, m["macro_f1"], stored_score))
        # E4c entries score the frozen 2026 EVAL pool (2026 train minus every 2025
        # train text), a different item set from an E4a dbo/2026 cell, which runs
        # on the whole deduplicated 2026 train pool -- so they align with each
        # other, never with E4a. Grouping them together would be a false alarm.
        group = "{}_{}".format(meta["subtask"], meta["edition"])
        if parse_run_key(key)["variant"] == "e4c":
            group += "/e4c"
        per_subtask.setdefault(group, []).append((key, np.asarray(rec["ids"])))

    # E3 aligns components item by item; two components of one subtask must
    # therefore carry the same ids in the same order.
    for group, entries in per_subtask.items():
        first_key, first_ids = entries[0]
        for key, ids in entries[1:]:
            assert np.array_equal(ids, first_ids), (
                "{} and {} carry different id vectors -- E3 would align two "
                "components row by row on different items".format(first_key, key))
    ok("class ordering and id vectors agree across pools, folds and every stored artefact")


# ---------------------------------------------------------------------------
# The artefacts themselves: two things that are silent when they go wrong
# ---------------------------------------------------------------------------

def check_k_every_fitted_temperature_is_interior_to_its_bounds():
    """`fit_temperature` optimises T inside hard bounds and returns
    `fitted: False` on a degenerate slice. Both failures are silent in the
    reported number -- a T pinned to a bound still produces an ECE, and an
    unfitted T of 1.0 is indistinguishable from a well-calibrated component.
    The interface hands these confidences to the cascade and the judge, so they are worth
    a standing check rather than a one-off look."""
    import json

    from src.calibration import _T_BOUNDS

    lo, hi = _T_BOUNDS
    temps = []
    for path in sorted(Path("results/e2").glob("e2_*.json")) + \
            sorted(Path("results/measurements").glob("e2_*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        for rep in run.get("repeats", []):
            for i, d in enumerate(rep.get("folds_detail", [])):
                t = d["temperature"]
                assert t["fitted"] is True, (
                    "{} seed {} fold {}: the temperature was NOT fitted ({}). "
                    "A degenerate calibration slice leaves T at 1.0 and the ECE "
                    "beside it means nothing.".format(
                        path.name, rep["seed"], i, t.get("note")))
                assert t["nll_after"] <= t["nll_before"] + 1e-12, \
                    "{} seed {} fold {}: calibration made the likelihood worse".format(
                        path.name, rep["seed"], i)
                temps.append((t["temperature"], path.name))
    assert temps, "no artefact carried a temperature"
    edge = [(t, f) for t, f in temps if t <= lo * 1.05 or t >= hi * 0.95]
    assert not edge, (
        "{} fitted temperatures sit on the optimiser bound {}: {}. A bounded "
        "optimum is not a calibration, it is a truncation.".format(
            len(edge), _T_BOUNDS, edge[:3]))
    note("{} fitted temperatures, range {:.3f}-{:.3f}, bounds {} -- all interior"
         .format(len(temps), min(t for t, _ in temps), max(t for t, _ in temps),
                 _T_BOUNDS))
    ok("every stored temperature is fitted, improves the NLL, and is interior")


def check_l_the_selection_diagnosis_reads_every_stored_artefact():
    """Not an assertion about the verdicts -- those are findings, and pinning
    them here would restate a constant beside the artefacts that decide it.
    What is asserted is that the check RUNS on every artefact and returns a
    complete shape; the verdicts are printed so a reader of the suite sees the
    current state of the grids."""
    import json

    from src.e2_runner import selection_diagnosis_for

    paths = sorted(Path("results/e2").glob("e2_*.json")) + \
        sorted(Path("results/measurements").glob("e2_*.json"))
    assert paths, "no E2 artefacts to read"
    flagged = 0
    single_cell = 0
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        diag = selection_diagnosis_for(run)
        assert diag is not None, path.name
        for key in ("flags", "notes", "axes", "n_distinct_winners", "n_fits",
                    "margin", "ok"):
            assert key in diag, "{}: diagnosis is missing {}".format(path.name, key)
        assert len(diag["flags"]) == len(diag["notes"]), (
            "{}: {} flags against {} notes -- every flag must carry its "
            "explanation".format(path.name, len(diag["flags"]),
                                 len(diag["notes"])))
        # ⚠️ `n_fits > 0` held only while every stored artefact came from a
        # family that SEARCHES a grid. The few-shot arm does not: its grid is a
        # single cell, nothing is selected, and n_fits == 0 is the CORRECT
        # answer, not a broken diagnosis. Both halves are asserted, each against
        # the shape its own family produces.
        if "single_cell_grid" in diag["flags"]:
            single_cell += 1
            assert diag["n_fits"] == 0 and diag["axes"] == {}, (
                "{}: flagged single_cell_grid but reports {} fits over axes {} "
                "-- a grid was searched after all, so the flag is wrong".format(
                    path.name, diag["n_fits"], sorted(diag["axes"])))
            assert diag["ok"], (
                "{}: a single-cell grid cannot sit on a grid edge, so `ok` "
                "must not be false".format(path.name))
        else:
            assert diag["n_fits"] > 0, (
                "{}: a grid-searching family reports 0 fits, so the selection "
                "diagnosis read nothing".format(path.name))
        edges = [f for f in diag["flags"] if f.startswith("grid_edge:")]
        if edges:
            flagged += 1
            note("{}: {} ({} distinct winners over {} fits)".format(
                path.name, ", ".join(edges), diag["n_distinct_winners"],
                diag["n_fits"]))
    note("{} of {} stored artefacts sit on a grid edge; {} have a single-cell "
         "grid and select nothing".format(flagged, len(paths), single_cell))
    ok("the grid-edge / selection check reads every stored artefact")


def check_r_env_pin_before_sentencebert_refuses_instead_of_segfaulting():
    """`env_pin()` imports torch and lightgbm; loading SentenceBERT afterwards
    pulls a SECOND OpenMP runtime into the process and the interpreter dies with
    SIGSEGV -- exit 139, no traceback, no artefact. env_pin's docstring says
    "call this AFTER all model runs", but a docstring is not a guard: a dict
    literal with `"env_pin": env_pin()` above `"runs": [...]` evaluates top to
    bottom, so the pin runs first and the C2A half dies.

    The refusal is what this asserts. A crash that names its cause costs a
    minute; one that does not cost this project an afternoon."""
    from unittest import mock

    from src import baselines as B
    from src import harness as H

    with mock.patch.object(H, "_ENV_PIN_CALLED", False):
        assert not H._ENV_PIN_CALLED
    # Not by calling env_pin() -- that would arm the flag for the rest of this
    # process and any later check that touches SentenceBERT would then refuse.
    with mock.patch.object(H, "_ENV_PIN_CALLED", True):
        try:
            B.C2ABaseline()._get_sbert()
            raise AssertionError(
                "SentenceBERT loaded after env_pin() with no refusal. On this "
                "machine that is a segmentation fault, not a warning")
        except RuntimeError as e:
            assert "env_pin" in str(e) and "OpenMP" in str(e), (
                "the refusal does not name the cause: {}".format(e))

    ok("SentenceBERT after env_pin() refuses with the reason")


def check_o_every_stored_run_still_matches_the_live_matrix():
    """A stored run's `config_id` is the identity of the protocol that produced
    its probabilities. `ComponentStore.load_many` compares the two and refuses a
    mismatch -- but only when something calls it, and the only caller is E3,
    which does not exist yet. So between an accidental matrix edit and the moment
    E3 is written, a stored arm can be silently orphaned with nothing failing.

    That gap is live right now. `encoder_protocol` still carries its decision
    record INSIDE the hashed block (v1.10 moved only llm_protocol's out, because
    moving the encoder's would move 7fc5d606bed9 and orphan the three encoder
    artefacts). Editing one word of that prose invalidates the whole encoder arm.

    This check closes it: every sidecar in the store, against the matrix as it
    stands. It is what makes 'the encoder keeps its notes' a safe decision rather
    than a hope."""
    import warnings

    from src.component_store import ComponentStore, parse_run_key

    store = ComponentStore()
    keys = sorted(str(p.parent.name + "/" + p.stem)
                  for p in Path("results/component_store").glob("*/*.json"))
    assert keys, "the component store is empty -- this check has nothing to guard"
    # ONE SET PER EXPERIMENT. load_many also
    # refuses a set with mixed fold counts, and E2 (5), E4a (3) and E4c (0) differ
    # by design. Loaded as one set, the first fetched E4a cell failed this check
    # for a reason that had nothing to do with the matrix. Every entry is still
    # checked against the live matrix, each inside its own experiment.
    groups: dict[str, list[str]] = {}
    for k in keys:
        variant = parse_run_key(k)["variant"]
        groups.setdefault("e2" if variant is None else variant.split("-")[0], []).append(k)
    with warnings.catch_warnings():
        # The machine spread (encoder on Linux, TML on Darwin) is by design and
        # warns rather than raises, deliberately. Only the
        # identity mismatch, which raises, is under test here.
        warnings.simplefilter("ignore")
        for experiment, group in sorted(groups.items()):
            try:
                store.load_many(group, check_generation=True)
            except RuntimeError as e:
                raise AssertionError(
                    "[{}] the store no longer matches e2_matrix.yaml, so a stored arm "
                    "has been orphaned by a matrix edit. Print every config_id before "
                    "and after the edit and compare.\n{}"
                    .format(experiment, e))
    ok("all {} stored runs ({}) still carry the identity the live matrix computes"
       .format(len(keys), ", ".join("{} {}".format(x, len(g))
                                    for x, g in sorted(groups.items()))))


# ---------------------------------------------------------------------------



def check_s_no_module_treats_a_help_probe_as_an_invocation():
    """`python -m src.X --help` must DESCRIBE X, never run it.

    A module that takes no arguments turns a `--help` probe into a real run:
    the probe *is* the invocation, and a committed artefact then survives only
    if the run happens not to reach its write.

    What is asserted, behaviourally rather than by reading the source, because
    the defect was invisible in the source:

    1. every `__main__` module answers `--help` inside `_HELP_TIMEOUT`;
    2. **the whole sweep changes not one byte of `results/`** -- the direct
       statement of "a probe does no work", whatever a module's argv handling
       looks like;
    3. every module that writes a *committed* artefact refuses a bare
       invocation instead of replacing it.

    The refusal is `harness.measuring_main_guard`, which is
    `e1_machine_baseline`'s lifted to one place so the next measuring
    module inherits it rather than re-deriving it. Note what is deliberately NOT
    required: a literal `usage:` line. `llm_components.__main__` prints the
    declared protocol and is inert, which is the property; demanding argparse
    everywhere would be demanding the mechanism instead of the behaviour.
    """
    import hashlib
    import subprocess

    _HELP_TIMEOUT = 60          # generous: the heaviest import here is torch

    def _fingerprint(paths):
        return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in paths if Path(p).is_file()}

    tracked = subprocess.run(["git", "ls-files", "results/"],
                             capture_output=True, text=True).stdout.split()
    # The judge cache is append-only CAMPAIGN STATE, written by a live process
    # over several days. Once committed it is tracked, and a campaign appending
    # to it during this sweep is not a --help probe doing work.
    # e3_combination's own --help is still covered: argparse exits before any
    # code that could open the cache.
    tracked = [t for t in tracked if not t.startswith("results/e3_judge/")]
    assert len(tracked) > 10, "git ls-files results/ returned {}".format(len(tracked))
    before = _fingerprint(tracked)

    src = Path("src")
    mains = sorted(p for p in src.glob("*.py")
                   if "__main__" in p.read_text(encoding="utf-8"))
    assert len(mains) >= 20, "src lost its __main__ modules: {}".format(len(mains))

    slow, silent = [], []
    for p in mains:
        mod = "src." + p.stem
        try:
            r = subprocess.run([sys.executable, "-m", mod, "--help"],
                               capture_output=True, text=True,
                               timeout=_HELP_TIMEOUT)
        except subprocess.TimeoutExpired:
            slow.append(mod)
            continue
        if not (r.stdout + r.stderr).strip():
            silent.append(mod)
    assert not slow, ("these modules RAN on a --help probe instead of "
                      "describing themselves: {}".format(", ".join(slow)))
    assert not silent, ("these modules answered --help with nothing at all, so "
                        "a caller cannot tell the probe was understood: "
                        "{}".format(", ".join(silent)))

    after = _fingerprint(tracked)
    changed = sorted(k for k in before if before[k] != after.get(k))
    assert not changed, (
        "a --help sweep MODIFIED committed artefacts, which means a probe is "
        "still an invocation: {}".format(", ".join(changed)))

    # (3) A module that writes a COMMITTED artefact has to say how it is
    # protected from replacing one by accident. Three answers are acceptable and
    # the table records which, because "guarded" is not one property: a refusal,
    # an explicit write flag, or writing nowhere real. The table is the claim and
    # the coverage assertion below is what keeps it honest -- a new writing
    # module fails this check until someone decides which column it is in.
    policy = {
        # refuses a bare invocation via harness.measuring_main_guard
        "verbaliser_probe.py": "refuses",
        "e1_platform_check.py": "refuses",
        "e1_machine_baseline.py": "refuses",   # its own refusal, the pattern the others copy
        "significance.py": "refuses",
        "significance_e4.py": "refuses",
        "multiplicity_by_experiment.py": "refuses",
        "b12_ablation.py": "refuses",
        "dimension2_selection.py": "refuses",
        "combination_headroom.py": "refuses",
        "e5_error_analysis.py": "refuses",
        "e5_irreducible.py": "refuses",
        # The drawn sample and its scoring rest on a human's hours of coding: a
        # re-draw or re-score must never silently replace what was read.
        "e5_label_sample.py": "refuses",
        "e5_label_analysis.py": "refuses",
        "e5_examples.py": "refuses",
        # argparse plus its own --force check, because the path is per-cell
        "lr_probe.py": "refuses",
        # bare call prints usage; only --go writes, and it refuses an existing
        # artefact without --force
        "e3_combination.py": "flag",
        "e3_combination_2026.py": "flag",
        # writes only on an explicit flag; a bare run reports or does nothing
        "data_manifest.py": "flag",          # --write
        "e4c_pool.py": "flag",               # --write
        "e4_config.py": "flag",              # --write; also refuses an existing freeze without --force
        "e4c_runner.py": "flag",             # --go; a bare call only describes the plan
        # writes an artefact the caller names, never a fixed committed path
        "data_profile.py": "named",
        "gwdg_smoke.py": "named",
        "verify_data_defects.py": "named",
    }

    tracked_names = {Path(t).name for t in tracked}
    writers = []
    for p in mains:
        body = p.read_text(encoding="utf-8")
        if not (".write_text(" in body or "json.dump(" in body):
            continue
        if not any(n in body for n in tracked_names):
            continue
        writers.append(p.name)

    uncovered = sorted(set(writers) - set(policy))
    assert not uncovered, (
        "these __main__ modules touch a committed artefact and this table does "
        "not say how they are protected from replacing one. Decide, then add "
        "the row -- refuses / flag / named / tmpdir: {}".format(
            ", ".join(uncovered)))
    # The reverse direction is NOT asserted, and the reason is a finding in its
    # own right: three of the rows above write through a format string
    # (`lr_probe_{}_{}_{}.json` and friends), so the substring detector cannot
    # see them at all. That is the more dangerous half of this hazard class --
    # the detector catches the writers it can name, and the table carries the
    # ones it cannot. What is asserted instead is that every row still exists.
    missing = sorted(n for n in policy if not (src / n).is_file())
    assert not missing, (
        "the table names modules that are gone: {}".format(", ".join(missing)))

    # The "refuses" column is the only one asserted behaviourally, because it is
    # the one that is easiest to lose. A bare invocation must SKIP, not measure.
    for name, kind in sorted(policy.items()):
        if kind != "refuses":
            continue
        r = subprocess.run([sys.executable, "-m", "src." + Path(name).stem],
                           capture_output=True, text=True, timeout=_HELP_TIMEOUT)
        out = r.stdout + r.stderr
        assert "SKIP" in out and "--force" in out, (
            "{} did not refuse a bare invocation; it printed: {!r}".format(
                name, out[:300]))

    after_bare = _fingerprint(tracked)
    changed = sorted(k for k in before if before[k] != after_bare.get(k))
    assert not changed, (
        "a bare invocation of a refusing module still rewrote: {}".format(
            ", ".join(changed)))

    ok("all {} __main__ modules describe themselves on --help, {} of them write "
       "a committed artefact and each says how it is protected, and the {} that "
       "refuse were seen refusing".format(
           len(mains), len(policy),
           sum(1 for v in policy.values() if v == "refuses")))


def main() -> int:
    checks = [(n, f) for n, f in sorted(globals().items())
              if n.startswith("check_") and callable(f)]
    print("\n=== data integrity ({} checks) ===".format(len(checks)))
    for name, fn in checks:
        needs_tmp = "tmp" in inspect.signature(fn).parameters
        d = Path(tempfile.mkdtemp()) if needs_tmp else None
        try:
            fn(d) if needs_tmp else fn()
        except AssertionError as e:
            bad("{}: {}".format(name[6:].replace("_", " "), e))
        except Exception as e:
            bad("{}: unexpected {}: {}".format(name[6:].replace("_", " "),
                                               type(e).__name__, e))
        finally:
            if d is not None:
                shutil.rmtree(d, ignore_errors=True)
    print("\n" + "=" * 50)
    print("  {} passed, {} failed".format(_PASS, _FAIL))
    print("=" * 50)
    return 1 if _FAIL else 0

def check_v_a_closed_gate_survives_its_own_refresh(tmp):
    """A gate decision is a JUDGEMENT; refreshing its evidence must not touch it.

    A notebook that resets `decision` to TODO on re-run silently reopens a
    closed gate. `write_g2_gate` rebuilds the measured half on every call --
    adding an arm is supposed to refresh the evidence and leave a closed gate
    closed -- and this asserts that.

    It also pins the OTHER half: `status`
    describes evidence COVERAGE, not the judgement (the gate state is
    `decision`, the convention G1 and the G1.5 smoke already use by carrying no
    status field at all). It had no value for "every arm has reported", so with
    `arms_pending` empty the file still said "in_progress" -- describing nothing
    true, and reading as a contradiction beside `decision: go` for anyone
    opening the file standalone.
    """
    from src.e2_summary import collect_runs, write_g2_gate

    live = Path("results/g2_gate_assessment.json")
    if not live.is_file():
        note("no g2_gate_assessment.json yet -- nothing to protect")
        return

    before = json.loads(live.read_text(encoding="utf-8"))
    scratch = Path(tmp) / "g2_gate_assessment.json"
    shutil.copy(live, scratch)

    runs = collect_runs()
    _, after = write_g2_gate(runs, path=scratch)

    assert after["decision"] == before["decision"], (
        "refreshing the G2 evidence CHANGED the decision, {!r} -> {!r}. A gate "
        "decision is a judgement and the refresh must carry it over "
        "untouched.".format(before["decision"], after["decision"]))
    assert after.get("decision_note") == before.get("decision_note"), (
        "refreshing the G2 evidence rewrote decision_note -- the reasoning a "
        "gate was closed on must survive its own evidence being updated")

    arms = set(after["arms_present"])
    expected = ("partial" if after["arms_present"] == ["tml"]
                else "complete" if {"encoder", "llm"} <= arms
                else "in_progress")
    assert after["status"] == expected, (
        "G2 status is {!r} but the arms present ({}) say it should be {!r}. "
        "status describes evidence coverage; a file that says in_progress with "
        "arms_pending empty describes nothing true.".format(
            after["status"], sorted(arms), expected))
    if after["decision"] != "TODO":
        assert not after["arms_pending"], (
            "G2 carries decision {!r} while arms {} have still not reported. It "
            "was pre-registered to be decided AFTER the LLM arm exists.".format(
                after["decision"], after["arms_pending"]))

    note("G2: decision {!r} survived a refresh; status {!r} matches arms {}".format(
        after["decision"], after["status"], sorted(arms)))
    ok("a closed gate survives its own evidence refresh")



if __name__ == "__main__":
    raise SystemExit(main())
