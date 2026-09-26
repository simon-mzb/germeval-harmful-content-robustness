"""
significance.py -- the selection-layer discipline, as a computation.

THE RULE
--------
Every cross-component or cross-strategy claim goes through McNemar
(`dietterich1998`) or non-overlapping bootstrap intervals; where the test is
negative the text says "indistinguishable" and does not crown the maximum
(§3.3). This module is where that rule is computed.

Nothing here needs a GPU or a re-run. `run_component` already stores the pooled
out-of-fold probabilities, the gold labels, the item ids and the fold index of
every run; two components evaluated on the same subtask therefore predicted the
SAME items in the SAME order, and the pairing McNemar needs is already on disk
(e.g. `dbo_2025/encoder_1b__seed42` and `dbo_2025/tml_svm__seed42` agree on all
7 210 ids, in order, with identical gold labels and fold assignments).

WHAT IS COMPUTED, AND WHY THREE NUMBERS RATHER THAN ONE
-------------------------------------------------------
1. **McNemar**, on the discordant pairs. `dietterich1998` recommends it for two
   classifiers on one test set precisely because it does not assume the two
   error rates are independent -- which they are not, having seen the same items.
   The chi-square form with continuity correction is used when the discordant
   count is large enough for it, and the **exact binomial** below that, because
   the approximation is unreliable on small b+c and the thin-class cells of this
   study are exactly where it would be used.

2. **The pre-registered interval rule**: whether the two components' own 95 %
   bootstrap intervals for macro-F1 overlap. Reported as written.

3. **The paired bootstrap of the DIFFERENCE**, reported beside it and flagged as
   the stronger test. ⚠️ These two can disagree, and when they do the
   disagreement is not a defect: non-overlapping intervals imply a real
   difference, but **overlapping intervals do NOT imply the absence of one** --
   the marginal intervals ignore the correlation between two systems scored on
   the same items, which is the very correlation McNemar exists to exploit. So
   the interval rule is CONSERVATIVE: it can call a real difference
   indistinguishable. `verdict` follows the PRE-REGISTERED rule, with
   `verdict_paired` beside it, so nothing is quietly upgraded after the fact.

⚠️ McNEMAR TESTS ACCURACY. THE PRIMARY METRIC OF THIS THESIS IS MACRO-F1.
------------------------------------------------------------------------
This is a **measured divergence**. On DBO, `tml_lightgbm` against `tml_svm` at
seed 42:

    accuracy   0.8723 vs 0.8703   (+0.19 pp)   McNemar p = 0.48
    macro-F1   0.3731 vs 0.4616   (-8.85 pp)   paired 95 % CI [-0.128, -0.052]
      of which `subversive` (n = 60):  F1 0.064 vs 0.385
                `nothing`   (n = 6123): F1 0.933 vs 0.931

McNemar's null is about the DISCORDANT PAIRS -- which system got which item
right -- so it is dominated by the 6 123 `nothing` items and is structurally
close to blind on the 60-item class the research question is about. Macro-F1
weights the four classes equally and sees exactly that class. Both numbers are
correct; they answer different questions. Where they disagree, the
pre-registered disjunction can be carried by its interval half against its own
test -- `carried_by` records which half decided each verdict, so no claim rests
on an unread OR.

⚠️ MACHINE MISMATCH IS FLAGGED, NOT ABSORBED
--------------------------------------------
`baseline_comparison` withholds an E1 delta whose two sides come from different
machines. The stored E2 arms sit on both sides of that line: the encoder and
LLM arms ran on a rented Linux GPU machine and the TML arm on Darwin/arm64.
Both saw the same items, the same folds and the same data rules, so the pairing
is sound and McNemar is computable -- but the *size* of a cross-machine
difference carries a platform component that no test here can separate out.
Every record therefore carries `machine_a`, `machine_b` and `machine_mismatch`.
This module does not refuse; it records the mismatch loudly, and the thesis
decides which comparisons it reads.

BOOTSTRAP SIZE AND THE MINORITY CLASS
-------------------------------------
1. **A verdict needs more resamples than a descriptive interval.** At B=2000 the
   vio `encoder_1b`-vs-`llm_llammlein` verdict flipped 4 times out of 20
   resampling seeds; at B=20000 its interval is [-0.0327, +0.0003]. This module
   therefore runs at `protocol.significance_bootstrap_n` (10 000), separate from
   `protocol.bootstrap_n` (1 000), which sizes the DESCRIPTIVE per-run intervals
   where no binary decision is taken.

2. **The minority-class claim gets its own interval.** "Highest minority-class
   F1" is a rank until something tests it: on c2a it holds (+4.64 pp, CI
   [+2.08, +7.23]); on vio it does not (+3.08 pp, CI [-0.05, +6.16]); on dbo the
   class holds 60 items and the interval is [-12.10, +17.43]. Every pair carries
   a `minority_class` block gated by `harness.tier_for_n`, so the reporting
   tiers decide what may be said.

Usage
-----
python -m src.significance --help
python -m src.significance                 # every cross-component pair, all subtasks
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from src.component_store import ComponentStore, load_matrix
from src.harness import measuring_main_guard, tier_for_n

RESULTS = Path(__file__).resolve().parents[1] / "results"

_MATRIX = load_matrix()
BOOTSTRAP_N = int(_MATRIX["protocol"]["bootstrap_n"])
CI_LEVEL = float(_MATRIX["protocol"]["ci_level"])
PRIMARY_SEED = int(_MATRIX["protocol"]["seed_base"])

# The resample count THIS module runs at, deliberately larger than the
# artefact-side `bootstrap_n`. See the matrix comment beside the key: here a
# bootstrap interval decides a verdict, and even at 2000 the vio encoder-vs-LLM
# verdict flipped with the resampling seed (measured).
SIG_BOOTSTRAP_N = int(_MATRIX["protocol"]["significance_bootstrap_n"])

# Below this many discordant pairs the chi-square approximation is not trusted
# and the exact binomial test is used instead. 25 is the conventional floor and
# is stated here rather than buried, because on a thin class the branch taken is
# the difference between a p-value and a wrong p-value.
EXACT_BELOW = 25


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def paired_predictions(rec_a: dict[str, Any], rec_b: dict[str, Any]) -> dict[str, Any]:
    """Align two stored runs, refusing anything that would make the pair a lie.

    Every refusal below is a way the test could return a number that means
    nothing: predictions compared item-by-item must BE the same items, in the
    same order, judged against the same gold labels, under the same data rules.
    None of these are hypothetical -- the store holds runs from two machines,
    three seeds and two rule generations, and `dbo_2025/tml_svm__seed43` is a
    perfectly valid neighbour of `__seed42` that must never be paired with it
    (a different partition means different fold assignments, so 'the same item'
    was predicted out of fold in a different context).
    """
    ids_a, ids_b = np.asarray(rec_a["ids"]), np.asarray(rec_b["ids"])
    if ids_a.shape != ids_b.shape or not np.array_equal(ids_a, ids_b):
        raise ValueError(
            "the two runs do not carry identical item ids in identical order "
            "({} vs {} items) -- McNemar's pairing does not exist".format(
                len(ids_a), len(ids_b)))
    if list(rec_a["y_true"]) != list(rec_b["y_true"]):
        raise ValueError("the two runs disagree on the gold labels of the same ids")
    if list(rec_a["classes"]) != list(rec_b["classes"]):
        raise ValueError(
            "class orderings differ: {!r} vs {!r}".format(
                rec_a["classes"], rec_b["classes"]))
    if not np.array_equal(np.asarray(rec_a["fold"]), np.asarray(rec_b["fold"])):
        raise ValueError(
            "the two runs used different fold assignments, so 'out of fold' "
            "means something different on each side -- check the partition seed")
    ma, mb = rec_a["meta"], rec_b["meta"]
    if ma.get("data_rules_id") != mb.get("data_rules_id"):
        raise ValueError(
            "different data-rule generations ({} vs {}); the pools are not the "
            "same pool (#65)".format(ma.get("data_rules_id"), mb.get("data_rules_id")))

    classes = list(rec_a["classes"])
    y_true = np.asarray([str(v) for v in rec_a["y_true"]])
    pred_a = np.asarray([classes[i] for i in np.asarray(rec_a["proba"]).argmax(axis=1)])
    pred_b = np.asarray([classes[i] for i in np.asarray(rec_b["proba"]).argmax(axis=1)])
    return {"classes": classes, "y_true": y_true, "pred_a": pred_a,
            "pred_b": pred_b, "n": int(len(y_true))}


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------

def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray,
            exact_below: int = EXACT_BELOW) -> dict[str, Any]:
    """McNemar's test on two boolean correctness vectors over the same items.

    Only the DISCORDANT cells carry information: b = items a got right and b got
    wrong, c = the reverse. The items both systems agree on -- typically the
    large majority -- say nothing about which is better, which is exactly why
    this test is more sensitive here than two marginal intervals.
    """
    from scipy import stats

    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError("correctness vectors differ in length")

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_disc = b + c

    if n_disc == 0:
        return {"b": b, "c": c, "n_discordant": 0, "method": "none",
                "statistic": None, "p_value": 1.0,
                "note": ("the two systems made identical predictions on every "
                         "item; there is nothing to test")}

    if n_disc < exact_below:
        # Two-sided exact binomial against p = 0.5. `binomtest` is the current
        # name; scipy removed `binom_test` in 1.12.
        p = float(stats.binomtest(min(b, c), n_disc, 0.5).pvalue)
        return {"b": b, "c": c, "n_discordant": n_disc,
                "method": "exact_binomial",
                "statistic": None, "p_value": p,
                "note": ("{} discordant pairs is below {}, so the chi-square "
                         "approximation is not used".format(n_disc, exact_below))}

    stat = (abs(b - c) - 1) ** 2 / n_disc          # continuity correction
    p = float(stats.chi2.sf(stat, df=1))
    return {"b": b, "c": c, "n_discordant": n_disc,
            "method": "chi2_continuity", "statistic": float(stat),
            "p_value": p,
            "note": "chi-square with Yates continuity correction, 1 df"}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _macro_f1(y_true, y_pred, classes) -> float:
    return float(f1_score(y_true, y_pred, labels=classes, average="macro",
                          zero_division=0))


def _f1_per_class_from_conf(conf: np.ndarray) -> np.ndarray:
    """Per-class F1 from a confusion matrix indexed [true, predicted].

    F1 = 2*tp / (2*tp + fp + fn) = 2*tp / (row_sum + column_sum), which is
    algebraically the 2*p*r/(p+r) sklearn computes and agrees with it in the
    degenerate cells too: a class with no predictions and no instances scores 0
    and still enters the macro average, which is `zero_division=0` with an
    explicit `labels=`. Verified bit-identical against `_macro_f1` -- see
    tests/test_significance.py.
    """
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    denom = conf.sum(-2) + conf.sum(-1)
    return np.where(denom > 0, 2.0 * tp / np.where(denom > 0, denom, 1.0), 0.0)


def paired_bootstrap(y_true, pred_a, pred_b, classes, *,
                     n_bootstrap: int = SIG_BOOTSTRAP_N, ci: float = CI_LEVEL,
                     seed: int = PRIMARY_SEED) -> dict[str, Any]:
    """Bootstrap both systems on the SAME resamples, and the difference with them.

    Resampling the two systems independently would break the pairing and give
    the difference a wider interval than it has: the correlation between two
    classifiers scored on the same items is real and is information. One index
    draw, applied to both.

    The loop scores each resample from bincounted confusion cells rather than
    through `f1_score`. That is not a micro-optimisation: it is what makes
    `SIG_BOOTSTRAP_N` affordable at all (45 pairs went from ~11 min to seconds),
    and a verdict that is only stable at a resample count nobody can afford to
    run is not stable. The index draws are unchanged, so the numbers are the
    numbers the previous implementation produced.

    Per-class F1 rides along on the SAME resamples, because a cross-component
    claim about a minority class needs the same pairing the macro claim gets.
    """
    y_true = np.asarray(y_true)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    classes = list(classes)
    K = len(classes)
    index = {c: i for i, c in enumerate(classes)}
    y_i = np.fromiter((index[v] for v in y_true), dtype=np.int64, count=len(y_true))
    a_i = np.fromiter((index[v] for v in pred_a), dtype=np.int64, count=len(pred_a))
    b_i = np.fromiter((index[v] for v in pred_b), dtype=np.int64, count=len(pred_b))
    cell_a, cell_b = y_i * K + a_i, y_i * K + b_i

    rng = np.random.default_rng(seed)
    n = len(y_true)
    a_s = np.empty(n_bootstrap)
    b_s = np.empty(n_bootstrap)
    a_c = np.empty((n_bootstrap, K))
    b_c = np.empty((n_bootstrap, K))
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        w = np.bincount(idx, minlength=n).astype(np.float64)
        ca = np.bincount(cell_a, weights=w, minlength=K * K).reshape(K, K)
        cb = np.bincount(cell_b, weights=w, minlength=K * K).reshape(K, K)
        fa, fb = _f1_per_class_from_conf(ca), _f1_per_class_from_conf(cb)
        a_c[i], b_c[i] = fa, fb
        a_s[i], b_s[i] = fa.mean(), fb.mean()
    d = a_s - b_s
    alpha = (1 - ci) / 2

    def q(arr):
        return [float(np.quantile(arr, alpha)), float(np.quantile(arr, 1 - alpha))]

    ci_a, ci_b, ci_d = q(a_s), q(b_s), q(d)

    # The p-value that belongs to the interval above: the INVERSION of the
    # percentile interval, 2*min(P(d<=0), P(d>=0)). Deliberately this and not a
    # null-shifted bootstrap (Berg-Kirkpatrick et al. 2012) -- the two are
    # asymptotically equivalent and the inversion is the one coherent with the
    # percentile CIs this module already reports, so interval and p-value cannot
    # disagree with each other. Floored at 1/B: a bootstrap cannot resolve below
    # its own resample count, and `p_resolution_floor` says so rather than
    # letting a reader read 0.0001 as a measurement.
    p_boot = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    p_boot = max(1.0 / n_bootstrap, min(1.0, p_boot))

    # Per-class point estimates and supports from the full sample, not the
    # resamples -- the resamples estimate the interval, never the estimate.
    conf_a = np.bincount(cell_a, minlength=K * K).reshape(K, K).astype(np.float64)
    conf_b = np.bincount(cell_b, minlength=K * K).reshape(K, K).astype(np.float64)
    f1_a_full, f1_b_full = _f1_per_class_from_conf(conf_a), _f1_per_class_from_conf(conf_b)
    support = conf_a.sum(1).astype(int)

    per_class = {}
    for k, cls in enumerate(classes):
        pc_a, pc_b = q(a_c[:, k]), q(b_c[:, k])
        pc_d = q(a_c[:, k] - b_c[:, k])
        per_class[str(cls)] = {
            "support": int(support[k]),
            "tier": tier_for_n(int(support[k])),
            "f1_a": float(f1_a_full[k]),
            "f1_b": float(f1_b_full[k]),
            "ci_a": pc_a,
            "ci_b": pc_b,
            "difference": float(f1_a_full[k] - f1_b_full[k]),
            "ci_difference": pc_d,
            "intervals_overlap": not (pc_a[0] > pc_b[1] or pc_b[0] > pc_a[1]),
            "difference_ci_excludes_zero": bool(pc_d[0] > 0 or pc_d[1] < 0),
        }

    return {
        "n_bootstrap": int(n_bootstrap),
        "ci_level": float(ci),
        "macro_f1_a": _macro_f1(y_true, pred_a, classes),
        "macro_f1_b": _macro_f1(y_true, pred_b, classes),
        "ci_a": ci_a,
        "ci_b": ci_b,
        "difference": _macro_f1(y_true, pred_a, classes) - _macro_f1(y_true, pred_b, classes),
        "ci_difference": ci_d,
        # The PRE-REGISTERED rule, reported as written.
        "intervals_overlap": not (ci_a[0] > ci_b[1] or ci_b[0] > ci_a[1]),
        # The paired rule, reported beside it. See the module docstring on why
        # these two can disagree and why neither is quietly preferred here.
        "difference_ci_excludes_zero": bool(ci_d[0] > 0 or ci_d[1] < 0),
        "p_value": p_boot,
        "p_resolution_floor": 1.0 / n_bootstrap,
        "p_value_note": (
            "the percentile interval inverted, on macro-F1 -- the PRIMARY "
            "metric, and therefore the quantity a multiplicity correction has "
            "to be applied to. McNemar's p beside it is about accuracy."),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# One comparison
# ---------------------------------------------------------------------------

def compare(rec_a: dict[str, Any], rec_b: dict[str, Any], *,
            key_a: str = "a", key_b: str = "b", alpha: float = 0.05,
            n_bootstrap: int = SIG_BOOTSTRAP_N, seed: int = PRIMARY_SEED
            ) -> dict[str, Any]:
    """The full record for one paired comparison, ready for Chapter 4."""
    pair = paired_predictions(rec_a, rec_b)
    y_true, classes = pair["y_true"], pair["classes"]
    ca = pair["pred_a"] == y_true
    cb = pair["pred_b"] == y_true

    mc = mcnemar(ca, cb)
    bs = paired_bootstrap(y_true, pair["pred_a"], pair["pred_b"], classes,
                          n_bootstrap=n_bootstrap, seed=seed)

    # Accuracy is reported not because anything claims it, but because it is
    # what McNemar's p-value is ABOUT. Without it beside the macro-F1 the
    # divergence documented in the module docstring reads as a contradiction
    # instead of as two answers to two questions.
    acc_a, acc_b = float(ca.mean()), float(cb.mean())

    # The verdict follows the PRE-REGISTERED disjunction: McNemar significant OR
    # the two intervals fail to overlap. Both halves are reported separately so
    # a reader can see which one carried it.
    mcnemar_sig = mc["p_value"] < alpha
    intervals_disjoint = not bs["intervals_overlap"]
    verdict = ("distinguishable" if (mcnemar_sig or intervals_disjoint)
               else "indistinguishable")
    if mcnemar_sig and intervals_disjoint:
        carried_by = "both"
    elif mcnemar_sig:
        carried_by = "mcnemar"
    elif intervals_disjoint:
        carried_by = "intervals"
    else:
        carried_by = "neither"
    verdict_paired = ("distinguishable" if (mcnemar_sig or bs["difference_ci_excludes_zero"])
                      else "indistinguishable")

    # --- the minority class, under the pre-registered reporting tiers -------
    # SQ1 is argued on the rare harmful classes, not on the macro average, so a
    # cross-component claim about them needs its own interval, and on dbo that
    # class holds 60 items, where the `ci_gated` tier already fixes what may be
    # said about it.
    pc = bs["per_class"]
    minority = min(pc, key=lambda c: (pc[c]["support"], str(c)))
    m = pc[minority]
    if m["tier"] == "report_only":
        permitted = False
        gate = ("D1 report_only (n <= 29): the score is reported for "
                "completeness and carries no claim, directional or otherwise")
    elif m["tier"] == "ci_gated":
        permitted = not m["intervals_overlap"]
        gate = ("D1 ci_gated (30 <= n <= 99), applied as pre-registered: a "
                "directional claim is permitted only where the two components' "
                "bootstrap intervals for this class do not overlap")
    else:
        permitted = m["difference_ci_excludes_zero"]
        gate = ("D1 interpret (n >= 100): #58(b) at class level -- the paired "
                "bootstrap interval of the difference must exclude zero")
    minority_block = dict(m)
    minority_block.update({
        "class": str(minority),
        "higher": (key_a if m["difference"] > 0 else key_b) if m["difference"] else None,
        "directional_claim_permitted": bool(permitted),
        "gate": gate,
        "wording_rule": (
            "where `directional_claim_permitted` is false the text must not say "
            "one component is better than the other ON THIS CLASS, and must not "
            "crown the maximum -- #58(b)'s rule, at the level SQ1 actually "
            "argues. A rank is not a win here either."),
    })

    ma, mb = rec_a["meta"], rec_b["meta"]
    machine_a, machine_b = ma.get("machine"), mb.get("machine")
    return {
        "a": key_a,
        "b": key_b,
        "n_items": pair["n"],
        "classes": classes,
        "alpha": float(alpha),
        "mcnemar": mc,
        "mcnemar_significant": bool(mcnemar_sig),
        "bootstrap": bs,
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "accuracy_difference": acc_a - acc_b,
        "metric_scope_note": (
            "McNemar's p-value above is about ACCURACY -- which system got which "
            "item right. The primary metric is macro-F1, which weights the "
            "classes equally. On an imbalanced pool the two come apart, measured "
            "and documented in this module's docstring; read the two together."),
        "verdict": verdict,
        "carried_by": carried_by,
        "verdict_rule": ("PRE-REGISTERED (#58 b): McNemar at alpha, OR the two "
                         "components' own bootstrap intervals do not overlap. "
                         "`carried_by` says which half decided, because an OR "
                         "whose halves disagree is not self-explanatory"),
        "minority_class": minority_block,
        "verdict_paired": verdict_paired,
        "verdict_paired_rule": ("the stronger form: McNemar at alpha, OR the "
                                "paired bootstrap interval of the DIFFERENCE "
                                "excludes zero. Reported beside the "
                                "pre-registered verdict, never in place of it"),
        "wording_rule": (
            "where the verdict is `indistinguishable`, the text says so and does "
            "not crown the maximum (#58 b). A larger macro-F1 that this test "
            "cannot separate is not a finding."),
        "machine_a": machine_a,
        "machine_b": machine_b,
        "machine_mismatch": bool(machine_a != machine_b),
        "machine_note": (
            None if machine_a == machine_b else
            "the two sides ran on different machines ({} vs {}). The pairing is "
            "sound -- same items, same folds, same data rules -- but the SIZE of "
            "the difference carries a platform component this test cannot "
            "separate out. #65/#67 withhold an E1 delta in this situation; "
            "whether an E2 cross-component claim is held to the same rule is a "
            "decision for the log and G2.".format(machine_a, machine_b)),
        "config_id_a": ma.get("config_id"),
        "config_id_b": mb.get("config_id"),
        "data_rules_id": ma.get("data_rules_id"),
        "seed_a": ma.get("seed"),
        "seed_b": mb.get("seed"),
    }


# ---------------------------------------------------------------------------
# Every cross-component pair the store holds
# ---------------------------------------------------------------------------

def compare_store(store: ComponentStore | None = None, *,
                  seed: int = PRIMARY_SEED,
                  n_bootstrap: int = SIG_BOOTSTRAP_N) -> dict[str, Any]:
    """Every cross-component pair, per subtask, at ONE partition seed.

    Seeds are not crossed: `__seed43` is a different partition, so pairing it
    with `__seed42` would compare predictions made under different fold
    assignments. Within a seed the pairing is exact, which is the whole reason
    this is a laptop job and not a re-run.
    """
    store = store or ComponentStore()
    suffix = "__seed{}".format(seed)
    by_cell: dict[str, list[str]] = {}
    from src.component_store import parse_run_key
    for key in sorted(store.list_runs()):
        # E2 runs only: an E4 key ends in `__seed42` too (parse_run_key's note).
        if not key.endswith(suffix) or parse_run_key(key)["variant"] is not None:
            continue
        cell = key.split("/")[0]
        by_cell.setdefault(cell, []).append(key)

    out: dict[str, Any] = {}
    for cell, keys in sorted(by_cell.items()):
        recs = {k: store.load(k) for k in keys}
        pairs = []
        for ka, kb in combinations(keys, 2):
            pairs.append(compare(recs[ka], recs[kb], key_a=ka, key_b=kb,
                                 n_bootstrap=n_bootstrap, seed=seed))
        out[cell] = pairs
    return out


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------

def _holm(p: list[float]) -> list[float]:
    """Holm (1979) step-down, returned as adjusted p-values.

    Chosen over plain Bonferroni because it is uniformly more powerful at an
    identical guarantee, and over Benjamini-Hochberg because **BH is not valid
    here**: it needs independence or positive regression dependency, and these
    45 tests are a round-robin -- six components compared pairwise on the same
    items, three subtasks -- whose dependence structure includes negative parts
    (if A beats B and B beats C, the A-against-C statistic is not free). Holm's
    FWER control holds under ARBITRARY dependence and therefore needs no
    assumption this design cannot support.
    """
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def _benjamini_yekutieli(p: list[float]) -> list[float]:
    """BY (2001): FDR control under ARBITRARY dependence.

    The FDR alternative to Holm, reported beside it because FWER and FDR answer
    different questions and this study has no reason to hide one of them. BY
    rather than BH for the reason in `_holm`: the extra sum(1/i) factor is
    exactly the price of not assuming a dependence structure we cannot verify.
    """
    m = len(p)
    c = sum(1.0 / i for i in range(1, m + 1))
    order = sorted(range(m), key=lambda i: p[i])
    out = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        out[i] = min(1.0, prev * c)
    return out


def apply_multiplicity(cells: dict[str, Any], *, alpha: float = 0.05) -> dict[str, Any]:
    """Correct every pair for multiple comparisons, over the LARGEST family.

    ⚠️ WHY THE FAMILY IS ALL 45 PAIRS AND NOT A SMALLER ONE. Which tests form a
    family is a decision that must be made before the p-values are looked at,
    or it is itself a degree of freedom. Two smaller families are defensible
    here -- one per subtask (15), or the three confirmatory SQ1 comparisons --
    and **both would be less strict, i.e. both would favour this thesis**. That
    is the reason for taking neither. The strictest family is corrected and
    reported; if a claim survives that, every smaller family is satisfied a
    fortiori and the choice never has to be argued. Same move as the
    minority-class tier gate: take the stricter rule, because it holds.

    A correction fixes ONE error source. It does not touch the two named in
    `multiplicity_does_not_cover` below, and those belong in the same paragraph
    of Chapter 4 as these numbers.
    """
    pairs = [r for rows in cells.values() for r in rows]
    m = len(pairs)
    if not m:
        return {"family": "empty", "m": 0}
    p_boot = [r["bootstrap"]["p_value"] for r in pairs]
    p_mcn = [r["mcnemar"]["p_value"] for r in pairs]
    hb, yb = _holm(p_boot), _benjamini_yekutieli(p_boot)
    hm, ym = _holm(p_mcn), _benjamini_yekutieli(p_mcn)
    for i, r in enumerate(pairs):
        r["multiplicity"] = {
            "family": "all cross-component pairs at the primary seed, every subtask",
            "m": m,
            "alpha": alpha,
            "p_bootstrap": p_boot[i],
            "holm_bootstrap": hb[i],
            "by_bootstrap": yb[i],
            "p_mcnemar": p_mcn[i],
            "holm_mcnemar": hm[i],
            "by_mcnemar": ym[i],
            "survives_holm_bootstrap": bool(hb[i] < alpha),
            "survives_by_bootstrap": bool(yb[i] < alpha),
            "primary": ("holm_bootstrap -- Holm on the paired macro-F1 p-value: "
                        "the correction that matches both the decision rule and "
                        "a dependence structure we cannot assume away"),
        }
    return {
        "family": "all cross-component pairs at the primary seed, every subtask",
        "m": m,
        "alpha": alpha,
        "methods": ["holm", "benjamini_yekutieli"],
        "method_not_used": {
            "benjamini_hochberg": (
                "NOT used and deliberately absent: BH controls FDR under "
                "independence or PRDS, and these are round-robin comparisons of "
                "six components on the same items, whose dependence includes "
                "negative parts. BY is the FDR procedure valid under arbitrary "
                "dependence and is reported instead."),
            "bonferroni": "superseded by Holm, which is uniformly more powerful "
                          "at the identical FWER guarantee",
        },
        "by_factor_sum_1_over_i": sum(1.0 / i for i in range(1, m + 1)),
        "family_choice_note": (
            "the LARGEST family, chosen before the p-values were read. Per-subtask "
            "(15) and the three confirmatory SQ1 comparisons are both defensible "
            "and both less strict, i.e. both would favour this thesis -- which is "
            "why neither was taken. A claim surviving m={} satisfies any smaller "
            "family a fortiori.".format(m)),
        "multiplicity_does_not_cover": [
            "TRAINING-SEED VARIANCE. Both tests condition on ONE trained model. "
            "The D3 replicates spread 2.36 pp on c2a against a 2.54 pp margin, so "
            "a corrected p says 'this model beat that model on these items', not "
            "'this method beats that method'.",
            "THE PAIRS ARE NOT A RANDOM SAMPLE. They are a complete round-robin, "
            "and most are uninteresting (TML against the LLM at 30 pp). That most "
            "survive Holm is not a quality signal; large differences stay large.",
        ],
    }


def _component_of(key: str) -> str:
    return key.split("/")[-1].split("__")[0]


def main() -> int:
    args = measuring_main_guard(__doc__, RESULTS / "significance.json")
    if args is None:
        return 0

    store = ComponentStore()
    cells = compare_store(store)
    multiplicity = apply_multiplicity(cells)

    n_pairs = sum(len(v) for v in cells.values())
    print("\n=== paired significance, {} cross-component pairs at seed {} ==="
          .format(n_pairs, PRIMARY_SEED))
    mismatched = 0
    for cell, pairs in cells.items():
        print("\n{}".format(cell))
        for p in pairs:
            bs = p["bootstrap"]
            mismatched += bool(p["machine_mismatch"])
            print("  {:<14s} {:>7.4f}   vs  {:<14s} {:>7.4f}   d={:+.4f} "
                  "[{:+.4f}, {:+.4f}]".format(
                      _component_of(p["a"]), bs["macro_f1_a"],
                      _component_of(p["b"]), bs["macro_f1_b"],
                      bs["difference"], *bs["ci_difference"]))
            print("      acc {:.4f}/{:.4f}  McNemar b={} c={} p={:.2e} ({})"
                  .format(p["accuracy_a"], p["accuracy_b"],
                          p["mcnemar"]["b"], p["mcnemar"]["c"],
                          p["mcnemar"]["p_value"], p["mcnemar"]["method"]))
            mb_ = p["minority_class"]
            print("      minority '{}' (n={}, {}): {:.4f} vs {:.4f}  d={:+.4f} "
                  "[{:+.4f}, {:+.4f}] -> directional claim {}".format(
                      mb_["class"], mb_["support"], mb_["tier"],
                      mb_["f1_a"], mb_["f1_b"], mb_["difference"],
                      *mb_["ci_difference"],
                      "PERMITTED" if mb_["directional_claim_permitted"]
                      else "NOT permitted"))
            mu = p["multiplicity"]
            print("      macro-F1 p={:.4f}  Holm(m={})={:.4f}  BY={:.4f}  -> {}"
                  .format(mu["p_bootstrap"], mu["m"], mu["holm_bootstrap"],
                          mu["by_bootstrap"],
                          "survives" if mu["survives_holm_bootstrap"]
                          else "does NOT survive correction"))
            print("      -> {} (carried by {}){}{}".format(
                p["verdict"].upper(), p["carried_by"],
                "" if p["verdict"] == p["verdict_paired"]
                else "   [paired rule: {}]".format(p["verdict_paired"].upper()),
                "   ⚠ cross-machine" if p["machine_mismatch"] else ""))

    payload = {
        "experiment": "selection-layer discipline (THESIS_LOG #58 b, #105 b)",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "anchor": "dietterich1998; 3.3 sec:evaluation; design lock D1 one level up",
        "seed": PRIMARY_SEED,
        "bootstrap_n": SIG_BOOTSTRAP_N,
        "artefact_bootstrap_n": BOOTSTRAP_N,
        "bootstrap_n_note": (
            "this module runs at protocol.significance_bootstrap_n, which is "
            "deliberately larger than the protocol.bootstrap_n the stored "
            "artefacts carry: here an interval decides a verdict. At 1000 the "
            "vio encoder-vs-LLM verdict flipped with the resampling seed "
            "(measured 2026-09-07, 4 of 20 seeds at B=2000)."),
        "ci_level": CI_LEVEL,
        "alpha": 0.05,
        "multiplicity": multiplicity,
        "scope": ("every cross-component pair the component store holds, per "
                  "subtask, at the primary partition seed. Seeds are not "
                  "crossed: a different seed is a different partition."),
        "cells": cells,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print("\nwrote {}".format(args.out))
    diverged = [p for pairs in cells.values() for p in pairs
                if p["verdict"] != p["verdict_paired"]
                or (p["carried_by"] == "intervals")]
    if mismatched:
        print("⚠ {} of {} pairs span two machines -- see machine_note in the "
              "artefact (#65/#67).".format(mismatched, n_pairs))
    surviving = sum(1 for pairs in cells.values() for p in pairs
                    if p["multiplicity"]["survives_holm_bootstrap"])
    print("\nmultiplicity: family m={}, Holm on the paired macro-F1 p-value -- "
          "{} of {} pairs survive at alpha=0.05 (BY beside it; BH is NOT used, "
          "see method_not_used in the artefact)"
          .format(multiplicity["m"], surviving, n_pairs))
    if diverged:
        print("⚠ {} of {} pairs where McNemar disagrees with the macro-F1 "
              "interval. McNemar tests accuracy; the primary metric is "
              "macro-F1. See metric_scope_note and the module docstring -- "
              "which test settles a Chapter 4 claim is a design decision, not "
              "one this module makes.".format(len(diverged), n_pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
