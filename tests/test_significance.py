"""test_significance.py -- the selection-layer discipline, against constructed truth.

WHY CONSTRUCTED AND NOT ONLY MEASURED. A significance layer is the one component
whose output nobody can eyeball: a p-value that is wrong by a factor of ten
looks exactly like one that is right, and it would be wrong in the direction of
crowning a maximum the data does not support -- the precise failure #58(b)
exists to prevent. So the tests below feed it cases whose answer is known before
the code runs (a system that is strictly better; two systems that are identical;
a hand-computed 2x2), and only then let it near the store.

WHAT IT GUARDS:
  * McNemar reproduces a hand-computed chi-square with continuity correction,
    and switches to the exact binomial where the approximation is not trusted;
  * two identical systems come back `indistinguishable` with p = 1, and a
    strictly better one comes back `distinguishable` -- the two ends of the
    scale, either of which being wrong would invalidate every claim;
  * the paired bootstrap really is PAIRED: it must give the difference a
    narrower interval than resampling the two systems independently would;
  * every way of pairing two runs that are not comparable is REFUSED --
    different ids, different order, different gold labels, different fold
    assignments, different data-rule generations. A silent mis-pairing produces
    a number, and a number is what gets believed;
  * against the real store, the premise #105(b) rests on: encoder_1b and tml_svm
    on dbo carry identical ids in identical order.

Run: .venv/bin/python -m tests.test_significance
"""
from __future__ import annotations

import numpy as np

from src import significance as sig
from src.component_store import ComponentStore

_PASS, _FAIL = 0, 0


def ok(m):
    global _PASS
    _PASS += 1
    print("  \033[32mPASS\033[0m  " + m)


def bad(m):
    global _FAIL
    _FAIL += 1
    print("  \033[31mFAIL\033[0m  " + m)


def fake_record(ids, y_true, pred, classes, *, machine="Linux/x86_64",
                fold=None, data_rules_id="rules0"):
    """A store-shaped record built from hard labels: proba is one-hot, so
    argmax reproduces `pred` exactly and the test controls the predictions."""
    classes = list(classes)
    index = {c: i for i, c in enumerate(classes)}
    proba = np.zeros((len(pred), len(classes)))
    for i, p in enumerate(pred):
        proba[i, index[p]] = 1.0
    return {
        "ids": np.asarray(ids), "y_true": [str(v) for v in y_true],
        "classes": classes, "proba": proba,
        "fold": np.asarray(fold if fold is not None else [0] * len(pred)),
        "meta": {"machine": machine, "data_rules_id": data_rules_id,
                 "config_id": "cfg", "seed": 42},
    }


def main() -> int:
    print("\n=== the selection layer (#58 b), against constructed truth ===")

    # --- McNemar against a hand-computed value -------------------------------
    try:
        from scipy import stats

        # b = 30, c = 10 -> chi2 = (|30-10|-1)^2 / 40 = 361/40 = 9.025
        a = np.array([True] * 30 + [False] * 10 + [True] * 60)
        b = np.array([False] * 30 + [True] * 10 + [True] * 60)
        r = sig.mcnemar(a, b)
        assert (r["b"], r["c"]) == (30, 10), (r["b"], r["c"])
        assert r["method"] == "chi2_continuity", r["method"]
        assert abs(r["statistic"] - 9.025) < 1e-9, r["statistic"]
        assert abs(r["p_value"] - float(stats.chi2.sf(9.025, 1))) < 1e-12
        ok("McNemar reproduces the hand-computed chi-square with continuity "
           "correction (b=30, c=10, chi2={:.3f}, p={:.5f})".format(
               r["statistic"], r["p_value"]))
    except AssertionError as e:
        bad("McNemar chi-square: {}".format(e))
    except Exception as e:
        bad("McNemar chi-square raised {}: {}".format(type(e).__name__, e))

    # --- and switches to the exact test where chi-square is not trusted ------
    try:
        a = np.array([True] * 8 + [False] * 2 + [True] * 90)
        b = np.array([False] * 8 + [True] * 2 + [True] * 90)
        r = sig.mcnemar(a, b)
        assert r["n_discordant"] == 10 < sig.EXACT_BELOW
        assert r["method"] == "exact_binomial", (
            "10 discordant pairs took the chi-square branch; the approximation "
            "is exactly what fails on the thin classes this study is about")
        expect = float(stats.binomtest(2, 10, 0.5).pvalue)
        assert abs(r["p_value"] - expect) < 1e-12, (r["p_value"], expect)
        ok("below {} discordant pairs the exact binomial is used, not the "
           "approximation (p={:.4f})".format(sig.EXACT_BELOW, r["p_value"]))
    except AssertionError as e:
        bad("McNemar exact branch: {}".format(e))
    except Exception as e:
        bad("McNemar exact branch raised {}: {}".format(type(e).__name__, e))

    # --- the two ends of the scale ------------------------------------------
    try:
        n = 400
        rng = np.random.default_rng(0)
        classes = ["0", "1"]
        y = np.array([str(v) for v in rng.integers(0, 2, n)])
        ids = np.arange(n)

        same = sig.compare(fake_record(ids, y, y, classes),
                           fake_record(ids, y, y, classes),
                           n_bootstrap=200)
        assert same["mcnemar"]["n_discordant"] == 0
        assert same["mcnemar"]["p_value"] == 1.0
        assert same["verdict"] == "indistinguishable", same["verdict"]
        assert same["carried_by"] == "neither", same["carried_by"]

        # a strictly better system: identical to gold, against one that is wrong
        # on a quarter of the items and never right where the other is wrong.
        worse = y.copy()
        flip = rng.choice(n, size=n // 4, replace=False)
        worse[flip] = np.where(y[flip] == "0", "1", "0")
        better = sig.compare(fake_record(ids, y, y, classes),
                             fake_record(ids, y, worse, classes),
                             n_bootstrap=200)
        assert better["verdict"] == "distinguishable", better["verdict"]
        assert better["mcnemar_significant"], better["mcnemar"]
        assert better["bootstrap"]["difference"] > 0
        ok("identical systems are indistinguishable (p=1, nothing discordant) "
           "and a strictly better one is distinguishable (d={:+.3f}, p={:.2e})"
           .format(better["bootstrap"]["difference"], better["mcnemar"]["p_value"]))
    except AssertionError as e:
        bad("the two ends of the scale: {}".format(e))
    except Exception as e:
        bad("the two ends of the scale raised {}: {}".format(type(e).__name__, e))

    # --- the bootstrap is PAIRED, and it matters ----------------------------
    try:
        n = 600
        rng = np.random.default_rng(1)
        classes = ["0", "1"]
        y = np.array([str(v) for v in rng.integers(0, 2, n)])
        pa = y.copy()
        pa[rng.choice(n, 120, replace=False)] = "9"       # a wrong label
        pa = np.where(pa == "9", np.where(y == "0", "1", "0"), pa)
        pb = pa.copy()
        idx = rng.choice(n, 40, replace=False)            # small extra error
        pb[idx] = np.where(y[idx] == "0", "1", "0")

        bs = sig.paired_bootstrap(y, pa, pb, classes, n_bootstrap=400, seed=7)
        paired_width = bs["ci_difference"][1] - bs["ci_difference"][0]

        # what an UNPAIRED bootstrap of the same difference would give
        r2 = np.random.default_rng(7)
        diffs = []
        for _ in range(400):
            ia = r2.integers(0, n, n)
            ib = r2.integers(0, n, n)
            diffs.append(sig._macro_f1(y[ia], pa[ia], classes)
                         - sig._macro_f1(y[ib], pb[ib], classes))
        unpaired_width = float(np.quantile(diffs, 0.975) - np.quantile(diffs, 0.025))
        assert paired_width < unpaired_width, (
            "the paired interval ({:.4f}) is not narrower than the unpaired one "
            "({:.4f}), so the resamples are not actually shared and the "
            "correlation between two systems on the same items is being thrown "
            "away".format(paired_width, unpaired_width))
        ok("the difference interval is paired: width {:.4f} against {:.4f} "
           "unpaired".format(paired_width, unpaired_width))
    except AssertionError as e:
        bad("paired bootstrap: {}".format(e))
    except Exception as e:
        bad("paired bootstrap raised {}: {}".format(type(e).__name__, e))

    # --- every incomparable pairing is REFUSED ------------------------------
    try:
        n = 50
        classes = ["0", "1"]
        y = np.array(["0", "1"] * (n // 2))
        ids = np.arange(n)
        base = fake_record(ids, y, y, classes)

        cases = {
            "different ids": fake_record(ids + 1, y, y, classes),
            "reordered ids": fake_record(ids[::-1], y, y, classes),
            "different gold labels": fake_record(
                ids, np.where(y == "0", "1", "0"), y, classes),
            "different fold assignment": fake_record(
                ids, y, y, classes, fold=[1] * n),
            "different data-rule generation": fake_record(
                ids, y, y, classes, data_rules_id="rules1"),
            "different class ordering": fake_record(ids, y, y, ["1", "0"]),
        }
        for what, other in cases.items():
            try:
                sig.paired_predictions(base, other)
                raise AssertionError(
                    "{} was accepted; the test would return a number that means "
                    "nothing".format(what))
            except ValueError:
                pass
        ok("all {} incomparable pairings are refused rather than silently "
           "compared".format(len(cases)))
    except AssertionError as e:
        bad("refusals: {}".format(e))
    except Exception as e:
        bad("refusals raised {}: {}".format(type(e).__name__, e))

    # --- and the premise #105(b) rests on, against the real store -----------
    try:
        store = ComponentStore()
        keys = set(store.list_runs())
        ka, kb = "dbo_2025/encoder_1b__seed42", "dbo_2025/tml_svm__seed42"
        if not {ka, kb} <= keys:
            print("  \033[33mSKIP\033[0m  the store does not hold {} and {}"
                  .format(ka, kb))
        else:
            pair = sig.paired_predictions(store.load(ka), store.load(kb))
            assert pair["n"] > 1000, pair["n"]
            r = sig.compare(store.load(ka), store.load(kb), key_a=ka, key_b=kb,
                            n_bootstrap=100)
            assert r["machine_mismatch"] is True, (
                "the encoder arm ran on the pod and the TML arm on the laptop; "
                "if this is no longer true the machine flag has gone blind")
            assert r["machine_note"], "a flagged mismatch carries no note"
            assert r["verdict"] in ("distinguishable", "indistinguishable")
            ok("the real store pairs exactly: {} items, one config each, and "
               "the cross-machine flag fires ({} vs {})".format(
                   pair["n"], r["machine_a"], r["machine_b"]))
    except AssertionError as e:
        bad("real store: {}".format(e))
    except Exception as e:
        bad("real store raised {}: {}".format(type(e).__name__, e))

    # --- the fast bootstrap loop still IS the sklearn score ----------------
    # The loop was rewritten to score resamples from bincounted confusion cells
    # so that SIG_BOOTSTRAP_N is affordable. That is only legitimate if it is
    # the same number: an approximation here would move verdicts, silently.
    try:
        rng = np.random.default_rng(3)
        classes = ["a", "b", "c"]
        y = np.array([classes[i] for i in rng.integers(0, 3, 500)])
        pa = np.where(rng.random(500) < 0.3,
                      np.array([classes[i] for i in rng.integers(0, 3, 500)]), y)
        conf = np.zeros((3, 3))
        for t_, p_ in zip(y, pa):
            conf[classes.index(t_), classes.index(p_)] += 1
        fast = float(sig._f1_per_class_from_conf(conf).mean())
        slow = sig._macro_f1(y, pa, classes)
        assert fast == slow, (
            "the confusion-cell macro-F1 ({!r}) is not bit-identical to the "
            "sklearn one ({!r}); the bootstrap loop and the point estimate "
            "would then disagree".format(fast, slow))
        # a class present in neither gold nor prediction must still score 0 and
        # still enter the macro average -- that is `zero_division=0` with labels=
        empty = np.zeros((3, 3)); empty[0, 0] = 10.0
        assert list(sig._f1_per_class_from_conf(empty)) == [1.0, 0.0, 0.0]
        ok("the fast bootstrap loop is bit-identical to the sklearn macro-F1, "
           "empty classes included")
    except AssertionError as e:
        bad("fast bootstrap: {}".format(e))
    except Exception as e:
        bad("fast bootstrap raised {}: {}".format(type(e).__name__, e))

    # --- the minority-class gate actually GATES ----------------------------
    # Added 2026-09-07. G2 recorded a cross-component minority-class claim with
    # no interval behind it while the reporting tiers -- pre-registered, and
    # implemented in harness.tier_for_n since B3 -- said what may be said about
    # a class that size. The tiers existed; nothing applied them to a PAIR.
    try:
        n = 4000
        rng = np.random.default_rng(11)
        classes = ["major", "rare"]
        y = np.array(["rare" if i < 60 else "major" for i in range(n)])
        # two systems that differ on the rare class by a couple of items only
        pa = y.copy(); pb = y.copy()
        pa[rng.choice(60, 20, replace=False)] = "major"
        pb[rng.choice(60, 23, replace=False)] = "major"
        r = sig.compare(
            {"ids": np.arange(n), "y_true": y, "classes": classes,
             "fold": np.zeros(n, dtype=int), "meta": {"data_rules_id": "x"},
             "proba": np.eye(2)[[classes.index(v) for v in pa]]},
            {"ids": np.arange(n), "y_true": y, "classes": classes,
             "fold": np.zeros(n, dtype=int), "meta": {"data_rules_id": "x"},
             "proba": np.eye(2)[[classes.index(v) for v in pb]]},
            n_bootstrap=400)
        m = r["minority_class"]
        assert m["class"] == "rare", m["class"]
        assert m["support"] == 60, m["support"]
        assert m["tier"] == "ci_gated", (
            "a 60-instance class must land in D1's ci_gated tier, not {!r} -- "
            "if this moved, the gate below is testing nothing".format(m["tier"]))
        assert m["intervals_overlap"] is True
        assert m["directional_claim_permitted"] is False, (
            "the two rare-class intervals overlap and D1 ci_gated permits no "
            "directional claim there, yet the gate said one is allowed")
        # and it must OPEN where the class is large and the gap is real
        y2 = np.array(["rare" if i < 800 else "major" for i in range(n)])
        pa2 = y2.copy(); pb2 = y2.copy()
        pb2[rng.choice(800, 500, replace=False)] = "major"
        r2 = sig.compare(
            {"ids": np.arange(n), "y_true": y2, "classes": classes,
             "fold": np.zeros(n, dtype=int), "meta": {"data_rules_id": "x"},
             "proba": np.eye(2)[[classes.index(v) for v in pa2]]},
            {"ids": np.arange(n), "y_true": y2, "classes": classes,
             "fold": np.zeros(n, dtype=int), "meta": {"data_rules_id": "x"},
             "proba": np.eye(2)[[classes.index(v) for v in pb2]]},
            n_bootstrap=400)
        assert r2["minority_class"]["tier"] == "interpret"
        assert r2["minority_class"]["directional_claim_permitted"] is True, (
            "a 800-instance class with a 500-item gap must permit a directional "
            "claim; a gate that never opens is not a gate")
        ok("the minority-class gate refuses a ci_gated class whose intervals "
           "overlap (n=60) and opens on an interpret-tier class that separates")
    except AssertionError as e:
        bad("minority gate: {}".format(e))
    except Exception as e:
        bad("minority gate raised {}: {}".format(type(e).__name__, e))

    # --- multiplicity: the procedures, against hand-computed truth ---------
    # 45 pairwise tests at alpha=0.05 expect ~2 false positives by construction.
    # A wrong adjustment here would be invisible and would inflate exactly the
    # claim Chapter 4 rests on, so both procedures are pinned to a sequence
    # small enough to compute on paper.
    try:
        pv = [0.01, 0.02, 0.03, 0.04]                 # m = 4
        # Holm step-down: 4*.01=.04 | 3*.02=.06 | 2*.03=.06 | 1*.04=.04,
        # each floored by the running maximum -> .04, .06, .06, .06
        assert [round(x, 10) for x in sig._holm(pv)] == [0.04, 0.06, 0.06, 0.06], \
            sig._holm(pv)
        # BH on the same input is .04 throughout; BY multiplies by
        # sum(1/i) = 1 + 1/2 + 1/3 + 1/4 = 2.0833...
        c = 1 + 1/2 + 1/3 + 1/4
        assert all(abs(x - 0.04 * c) < 1e-12 for x in sig._benjamini_yekutieli(pv)), \
            sig._benjamini_yekutieli(pv)
        # an adjusted p is never smaller than its raw p, in either procedure
        import random as _r
        rp = [_r.random() for _ in range(30)]
        assert all(a >= b - 1e-12 for a, b in zip(sig._holm(rp), rp))
        assert all(a >= b - 1e-12 for a, b in zip(sig._benjamini_yekutieli(rp), rp))
        ok("Holm and Benjamini-Yekutieli reproduce a hand-computed sequence "
           "(BY factor {:.4f} at m=4) and never shrink a p-value".format(c))
    except AssertionError as e:
        bad("multiplicity procedures: {}".format(e))
    except Exception as e:
        bad("multiplicity raised {}: {}".format(type(e).__name__, e))

    # --- BH must stay OUT, and the family must stay the largest one --------
    # BH needs independence or PRDS. These are round-robin comparisons of six
    # components on the same items, whose dependence includes negative parts,
    # so BH's guarantee does not hold and BY is used instead. And the family is
    # deliberately the largest available: every smaller one would be less
    # strict, i.e. would favour this thesis.
    try:
        cells = {"x_2025": [
            {"bootstrap": {"p_value": 0.001}, "mcnemar": {"p_value": 0.002}},
            {"bootstrap": {"p_value": 0.400}, "mcnemar": {"p_value": 0.500}},
        ]}
        meta = sig.apply_multiplicity(cells)
        assert meta["m"] == 2
        assert "benjamini_hochberg" not in meta["methods"], (
            "BH reappeared in the reported methods; its PRDS assumption is not "
            "supportable for a round-robin design")
        assert "benjamini_hochberg" in meta["method_not_used"], (
            "the artefact no longer records WHY BH is absent -- an omission "
            "nobody can see is indistinguishable from an oversight")
        assert meta["by_factor_sum_1_over_i"] == 1 + 1/2
        r0 = cells["x_2025"][0]["multiplicity"]
        assert r0["holm_bootstrap"] == 0.002 and r0["survives_holm_bootstrap"] is True
        r1 = cells["x_2025"][1]["multiplicity"]
        assert r1["survives_holm_bootstrap"] is False
        assert r0["primary"].startswith("holm_bootstrap"), (
            "the primary correction is no longer the one on the paired macro-F1 "
            "p-value -- correcting McNemar alone corrects a quantity no claim "
            "rests on")
        assert len(meta["multiplicity_does_not_cover"]) >= 2
        ok("the multiplicity block corrects the PRIMARY-metric p-value, keeps BH "
           "out with its reason on file, and states what it does not cover")
    except AssertionError as e:
        bad("multiplicity block: {}".format(e))
    except Exception as e:
        bad("multiplicity block raised {}: {}".format(type(e).__name__, e))

    # --- E4 layer (src/significance_e4.py) -------------
    # The reference of every E4 comparison is fixed by the log, not by whatever
    # key sorts first: E4a against its OWN `none` cell at three folds (never the
    # five-fold E2 run), E4b against its E2 run, E4c only within 2026.
    from src import significance_e4 as s4
    keys = [
        "c2a_2025/encoder_1b__seed42",
        "c2a_2025/encoder_1b__e4a-none__seed42",
        "c2a_2025/encoder_1b__e4a-focal_loss__seed42",
        "c2a_2025/encoder_1b__e4b__seed42",
        "c2a_2025/tml_svm__e4a-none__seed42",
        "c2a_2025/tml_svm__e4a-class_weighting__seed42",
        "c2a_2025/tml_svm__seed42",
        "c2a_2025/tml_svm__seed43",
        "dbo_2026/encoder_1b__e4a-none__seed42",
        "dbo_2026/encoder_1b__e4a-class_weighting__seed42",
        "c2a_2026/encoder_1b__e4c__seed42",
        "c2a_2026/tml_svm__e4c__seed42",
        "c2a_2026/llm_llammlein__e4c__seed42",
        "dbo_2026/tml_svm__e4c__seed42",
    ]
    try:
        plan = s4.plan_pairs(keys)
        e4a = [q for q in plan if q["block"] == "e4a"]
        assert sorted((q["reference"], q["treatment"]) for q in e4a) == sorted([
            ("c2a_2025/encoder_1b__e4a-none__seed42", "c2a_2025/encoder_1b__e4a-focal_loss__seed42"),
            ("c2a_2025/tml_svm__e4a-none__seed42", "c2a_2025/tml_svm__e4a-class_weighting__seed42"),
            ("dbo_2026/encoder_1b__e4a-none__seed42", "dbo_2026/encoder_1b__e4a-class_weighting__seed42"),
        ]), "E4a pairs are not condition-vs-own-none within one component and cell"
        assert all("__e4a-none__" in q["reference"] for q in e4a), "an E4a reference is not a none cell"
        e4b = [q for q in plan if q["block"] == "e4b"]
        assert [(q["reference"], q["treatment"]) for q in e4b] == [
            ("c2a_2025/encoder_1b__seed42", "c2a_2025/encoder_1b__e4b__seed42")], "E4b is not paired with its E2 run"
        e4c = [q for q in plan if q["block"] == "e4c"]
        assert len(e4c) == 3 and all(q["reference"].startswith("c2a_2026/") and q["treatment"].startswith("c2a_2026/")
                                     for q in e4c), "E4c pairs cross a subtask or are incomplete"
        assert not any("seed43" in q["reference"] or "seed43" in q["treatment"] for q in plan), "a second partition seed was paired"
        ok("E4 plan: E4a vs its own none cell, E4b vs its E2 run, E4c within 2026 only, one seed")
    except AssertionError as e:
        bad("E4 plan: {}".format(e))
    try:
        s4.plan_pairs(["c2a_2025/encoder_1b__e4a-focal_loss__seed42"])
        bad("an E4a condition without its none cell is refused")
    except ValueError:
        ok("an E4a condition without its none cell is refused")
    try:
        s4.plan_pairs(["c2a_2025/encoder_1b__e4b__seed42"])
        bad("an E4b cell without its E2 run is refused")
    except ValueError:
        ok("an E4b cell without its E2 run is refused")
    real = s4.plan_pairs(ComponentStore().list_runs())
    counts = {b: sum(1 for q in real if q["block"] == b) for b in ("e4a", "e4b", "e4c")}
    if counts == {"e4a": 48, "e4b": 15, "e4c": 45}:
        ok("E4 plan on the real store: 48 / 15 / 45 comparisons, one Holm family per experiment (NS 123 h24)")
    else:
        bad("E4 plan on the real store: {} (NS 124 d fixed 48 / 15 / 45)".format(counts))

    # --- one Holm family per experiment -------------------------
    # A family is corrected on its own p-values only: adding an unrelated family
    # must not move a single adjusted value. Mutation: pool the families inside
    # correct_per_family and the first check turns red.
    import json as _json
    from src import multiplicity_by_experiment as mbe
    def _r(pb, pm=0.5):
        return {"bootstrap": {"p_value": pb}, "mcnemar": {"p_value": pm}, "reference": "r", "treatment": "t"}
    alone = mbe.correct_per_family({"x": [_r(0.01), _r(0.02)]})
    both = mbe.correct_per_family({"x": [_r(0.01), _r(0.02)], "y": [_r(0.001), _r(0.9), _r(0.03)]})
    ax = [row["multiplicity"]["holm_bootstrap"] for row in alone["families"]["x"]["rows"]]
    bx = [row["multiplicity"]["holm_bootstrap"] for row in both["families"]["x"]["rows"]]
    (ok if ax == bx == [0.02, 0.02] and both["families"]["y"]["m"] == 3
     else bad)("per-experiment Holm: a family's adjusted p-values do not depend on any other family")
    e4 = mbe.correct_per_family(mbe.e4_families(_json.loads(mbe.E4_SOURCE.read_text(encoding="utf-8"))))
    b12 = mbe.correct_per_family(mbe.b12_families(_json.loads(mbe.B12_SOURCE.read_text(encoding="utf-8"))))
    (ok if {k: v["m"] for k, v in e4["families"].items()} == {"e4a": 48, "e4b": 15, "e4c": 45}
     and {k: v["m"] for k, v in b12["families"].items()} == {"b12_2025": 18, "b12_2026": 18}
     else bad)("per-experiment families on the stored artefacts: E4 48 / 15 / 45, B12 18 per parent experiment")

    print("\n" + "=" * 50)
    print("  {} passed, {} failed".format(_PASS, _FAIL))
    print("=" * 50)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
