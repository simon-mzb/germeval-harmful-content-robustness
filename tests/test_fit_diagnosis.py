"""Does the over/underfit alarm actually fire?

A fine-tune that runs for hours and then turns out to over- or underfit is
thrown-away time. Best-epoch selection protects the number;
diagnose_fit protects the PERSON, by saying the run sat at the edge of its
budget. Both halves are tested here -- including that a healthy run stays
quiet, because an alarm that always fires is not an alarm.

Usage: .venv/bin/python -m tests.test_fit_diagnosis
"""
import sys

from src.encoder_components import diagnose_fit


def trace(*scores):
    return [{"epoch": i + 1, "selection_macro_f1": s} for i, s in enumerate(scores)]


PASS = FAIL = 0


def check(name, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print("  PASS  {}".format(name))
    else:
        FAIL += 1
        print("  FAIL  {} (expected {!r}, got {!r})".format(name, want, got))


def has(name, d, flag):
    check(name, flag in d["flags"], True)


def hasnt(name, d, flag):
    check(name, flag in d["flags"], False)


print("\n=== underfitting: the cap stopped a model that was still improving ===")
d = diagnose_fit(trace(0.40, 0.50, 0.58, 0.63, 0.69), 5, 5, 0.69, 4)
has("best epoch == max_epochs is flagged", d, "budget_binding")
check("  ...and the run is not called ok", d["ok"], False)
check("  ...and it says to raise the cap for ALL folds", any("ALL folds" in n for n in d["notes"]), True)

print("\n=== the healthy case must stay silent ===")
d = diagnose_fit(trace(0.40, 0.61, 0.66, 0.64, 0.62), 3, 5, 0.66, 4)
check("a mid-budget peak raises nothing", d["flags"], [])
check("  ...and is reported ok", d["ok"], True)

print("\n=== overfitting inside one pass ===")
d = diagnose_fit(trace(0.62, 0.51, 0.44), 1, 5, 0.62, 4)
has("peak at epoch 1 then decline is flagged", d, "collapsed_early")
hasnt("  ...and it is NOT called budget-binding", d, "budget_binding")

print("\n=== a peak at epoch 1 that does NOT decline is not overfitting ===")
d = diagnose_fit(trace(0.62, 0.62, 0.63), 1, 5, 0.62, 4)
hasnt("flat-then-flat is not collapse", d, "collapsed_early")

print("\n=== nothing was learned ===")
d = diagnose_fit(trace(0.25, 0.25, 0.25), 1, 5, 0.25, 4)
has("a frozen score is flagged", d, "degenerate")

print("\n=== the imbalance trap: DBO subversive is 0.80% of the data ===")
# A model that predicts only the majority class on a 4-class task where the
# majority holds 90% scores macro-F1 ~0.237. That must not read as success.
d = diagnose_fit(trace(0.23, 0.23), 1, 5, 0.23, 4, majority_share=0.90)
has("a majority-only model is flagged", d, "no_better_than_majority")
d = diagnose_fit(trace(0.23, 0.55), 2, 5, 0.55, 4, majority_share=0.90)
hasnt("  ...but a model that beats the floor is not", d, "no_better_than_majority")

print("\n=== the replicate path must be diagnosed too (regression, 2026-08-27) ===")
# run_seed_replicate computed the diagnosis and then dropped it from its return
# value, so _report_fit_diagnoses skipped every replicate. It surfaced only when
# the first real run's replicate seed 44 selected epoch 5 of 5 -- a budget_binding
# case that went unreported while the folds were being called healthy.
import inspect

from src import e2_runner

src_txt = inspect.getsource(e2_runner.run_seed_replicate)
tail = src_txt[src_txt.rindex("return {"):]
check("run_seed_replicate returns fit_diagnosis", "fit_diagnosis" in tail, True)

# And the reporter must actually count a replicate that carries one.
import contextlib
import io

res = {"repeats": [{"folds_detail": [
           {"selected_epoch": 3, "fit_diagnosis": {"flags": [], "notes": []}}]}],
       "seed_replicates": [
           {"selected_epoch": 5, "fit_diagnosis":
               {"flags": ["budget_binding"], "notes": ["best epoch = 5 = max_epochs"]}}]}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    e2_runner._report_fit_diagnoses(res)
out = buf.getvalue()
check("a budget-binding replicate is reported", "budget_binding" in out, True)
check("  ...and counted against both records", "1 of 2 folds" in out, True)

print("\n=== it must never crash on missing data ===")
d = diagnose_fit(None, None, None, None, 4)
check("no trace at all does not raise", d["ok"], True)
d = diagnose_fit([], 1, 5, 0.5, 4)
check("an empty trace does not raise", isinstance(d["flags"], list), True)


# ---------------------------------------------------------------------------
# v1.7: early stopping with patience closes the epoch axis
# ---------------------------------------------------------------------------

def test_early_stopping():
    """The stopping rule itself, exercised on the traces that motivated it.

    A hard cut at five epochs has to be defensible and it was not -- the vio probe peaked at 5 of 5, which is the cap binding. Patience is
    derived from data: across all 14 epoch traces this project has produced, the
    longest run of consecutive non-improving epochs that was still FOLLOWED by a
    new best is 1 (in 8 of the 14), and it is never 2. These are the real traces.
    """
    import src.encoder_components as enc

    print("\n=== early stopping (v1.7): patience 2 clears every observed dip ===")

    def stop_epoch(scores, patience):
        """Where early stopping would end, and which epoch it would select."""
        best_i, best, since = -1, -1.0, 0
        for i, s in enumerate(scores):
            if s > best:
                best, best_i, since = s, i, 0
            else:
                since += 1
                if since >= patience:
                    return i + 1, best_i + 1, best
        return len(scores), best_i + 1, best

    # The trace that decides it: 2026-08-26 c2a, r=8 lr=1e-4, fold 1. It dips at
    # epoch 2, recovers at 3, dips at 4, and sets its BEST at 5.
    decisive = [0.7926, 0.7818, 0.8034, 0.7818, 0.8039]
    _, sel1, sc1 = stop_epoch(decisive, 1)
    _, sel2, sc2 = stop_epoch(decisive, 2)
    check("patience 1 truncates the decisive trace at a worse epoch", sel1, 1)
    check("  ...losing the best score it would have found", round(sc2 - sc1, 4), 0.0113)
    check("patience 2 still reaches the real best", sel2, 5)

    # Every trace on file: patience 2 must never select a worse epoch than an
    # exhaustive scan of the same range would.
    traces = [
        [0.5013, 0.6111, 0.7827, 0.7873, 0.7692],
        [0.6590, 0.6035, 0.7646, 0.7873, 0.7757],
        [0.6781, 0.6111, 0.7650, 0.7412, 0.7463],
        [0.7131, 0.6667, 0.7712, 0.7988, 0.7870],
        [0.7926, 0.7818, 0.8034, 0.7818, 0.8039],
        [0.7329, 0.7930, 0.7787, 0.8202, 0.8047],
        [0.7442, 0.7876, 0.8135, 0.7623, 0.7764],
        [0.7237, 0.7876, 0.8180, 0.7930, 0.8043],
        [0.8166, 0.8021, 0.8506, 0.8425, 0.8673],
        [0.8059, 0.8514, 0.8245, 0.8327, 0.8358],
        [0.8021, 0.8610, 0.8765, 0.8617, 0.8610],
        [0.5463, 0.8149, 0.8452, 0.8182, 0.8659],
        [0.4130, 0.5156, 0.6228, 0.6551, 0.6470],   # dbo
        [0.7956, 0.7853, 0.8158, 0.7743, 0.8256],   # vio -- peaked at the old cap
    ]
    lost_at_1 = sum(1 for t in traces if stop_epoch(t, 1)[2] < max(t) - 1e-9)
    lost_at_2 = sum(1 for t in traces if stop_epoch(t, 2)[2] < max(t) - 1e-9)
    check("patience 1 would have lost the best model in 8 of 14 traces", lost_at_1, 8)
    check("patience 2 loses it in none of them", lost_at_2, 0)

    # And the ceiling: 8 = highest observed peak (5) + patience (2) + margin (1).
    check("the ceiling is read from the matrix, not hardcoded", enc.MAX_EPOCHS, 8)
    check("so is the patience", enc.PATIENCE, 2)
    over = [t for t in traces if stop_epoch(t, enc.PATIENCE)[0] > enc.MAX_EPOCHS]
    check("no trace on file would reach the ceiling", len(over), 0)




test_early_stopping()

print("\n" + "-" * 40)
print("  {} passed, {} failed".format(PASS, FAIL))
print("-" * 40)

sys.exit(1 if FAIL else 0)
