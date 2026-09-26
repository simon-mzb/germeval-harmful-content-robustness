"""Does load_many refuse the mismatches that actually happened?

#83a: config_id and data_rules_id were the only checks, and both mismatches
that occurred in practice slipped through -- a 2-fold encoder stored beside
5-fold TML runs, and a Linux entry beside Darwin ones. Neither is visible in
the arrays: pooled out-of-fold probabilities have the same shape whatever the
fold count, so the entries look like peers to E3.

Usage: .venv/bin/python -m tests.test_store_generation
"""
import sys
from unittest import mock

from src import component_store as cs

PASS = FAIL = 0


def check(name, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print("  PASS  {}".format(name))
    else:
        FAIL += 1
        print("  FAIL  {} (expected {!r}, got {!r})".format(name, want, got))


def rec(component, n_splits, machine, cid="CID", rules="RULES"):
    return {"meta": {"component": component, "n_splits": n_splits,
                     "machine": machine, "config_id": cid,
                     "data_rules_id": rules}}


def run(recs, want_warnings=False):
    """Drive load_many's generation check with canned sidecars.

    Returns the RuntimeError text, or ("", [warning messages]) when asked --
    because machine mismatches warn rather than raise (see below).
    """
    store = cs.ComponentStore.__new__(cs.ComponentStore)
    with mock.patch.object(cs.ComponentStore, "load", side_effect=lambda k: recs[k]), \
         mock.patch.object(cs, "load_matrix", return_value={}), \
         mock.patch.object(cs, "compute_config_id", return_value="CID"), \
         mock.patch.object(cs, "compute_data_rules_id", return_value="RULES"):
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            try:
                cs.ComponentStore.load_many(store, list(recs), check_generation=True)
                err = ""
            except RuntimeError as e:
                err = str(e)
        msgs = [str(x.message) for x in caught]
        return (err, msgs) if want_warnings else err


print("\n=== the healthy case must load ===")
err = run({"a": rec("tml_svm", 5, "Darwin/arm64"),
           "b": rec("encoder_1b", 5, "Darwin/arm64")})
check("matching folds and machine load without complaint", err, "")

print("\n=== the mismatch that actually occurred ===")
err, warns = run({"svm": rec("tml_svm", 5, "Darwin/arm64"),
                  "enc": rec("encoder_1b", 2, "Linux/x86_64")}, want_warnings=True)
check("a 2-fold entry beside 5-fold ones is refused", "mixed fold counts" in err, True)
check("  ...naming the offending keys", "enc=2" in err, True)
# The machine difference WARNS and does not raise. Raising would have blocked
# E3 outright the moment B4 landed -- TML is Darwin, the encoder arm is Linux
# by design -- and it is a provenance fact, not an incomparability: C2A
# reproduced bit-identically across three platforms.
check("  ...while the machine difference only warns",
      any("several machines" in w for w in warns), True)

print("\n=== each axis alone ===")
err = run({"a": rec("tml_svm", 5, "Darwin/arm64"),
           "b": rec("encoder_1b", 2, "Darwin/arm64")})
check("folds alone are enough to refuse", "mixed fold counts" in err, True)
check("  ...without inventing a machine problem", "mixed machines" in err, False)

err, warns = run({"a": rec("tml_svm", 5, "Darwin/arm64"),
                  "b": rec("encoder_1b", 5, "Linux/x86_64")}, want_warnings=True)
check("machine alone does NOT block the load", err, "")
check("  ...but is warned about", any("several machines" in w for w in warns), True)
check("  ...and invents no fold problem", any("fold" in w for w in warns), False)

print("\n=== a sidecar with no n_splits cannot be checked, and must say so ===")
err = run({"a": rec("tml_svm", 5, "Darwin/arm64"),
           "b": rec("encoder_1b", None, "Darwin/arm64")})
check("a missing n_splits is refused, not assumed", "no n_splits" in err, True)

print("\n=== the pre-existing checks still work ===")
err = run({"a": rec("tml_svm", 5, "Darwin/arm64", cid="STALE")})
check("a drifted config_id still refuses", "config_id" in err, True)
err = run({"a": rec("tml_svm", 5, "Darwin/arm64", rules="OLD")})
check("a drifted data_rules_id still refuses", "data_rules_id" in err, True)

print("\n" + "-" * 40)
print("  {} passed, {} failed".format(PASS, FAIL))
print("-" * 40)
sys.exit(1 if FAIL else 0)
