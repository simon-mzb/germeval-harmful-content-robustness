"""
e4_config.py -- freeze the one configuration E4a and E4b run at, before either runs.

WHY A FROZEN ARTEFACT AND NOT A LOOKUP INSIDE THE RUNNER. E4a and E4b do not
tune: E4a varies the imbalance condition and E4b holds one configuration fixed
across the three subtasks, and both are only interpretable if the configuration
was chosen by a rule stated before any E4 number existed: picking it after
seeing the results would build in an optimistic bias. The modal per-fold
winner was proposed on 2026-08-26, before a single encoder or LLM selection had
been recorded. `e4_protocol` in the matrix
states the rule; this module applies it to the E2 artefacts and writes the
result down with its tally, so the choice is a file with a date, not a value
computed on the fly that nobody can show was fixed in advance. `e4c_pool.py`
freezes Dimension 3's evaluation pool for the same reason.

THE RULE (e4_protocol.e4a.configuration / e4b.configuration):
  * the MODE of the selected configuration over the primary-seed (42) folds of
    E2 -- the 5 folds of the same subtask for E4a, all 15 for E4b;
  * a tie in the count is broken first towards the SMALLER CAPACITY (lower
    lora_r), the tie-break stated with the rule on 2026-08-26, wherever the
    grid has that axis;
  * a remaining tie is broken by the higher MEAN selection-slice macro-F1 of
    the tied configurations across those same folds. Every fold scored every
    grid cell, so the mean is defined for each of them;

WHAT THIS RULE CAN AND CANNOT CLAIM. "Applied unseen" would be too strong:
parts of the E2 selections were known when the rule was confirmed (the SVM's
C = 10 in 14 of 15 folds).
What holds instead, and is the stronger argument: within E4a all four
conditions run at the SAME configuration, so its choice cannot favour one
imbalance condition over another. Two directions of bias are stated, not hidden:
the configuration was tuned under `none`, so an E4a GAIN is understated and an
absent gain holds only "at this configuration"; and the E4b mode is read off
selection slices whose items are evaluated out-of-fold in other folds, so the
E4b stability penalty is a LOWER BOUND.
  * a tie that survives both is REFUSED rather than broken by list order;
  * dbo/2026 has no E2 selection of its own and takes dbo/2025's.

Usage
-----
    python -m src.e4_config            # describe and print the derivation; writes nothing
    python -m src.e4_config --write    # freeze results/e4_fixed_configurations.json
    python -m src.e4_config --verify   # re-derive and compare against the frozen file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
E2_DIR = _ROOT / "results" / "e2"
OUT_PATH = _ROOT / "results" / "e4_fixed_configurations.json"

COMPONENTS = ("tml_svm", "tml_xgboost", "tml_lightgbm", "encoder_1b", "llm_llammlein")
SUBTASKS = ("c2a", "dbo", "vio")
PRIMARY_SEED = 42


def _canon(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def fold_selections(component: str, subtask: str, e2_dir: Path = E2_DIR) -> list[dict]:
    """The primary-seed fold records of one E2 artefact: selected params + grid scores."""
    path = e2_dir / "e2_{}_2025_{}.json".format(subtask, component)
    art = json.loads(path.read_text(encoding="utf-8"))
    rep = next(r for r in art["repeats"] if r["seed"] == PRIMARY_SEED)
    out = []
    for rec in rep["folds_detail"]:
        out.append({
            "subtask": subtask, "fold": rec["fold"],
            "selected": dict(rec["selected_params"]),
            "grid_scores": {_canon(c["params"]): float(c["selection_macro_f1"])
                            for c in rec["selection_grid"]},
        })
    return out


def modal_configuration(folds: list[dict]) -> dict[str, Any]:
    """Apply the pre-stated rule to a list of fold selections."""
    tally: dict[str, int] = {}
    for f in folds:
        key = _canon(f["selected"])
        tally[key] = tally.get(key, 0) + 1
    top = max(tally.values())
    tied = sorted(k for k, n in tally.items() if n == top)
    means = None
    capacity_break = False
    # The tie-break stated with the rule: towards the smaller capacity. It
    # exists only where the grid has a capacity axis (the encoder's lora_r).
    if len(tied) > 1 and all("lora_r" in json.loads(k) for k in tied):
        smallest = min(json.loads(k)["lora_r"] for k in tied)
        narrowed = [k for k in tied if json.loads(k)["lora_r"] == smallest]
        capacity_break = len(narrowed) < len(tied)
        tied = narrowed
    if len(tied) > 1:
        means = {}
        for k in tied:
            scores = [f["grid_scores"][k] for f in folds if k in f["grid_scores"]]
            if len(scores) != len(folds):
                raise ValueError(
                    "tie-break needs every fold's selection score for {} and {} of "
                    "{} folds carry one".format(k, len(scores), len(folds)))
            means[k] = sum(scores) / len(scores)
        best = max(means.values())
        winners = [k for k, v in means.items() if v == best]
        if len(winners) > 1:
            raise ValueError(
                "a tie survives both the count and the mean selection score: {}. "
                "The rule refuses rather than breaking it by list order.".format(winners))
        chosen = winners[0]
    else:
        chosen = tied[0]
    return {
        "params": json.loads(chosen),
        "n_selections": len(folds),
        "tally": dict(sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))),
        "count_tie": top < len(folds) and sum(1 for n in tally.values() if n == top) > 1,
        "tie_broken_by_capacity": capacity_break,
        "tie_break_mean_selection_macro_f1": means,
    }


def protocol_digest(matrix: dict[str, Any]) -> str:
    """Hash of the configuration rules the freeze was derived under."""
    block = matrix.get("e4_protocol") or {}
    rules = {exp: (block.get(exp) or {}).get("configuration") for exp in ("e4a", "e4b")}
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:12]


def derive(e2_dir: Path = E2_DIR, components=COMPONENTS) -> dict[str, Any]:
    e4a: dict[str, Any] = {}
    e4b: dict[str, Any] = {}
    for c in components:
        per = {st: fold_selections(c, st, e2_dir) for st in SUBTASKS}
        e4a[c] = {"{}_2025".format(st): modal_configuration(per[st]) for st in SUBTASKS}
        e4a[c]["dbo_2026"] = dict(e4a[c]["dbo_2025"],
                                  borrowed_from="dbo_2025 (no E2 selection on 2026)")
        e4b[c] = modal_configuration([f for st in SUBTASKS for f in per[st]])
    return {"e4a": e4a, "e4b": e4b}


def fixed_params_for(component: str, subtask: str, edition: str, experiment: str,
                     path: Path | None = None, matrix: dict | None = None) -> dict:
    """The frozen configuration for one E4 cell. Refuses rather than derives."""
    path = Path(path) if path is not None else OUT_PATH
    if not path.exists():
        raise FileNotFoundError(
            "{} does not exist. The E4 configuration is FROZEN before any E4 run, "
            "not derived inside one: run `python -m src.e4_config --write` once "
            "e4_protocol is confirmed.".format(path))
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if matrix is not None and frozen.get("protocol_digest") != protocol_digest(matrix):
        raise ValueError(
            "{} was frozen under configuration rule {} and the matrix now states "
            "{}. The rule changed after the freeze; re-freezing is a protocol "
            "decision, not a refresh.".format(
                path.name, frozen.get("protocol_digest"), protocol_digest(matrix)))
    exp = experiment.lower()
    try:
        if exp == "e4a":
            return dict(frozen["e4a"][component]["{}_{}".format(subtask, edition)]["params"])
        if exp == "e4b":
            return dict(frozen["e4b"][component]["params"])
    except KeyError as e:
        raise KeyError("no frozen {} configuration for {} {}/{}".format(
            exp, component, subtask, edition)) from e
    raise ValueError("unknown E4 experiment {!r}".format(experiment))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="freeze the artefact")
    ap.add_argument("--verify", action="store_true", help="re-derive and compare")
    ap.add_argument("--force", action="store_true", help="overwrite a frozen artefact")
    args = ap.parse_args(argv)

    from src.component_store import load_matrix
    matrix = load_matrix()

    # ⚠️ NOT EVEN A DESCRIPTION UNTIL THE RULE IS CONFIRMED. Applying the rule
    # prints which configuration each cell gets; seen before the rule is
    # confirmed, that output is exactly the look at the outcome a
    # pre-registration exists to rule out -- a rule could then be "confirmed"
    # or "adjusted" knowing what it selects. So a proposed protocol gets the
    # usage text and nothing derived.
    status = str((matrix.get("e4_protocol") or {}).get("status", ""))
    if not status.startswith("confirmed"):
        print((__doc__ or "").strip())
        print("\nNOT DERIVED: e4_protocol.status is {!r}. The rule is applied to "
              "the E2 selections only once it is confirmed, so that nobody "
              "has seen what it selects before fixing it."
              .format(status))
        return 0 if not (args.write or args.verify) else 1

    derived = derive()

    for exp in ("e4a", "e4b"):
        print("\n{}:".format(exp.upper()))
        for c, entry in derived[exp].items():
            rows = entry.items() if exp == "e4a" else [("all 15 folds", entry)]
            for cell, r in rows:
                print("  {:<14s} {:<13s} {}  tally {}{}".format(
                    c, cell, r["params"], list(r["tally"].values()),
                    "  (count tie, broken by mean selection score)" if r["count_tie"] else ""))

    if args.verify:
        if not OUT_PATH.exists():
            print("\nVERIFY: {} does not exist".format(OUT_PATH))
            return 1
        frozen = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        same = (frozen["e4a"] == derived["e4a"] and frozen["e4b"] == derived["e4b"])
        print("\nVERIFY: {}".format("matches the frozen artefact" if same
                                    else "DIFFERS from the frozen artefact"))
        return 0 if same else 1

    if not args.write:
        print("\n(describe only -- nothing written; --write freezes it)")
        return 0

    status = str((matrix.get("e4_protocol") or {}).get("status", ""))
    if not status.startswith("confirmed"):
        print("\nREFUSED: e4_protocol.status is {!r}. The configuration rule is "
              "applied only once it is confirmed; freezing under a proposed rule "
              "would make the pre-registration a formality.".format(status))
        return 1
    if OUT_PATH.exists() and not args.force:
        print("\nREFUSED: {} is already frozen. Overwriting a freeze is a protocol "
              "decision; pass --force only if you mean it.".format(OUT_PATH.name))
        return 1

    payload = {
        "what": "the fixed configuration of every E4a and E4b cell, frozen before any E4 run",
        "rule": {exp: matrix["e4_protocol"][exp]["configuration"] for exp in ("e4a", "e4b")},
        "rule_first_proposed": "THESIS_LOG #82, 2026-08-26 (E4b)",
        "protocol_digest": protocol_digest(matrix),
        "source": "results/e2/e2_<subtask>_2025_<component>.json, seed {} folds_detail".format(
            PRIMARY_SEED),
        **derived,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nwrote {}".format(OUT_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
