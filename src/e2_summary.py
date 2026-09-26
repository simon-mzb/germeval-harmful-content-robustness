"""
e2_summary.py -- aggregation and gate evidence for the E2 component results.

The runner (`src/e2_runner.py`) owns the measurement; this module owns the
reading of it. Keeping the two apart means the summary tables, the figures and
the G2 evidence (the component gate: does any component clear the baseline?)
are all recomputed from the same JSON artefacts and cannot
drift apart from each other or from the component store.

Gate files follow a preservation pattern: the code owns the measured numbers, the human owns
`decision` and `decision_note`, and re-running must never quietly reopen a
gate that was closed.

Usage
-----
from src.e2_summary import (
    collect_runs, summary_table, per_class_table, baseline_comparison,
    reliability_figure, write_summary, write_g2_gate, load_baselines,
)
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.component_store import compute_data_rules_id
from src.plotstyle import (COLOURS, FULL_WIDTH, REFERENCE_COLOUR,
                          apply_thesis_style, save_figure)

_ROOT = Path(__file__).parent.parent
RESULTS_DIR = _ROOT / "results"
E2_DIR = RESULTS_DIR / "e2"

# The organiser reference points. These are literals on purpose: they are
# published figures measured by someone else on a gold test set we do not have,
# so there is no artefact in this tree they could be read from.
ORGANIZER = {"c2a": 59.13, "dbo": 47.44, "vio": 68.97}

# Our own reproduced baselines are the opposite case and are never literals
# here. A constant copied from the E1 artefacts goes stale the moment those
# artefacts are recomputed (for instance under new data rules), and the
# comparison then subtracts across a machine *and* a rule generation. A stored
# constant must either be recomputed from its source or name the generation it
# belongs to; this reads it from the source.
#
# Discovery patterns, not filenames. An E1 reference is a *set* keyed by the
# machine that produced it: the encoder and LLM arms ran on a rented Linux GPU
# machine while the TML arm is Darwin/arm64, so whichever single
# artefact were named here would leave the other side of the comparison with no
# reference at all. Nothing is special-cased -- `e1_c2a_baseline.json` is found
# by the same glob as any sibling and is keyed by the platform it carries in its
# own `env_pin`, so a machine is never asserted here, only read.
# For VIO the pattern finds both the reported reference and the machine-keyed
# reproduction (E1b); the reference carries no macro_f1 and is skipped.
E1_PATTERNS = {
    "c2a": "e1_c2a_baseline*.json",
    "dbo": "e1_dbo_baseline*.json",
    "vio": "e1_vio_*.json",           # reported reference + E1b reproductions
}


def load_baselines(results_dir: Path = RESULTS_DIR) -> dict[str, dict[str, Any]]:
    """
    Read the E1 reference points from their artefacts, keyed by machine.

    Returns per subtask the organiser figure plus a `by_machine` map: platform
    string -> {reproduced (percentage points), data_rules_id, source}. The
    provenance is what `baseline_comparison` needs in order to refuse a delta it
    must not compute; a value without it would leave the check to discipline,
    which is what failed here twice.

    Two artefacts claiming the same machine for one subtask is a hard error, for
    the same reason `component_store.load_many` raises on mixed generations: the
    module cannot know which one the caller meant, and must not pick silently.
    """
    out: dict[str, dict[str, Any]] = {}
    for st, pattern in E1_PATTERNS.items():
        by_machine: dict[str, dict[str, Any]] = {}
        unusable: list[str] = []
        for path in sorted(Path(results_dir).glob(pattern)):
            art = json.loads(path.read_text(encoding="utf-8"))
            if art.get("macro_f1") is None:
                continue                       # a reference, not a reproduction
            machine = (art.get("env_pin") or {}).get("platform")
            if machine is None:
                unusable.append(path.name)     # cannot be compared against
                continue
            if machine in by_machine:
                raise ValueError(
                    f"two E1 artefacts claim machine {machine!r} for {st}: "
                    f"{by_machine[machine]['source']} and {path.name}. "
                    "Remove or rename one; a baseline set may hold each machine once."
                )
            by_machine[machine] = {
                "reproduced": 100 * float(art["macro_f1"]),
                "data_rules_id": art.get("data_rules_id"),
                "source": path.name,
            }
        out[st] = {
            "organizer": ORGANIZER[st],
            "by_machine": by_machine,
            "unusable_sources": unusable,
        }
    return out


def baseline_for(baselines: dict[str, dict[str, Any]],
                 subtask: str,
                 machine: str | None) -> dict[str, Any]:
    """
    Select the E1 reference measured on `machine`, or state why there is none.

    Flat record so callers read one shape whether or not a match exists:
    `reproduced` is None exactly when no usable same-machine artefact was found,
    and `blocked_reason` then says which of the three cases it is.
    """
    entry = baselines[subtask]
    by_machine = entry["by_machine"]
    record: dict[str, Any] = {
        "reproduced": None,
        "organizer": entry["organizer"],
        "reproduced_source": None,
        "machine": None,
        "data_rules_id": None,
        "available_machines": sorted(by_machine),
        "blocked_reason": None,
    }
    if not by_machine:
        note = ("no reproduced E1 baseline exists for this subtask"
                if not entry["unusable_sources"] else
                "E1 artefact(s) {} record no machine".format(
                    ", ".join(entry["unusable_sources"])))
        record["blocked_reason"] = note
        return record
    if machine is None:
        record["blocked_reason"] = "machine not recorded on the E2 side"
        return record
    if machine not in by_machine:
        record["blocked_reason"] = (
            "no E1 baseline measured on {} (have: {})".format(
                machine, ", ".join(sorted(by_machine))))
        return record
    hit = by_machine[machine]
    record.update(reproduced=hit["reproduced"], machine=machine,
                  data_rules_id=hit["data_rules_id"],
                  reproduced_source=hit["source"])
    return record


# Lowest top-label confidence any of these subtasks can produce (1/K).
CONFIDENCE_FLOOR = 0.25

SUBTASK_ORDER = ("c2a", "dbo", "vio")
COMPONENT_ORDER = ("tml_svm", "tml_xgboost", "tml_lightgbm",
                   "encoder_1b", "encoder_134m")


def collect_runs(e2_dir: Path = E2_DIR) -> list[dict[str, Any]]:
    """Load every E2 result JSON, newest schema only, sorted for stable tables."""
    runs = []
    for path in sorted(Path(e2_dir).glob("e2_*.json")):
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    order = {c: i for i, c in enumerate(COMPONENT_ORDER)}
    sub = {s: i for i, s in enumerate(SUBTASK_ORDER)}
    runs.sort(key=lambda r: (sub.get(r["subtask"], 99), order.get(r["component"], 99)))
    return runs


def summary_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """
    One row per (subtask, component): the headline numbers for Chapter 4.

    macro_f1 is the mean over the three repeats of the within-repeat pooled
    out-of-fold score and is the figure Chapter 4 reports. The bootstrap
    interval belongs to the primary repeat alone, so macro_f1_primary is
    carried alongside it: the interval brackets that value, not the mean, and
    reading it around the mean would be a category error. seed_sd and fold_sd
    are the two dispersion statistics the protocol reports separately rather
    than merged into one number.

    The two families fill seed_sd from different quantities, which is why the
    table carries seed_sd_basis beside it. For the classical components it is
    the spread of three pooled out-of-fold estimates over three partitions; for
    the encoder, whose partition is held fixed and whose seed varies training
    it is the spread of one fold retrained under three seeds. Merging the
    two into one unlabelled column would invite exactly the comparison neither
    number supports.
    """
    rows = []
    for r in runs:
        p, d = r["primary"], r["dispersion"]
        rows.append({
            "subtask": r["subtask"],
            "edition": r["edition"],
            "component": r["component"],
            "macro_f1": 100 * d["mean_pooled_macro_f1"],
            "macro_f1_primary": 100 * p["macro_f1"],
            "ci_lower": 100 * p["macro_f1_ci_lower"],
            "ci_upper": 100 * p["macro_f1_ci_upper"],
            "seed_sd": 100 * (d["replicate_sd_macro_f1"] if r.get("seed_replicates")
                              else d["seed_sd_macro_f1"]),
            "seed_sd_basis": ("fold {} replicates".format(
                r["seed_replicates"][0]["fold"]) if r.get("seed_replicates")
                else "{} partition repeats".format(len(d["per_seed_pooled_macro_f1"]))),
            "fold_sd": 100 * d["fold_sd_macro_f1_primary"],
            "weighted_f1": 100 * p["weighted_f1"],
            "ece": p["calibration"]["ece"],
            "brier": p["calibration"]["brier"],
            "n": p["n_samples"],
            # Prefer the summed per-fold training time: on a cell resumed from
            # checkpoints the top-level wall clock covers only the retrained
            # folds. Older artefacts have neither field set and fall back.
            "runtime_min": ((_fold_runtime_sum(r) or r["runtime_seconds"]) / 60.0),
            "runtime_is_wall_clock": _fold_runtime_sum(r) is None,
            "resumed": bool(r.get("resumed", False)),
            "machine": (r["meta"].get("env_pin") or {}).get("platform"),
            "data_rules_id": r["meta"].get("data_rules_id"),
            # ⚠️ The fold count travels with every row, so a 2-fold measurement
            # run next to 5-fold campaign rows cannot read as a peer. The
            # count comes from the artefact itself (how many folds it actually
            # recorded), not from a field it claims, because a field can be stale.
            "n_folds": _fold_count(r),
        })
    df = pd.DataFrame(rows)
    _warn_on_mixed_folds(df)
    return df


def _fold_count(run: dict) -> int | None:
    """How many folds this artefact actually recorded, COUNTED not claimed.

    `meta.n_splits` records what the run was asked for; this counts what it
    produced. They should agree, and a disagreement is worth a warning rather
    than a silent preference for either: a claim can go stale when a run is
    resumed or edited, and the data can be truncated by a crash. Counting wins
    for the table because it is what the numbers were actually computed from.
    """
    repeats = run.get("repeats") or []
    if not repeats:
        return None
    detail = repeats[0].get("folds_detail")
    actual = None
    if detail is not None:
        actual = len(detail)
    else:
        per_fold = repeats[0].get("per_fold_macro_f1")
        actual = len(per_fold) if per_fold is not None else None
    claimed = (run.get("meta") or {}).get("n_splits")
    if actual is not None and claimed is not None and actual != claimed:
        warnings.warn(
            "E2 artefact {}/{}: meta.n_splits says {} but {} folds are recorded. "
            "The table uses the recorded count. A run that was interrupted, "
            "resumed or hand-edited is the usual cause -- do not report it "
            "until the difference is explained.".format(
                run.get("subtask"), run.get("component"), claimed, actual),
            RuntimeWarning, stacklevel=3)
    return actual


def _warn_on_mixed_folds(df: pd.DataFrame) -> None:
    """Say loudly when one subtask's components were not run alike.

    Not a raise: the summary is also how you LOOK at a mixed tree in order to
    fix it, and a table that refuses to render is useless for that. But it must
    never render silently.
    """
    if "n_folds" not in df.columns or df.empty:
        return
    for subtask, grp in df.groupby("subtask"):
        counts = {int(v) for v in grp["n_folds"].dropna().unique()}
        if len(counts) > 1:
            rows = ", ".join("{}={}".format(c, int(n)) for c, n in
                             zip(grp["component"], grp["n_folds"]) if n == n)
            warnings.warn(
                "E2 summary: {} mixes fold counts across components ({}). "
                "These rows are NOT comparable and must not share a table. "
                "A measurement run probably leaked into results/e2/ -- see "
                "results/measurements/README.md.".format(subtask, rows),
                RuntimeWarning, stacklevel=2)
    machines = {m for m in df["machine"].dropna().unique()}
    if len(machines) > 1:
        warnings.warn(
            "E2 summary: results span several machines ({}). That is expected "
            "when the GPU components ran on another machine, but every cross-component "
            "comparison in a chapter must then say so.".format(
                ", ".join(sorted(machines))),
            RuntimeWarning, stacklevel=2)


def _fold_runtime_sum(run: dict) -> float | None:
    """Summed per-fold training time of the primary repeat, or None.

    Guarded rather than indexed directly: reading repeats[0] would give
    summary_table an IndexError on an artefact with no repeats, a failure mode
    it did not have before the resumed-cost field was added.
    """
    repeats = run.get("repeats") or []
    return repeats[0].get("fold_runtime_seconds_sum") if repeats else None


def per_class_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Per-class F1 of the primary repeat with its reporting tier and bootstrap interval.

    The tier travels with the number on purpose: a reader of the table can see
    without consulting the prose whether a value may carry a directional claim
    (n >= 100), needs non-overlapping intervals first (30 <= n < 100), or
    carries none at all (n < 30).
    """
    rows = []
    for r in runs:
        for cls, rec in r["primary"]["per_class"].items():
            rows.append({
                "subtask": r["subtask"],
                "component": r["component"],
                "class": cls,
                "f1": 100 * rec["f1"],
                "support": rec["support"],
                "tier": rec["tier"],
                "ci_lower": 100 * rec.get("ci_lower", float("nan")),
                "ci_upper": 100 * rec.get("ci_upper", float("nan")),
            })
    return pd.DataFrame(rows)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    DataFrame rows as JSON-serialisable records, with NaN written as null.

    `to_dict` leaves NaN in place and `json.dumps` writes it as the bare token
    NaN, which Python reads back but which is not valid JSON: any strict parser
    rejects the file. That matters beyond tidiness, because these result files
    are published with this repository and they have to stand on their own for a reader who is not using Python.
    """
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v)
             for k, v in rec.items()}
            for rec in df.to_dict(orient="records")]


def baseline_comparison(summary: pd.DataFrame,
                        baselines: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
    """
    Best TML component per subtask against the E1 reference points.

    The delta column is reported in percentage points and is explicitly not a
    like-for-like comparison: E1 measured a single 80/20 holdout, E2 pools
    out-of-fold predictions over the whole training split, and the organiser
    figures were measured on a gold test set we do not have.

    Against the *reproduced* baseline the delta is additionally withheld unless
    both sides were produced on the same machine under the same data-rules
    generation. Section 3.3 states the rule -- a performance difference is only
    ever computed between runs produced on the same machine -- and a rule of
    that kind that lives only in prose is not enforced by anything.
    `blocked_reason` says which side differs, so a withheld delta reads as a
    stated fact rather than a missing number. The delta against the *organiser*
    figure is never withheld: it spans not just machines but implementations
    and evaluation sets, which the eval_note has always said, and gating it on
    hardware would imply a comparability it never claimed.
    """
    baselines = baselines if baselines is not None else load_baselines()
    rules_now = compute_data_rules_id()
    rows = []
    for st in SUBTASK_ORDER:
        sub = summary[summary["subtask"] == st]
        if sub.empty:
            continue
        best = sub.loc[sub["macro_f1"].idxmax()]
        e2_machine = best.get("machine")
        # Selected by the *winning component's* machine rather than taken as the
        # only one on file: the same-machine E1 reference is looked up and the withholding is a real finding about
        # the artefacts rather than an artefact of the lookup.
        base = baseline_for(baselines, st, e2_machine)

        blocked: list[str] = []
        if base["reproduced"] is None:
            blocked.append(base["blocked_reason"])
        else:
            e2_rules = best.get("data_rules_id") or rules_now
            if base["data_rules_id"] is None:
                blocked.append("data-rules generation not recorded on the E1 artefact")
            elif base["data_rules_id"] != e2_rules:
                blocked.append("different data-rules generation (E1 {}, E2 {})".format(
                    base["data_rules_id"], e2_rules))

        rows.append({
            "subtask": st,
            "best_component": best["component"],
            "best_macro_f1": best["macro_f1"],
            "e1_reproduced": base["reproduced"],
            "e1_machine": base["machine"],
            "e1_source": base["reproduced_source"],
            "e2_machine": e2_machine,
            "delta_vs_reproduced_pp": (None if blocked
                                       else best["macro_f1"] - base["reproduced"]),
            # A boolean, not just the reason string. pandas stores a None in an
            # object column as NaN, and `bool(nan)` is True, so `if
            # row.blocked_reason` reads every *unblocked* row as blocked -- which
            # is how the first version of this table announced two withheld
            # deltas it had in fact computed. A flag cannot be misread that way.
            "delta_blocked": bool(blocked),
            "blocked_reason": "; ".join(blocked) if blocked else None,
            "organizer_reference": base["organizer"],
            "delta_vs_organizer_pp": best["macro_f1"] - base["organizer"],
        })
    return pd.DataFrame(rows)


def reliability_figure(runs: list[dict[str, Any]],
                       out_name: str = "reliability_tml") -> Path:
    """
    Reliability diagrams for every component, one column per subtask.

    The canonical two-panel form (DeGroot and Fienberg 1983; guo2017): the
    calibration curve on top, the bin populations directly beneath it on a
    logarithmic axis. Every measured bin is plotted and connected, and the
    panel below says how much each one weighs.

    That second panel is the whole point. Equal-width confidence bins leave
    the low-confidence end nearly empty -- on DBO the lowest occupied bin of
    one component holds a single item, whose accuracy can only be 0 or 1 --
    and an earlier version of this figure suppressed the connecting line below
    a minimum count to stop that from reading as a component collapsing.
    A threshold that decides which measured points get joined is a
    presentation choice with no principled basis, and it is exactly the kind
    of choice an examiner is right to challenge. Showing the counts instead
    filters nothing and explains more.

    Bins are equal-width because the reported ECE is (see calibration.py);
    the figure and the metric must be the same measurement.
    """
    import matplotlib.pyplot as plt

    apply_thesis_style()
    subtasks = [s for s in SUBTASK_ORDER if any(r["subtask"] == s for r in runs)]

    fig, axes = plt.subplots(
        2, len(subtasks),
        figsize=(FULL_WIDTH, 2.95),
        sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12, "wspace": 0.28},
    )

    handles: list = []
    labels: list[str] = []

    for col, st in enumerate(subtasks):
        ax, ax_n = axes[0][col], axes[1][col]
        ax.plot([CONFIDENCE_FLOOR, 1], [CONFIDENCE_FLOOR, 1],
                color=REFERENCE_COLOUR, linewidth=0.7,
                linestyle=(0, (4, 3)), zorder=1)

        panel = [r for r in runs if r["subtask"] == st]
        for r in panel:
            cal = r["primary"]["calibration"]
            bins = cal["reliability_bins"]
            xs = [b["mean_confidence"] for b in bins]
            ys = [b["accuracy"] for b in bins]
            ns = [b["n"] for b in bins]
            name = r["component"].replace("tml_", "")
            line, = ax.plot(xs, ys, marker="o", zorder=2, label=name)
            ax_n.plot(xs, ns, marker="o", markersize=2, linewidth=0.8,
                      color=line.get_color())
            if name not in labels:
                handles.append(line)
                labels.append(name)

        # ECE goes in the bottom-right corner, which a calibration curve always
        # leaves empty, and is colour-matched to its component. The legend is
        # shared across the whole figure instead of repeated three times, and
        # sits outside the axes: inside the panel the curve runs straight
        # through it, which an earlier version of this figure did.
        for i, r in enumerate(reversed(panel)):
            colour = COLOURS[len(panel) - 1 - i]
            ax.text(0.97, 0.04 + 0.085 * i,
                    "{} {:.3f}".format(r["component"].replace("tml_", ""),
                                       r["primary"]["calibration"]["ece"]),
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6, color=colour)
        ax.text(0.97, 0.04 + 0.085 * len(panel), "ECE", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color=REFERENCE_COLOUR)

        ax.set_title(st.upper())
        # A top-label confidence cannot fall below 1/K, so 0.25 is the floor
        # across every panel here (DBO has four classes, the others two).
        # Starting the axis there costs no data and buys a quarter of the width.
        ax.set_xlim(CONFIDENCE_FLOOR, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax_n.set_yscale("log")
        ax_n.set_ylim(0.6, 2e4)
        ax_n.set_yticks([1, 100, 10000])

    axes[0][0].set_ylabel("observed accuracy")
    axes[1][0].set_ylabel("items")
    # One x-label and one legend for the whole figure. Three copies of each is
    # noise, and at this height a bottom legend collides with the labels.
    fig.supxlabel("mean confidence", fontsize=8)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.05), columnspacing=2.5)
    return save_figure(fig, "e2", out_name)


def write_summary(runs: list[dict[str, Any]],
                  path: Path | None = None) -> Path:
    """Compact machine-readable summary of the whole arm, for Chapter 4 tables."""
    path = Path(path or (RESULTS_DIR / "e2_tml_summary.json"))
    summary = summary_table(runs)
    payload = {
        "experiment": "E2",
        "arm": "tml",
        "package": "B3",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aggregation": "pooled_out_of_fold, mean over 3 partition repeats",
        "reporting_rule": "D1 tiers: n>=100 interpreted, 30<=n<100 CI-gated, n<30 not interpreted",
        "imbalance_condition": "none (D4 reference cell)",
        "runs": _records(summary),
        "per_class": _records(per_class_table(runs)),
        "baseline_comparison": _records(baseline_comparison(summary)),
        "env_pin": runs[0]["meta"]["env_pin"] if runs else None,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# G2 gate evidence (preservation pattern)
# ---------------------------------------------------------------------------

_HUMAN_FIELDS = ("decision", "decision_note")

G2_CRITERION = ("at least one component meaningfully clears the reproduced baseline; "
                "decided after B6, when the encoder and LLM arms exist")


def _gate_env_pins(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The environments that produced this result set, keyed by machine.

    Returns both fields the gate file carries: `env_pin` for the single-machine
    case (unchanged for every gate written so far) and `env_pin_by_machine`,
    which is the honest shape once components come from several machines. A run
    whose artefact records no platform is grouped under "unknown" rather than
    dropped -- a missing environment is itself worth seeing in a gate file.
    """
    by_machine: dict[str, Any] = {}
    for r in runs:
        pin = r.get("meta", {}).get("env_pin") or {}
        key = pin.get("platform") or "unknown"
        by_machine.setdefault(key, pin)
    return {
        "env_pin": next(iter(by_machine.values())) if len(by_machine) == 1 else None,
        "env_pin_by_machine": by_machine,
    }


def write_g2_gate(runs: list[dict[str, Any]],
                  path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """
    Write (or refresh) the G2 gate file without touching the human judgement.

    The measured half is rebuilt from the current results on every call. The
    two human fields are carried over from an existing file unless they still
    start with "TODO", the same guard the E1 notebook uses for G1. Adding an arm
    later refreshes the evidence and leaves a closed gate closed.
    """
    path = Path(path or (RESULTS_DIR / "g2_gate_assessment.json"))
    summary = summary_table(runs)
    comparison = baseline_comparison(summary)

    arms_present = sorted({r["component"].split("_")[0] for r in runs})
    payload: dict[str, Any] = {
        "gate": "G2",
        "criterion": G2_CRITERION,
        # `status` describes EVIDENCE COVERAGE, not the judgement -- the gate
        # state is `decision`, which is the convention G1 and the G1.5 smoke
        # already use (neither carries a status field at all).
        "status": ("partial" if arms_present == ["tml"]
                   else "complete" if not [a for a in ("encoder", "llm")
                                           if a not in arms_present]
                   else "in_progress"),
        "arms_present": arms_present,
        "arms_pending": [a for a in ("encoder", "llm") if a not in arms_present],
        "eval_note": ("E2 pools out-of-fold predictions over the whole training split; "
                      "E1 measured a single 80/20 holdout and the organiser figures a "
                      "gold test set we do not have. Deltas are indicative, not "
                      "like-for-like."),
        "imbalance_condition": "none (D4 reference cell; the organiser DBO baseline "
                               "uses balanced class weights, so DBO is not comparable "
                               "on equal terms until E4a)",
        "measured": {
            "runs": _records(summary),
            "baseline_comparison": _records(comparison),
        },
        "decision": "TODO",
        "decision_note": "TODO - G2 is decided after B6 (encoder + LLM arms complete).",
        # ⚠️ ONE FIELD CANNOT NAME TWO MACHINES. Taking `runs[0]`'s env_pin
        # would claim a single Darwin environment for a result set produced on
        # two. The environment is a MEASUREMENT,
        # and a measurement that cannot represent what happened must not be
        # flattened into something that can. `env_pin` therefore holds a value
        # only while there is exactly one, and the map is always written.
        **_gate_env_pins(runs),
        "refreshed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for field in _HUMAN_FIELDS:
            value = previous.get(field)
            if isinstance(value, str) and not value.strip().startswith("TODO"):
                payload[field] = value
        if previous.get("decided_on"):
            payload["decided_on"] = previous["decided_on"]

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path, payload
