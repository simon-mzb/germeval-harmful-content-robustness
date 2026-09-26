"""
data_profile.py -- the measurement layer for the Data chapter (Section 3.1).

This module measures; `data_figures.py` draws. Nothing
here imports matplotlib, so a number can never depend on a drawing decision.

Everything the chapter asserts about the data is computed here and written to
`results/data_profile.json`, so that a claim in the text can be traced to a
field in one file. Where a measurement contradicts an organiser statement the
field records both, because the discrepancy is itself a finding.

Usage
-----
python -m src.data_profile              # measure everything, write the JSON
python -m src.data_profile --section provenance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loading import (
    SUBTASKS,
    _EDITION_DIR,
    _FILENAME_PATTERNS,
    load_split,
)

_ROOT = Path(__file__).parent.parent
OUT_PATH = _ROOT / "results" / "data_profile.json"

# Which rows cannot belong to the 2014-2016 corpus, read off the identifier.
# Not "a real Twitter snowflake id has 15-19 digits" (counting only the short ids); that premise is
# false in two ways:
#   - the corpus rows carry 15-16-digit ids (8.5e14 to 1.2e15). They are NOT Twitter status ids of
#     2014-2016, which have 18 digits; read as snowflakes they would date from November 2010. What they
#     are is not documented.
#   - the 19-digit ids ARE Twitter status ids, and every one of them decodes to April-June 2021 (DeTox /
#     BoTox era), after the collection ended. On DBO 2025 they are exactly the 574 added items the short
#     ids miss: 767 + 574 = the 1,341 the overview reports.
# So a row is marked as added material when its id is short (three or four digits, which no platform
# assigns) or a 19-digit Twitter id. Section `provenance` below tests the identification rather than
# assuming it. The JSON keeps its older key name "injected".
SNOWFLAKE_MIN = 10 ** 12          # below: short ids, no platform assigns them
TWITTER_2021_MIN = 10 ** 18       # at or above: 19-digit Twitter ids, all dated 2021

PLACEHOLDERS = ("[@PRE]", "[@POL]", "[@GRP]", "[@IND]")

EDITIONS = ("2025", "2026")
SPLITS = ("train", "test", "trial")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _texts(df: pd.DataFrame) -> pd.Series:
    return df["description"].astype(str)


def _ids(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["id"], errors="coerce")


def _is_injected(df: pd.DataFrame) -> pd.Series:
    ids = _ids(df)
    return (ids < SNOWFLAKE_MIN) | (ids >= TWITTER_2021_MIN)


def _pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 3) if whole else 0.0


def _label_series(df: pd.DataFrame) -> pd.Series | None:
    if "label" not in df.columns:
        return None
    return df["label"].astype(str)


def _length_stats(s: pd.Series) -> dict[str, Any]:
    chars = s.str.len()
    words = s.str.split().str.len()
    out = {}
    for name, v in (("char", chars), ("word", words)):
        out[name] = {
            "mean": round(float(v.mean()), 1),
            "median": float(v.median()),
            "p25": float(v.quantile(0.25)),
            "p75": float(v.quantile(0.75)),
            "p95": float(v.quantile(0.95)),
            "p99": float(v.quantile(0.99)),
            "max": int(v.max()),
        }
    return out


# ---------------------------------------------------------------------------
# 1. inventory + file identity
# ---------------------------------------------------------------------------

def inventory() -> dict[str, Any]:
    """Every split: size, whether it carries labels, and the file's sha256."""
    rows = []
    for (ed, st, sp), rel in sorted(_FILENAME_PATTERNS.items()):
        path = _EDITION_DIR[ed] / rel
        if not path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            df = load_split(st, ed, sp)
            n, labelled = len(df), "label" in df.columns
            labels = sorted(_label_series(df).unique()) if labelled else []
        except Exception:
            n, labelled, labels = -1, False, []
        rows.append({
            "subtask": st, "edition": ed, "split": sp,
            "file": rel, "bytes": path.stat().st_size,
            "sha256": digest, "sha256_short": digest[:16],
            "n": n, "labelled": labelled, "label_space": labels,
        })

    # the organisers claim ST1 and ST2 test sets are shared across editions
    identity = {}
    for st in SUBTASKS:
        pair = [r for r in rows if r["subtask"] == st and r["split"] == "test"]
        if len(pair) == 2:
            identity[st] = {
                "sha_2025": pair[0]["sha256_short"], "sha_2026": pair[1]["sha256_short"],
                "byte_identical": pair[0]["sha256"] == pair[1]["sha256"],
                "bytes_2025": pair[0]["bytes"], "bytes_2026": pair[1]["bytes"],
            }
    identity["_organiser_claim"] = (
        "The 2026 task pages state the test sets of ST1 and ST2 are identical to 2025. "
        "Measured: ST1 is byte-identical, ST2 is not."
    )
    return {"splits": rows, "test_set_identity": identity}


# ---------------------------------------------------------------------------
# 2. label distributions and imbalance
# ---------------------------------------------------------------------------

def label_distributions() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for st in SUBTASKS:
        out[st] = {}
        for ed in EDITIONS:
            df = load_split(st, ed, "train")
            lab = _label_series(df)
            if lab is None:
                continue
            counts = lab.value_counts()
            shares = (counts / len(df) * 100).round(3)
            out[st][ed] = {
                "n": int(len(df)),
                "n_classes": int(counts.size),
                "counts": {k: int(v) for k, v in counts.items()},
                "share_pct": {k: float(v) for k, v in shares.items()},
                "majority_class": str(counts.idxmax()),
                "minority_class": str(counts.idxmin()),
                "imbalance_ratio": round(float(counts.max() / counts.min()), 1),
                "minority_n": int(counts.min()),
            }
    return out


# ---------------------------------------------------------------------------
# 3. text length
# ---------------------------------------------------------------------------

# Shared bin edges for the length histograms, so the three subtasks and both
# editions are directly comparable and the figure layer does no binning of its
# own. Logarithmic, because tweet length spans three orders of magnitude.
LENGTH_BINS = np.logspace(np.log10(5), np.log10(11000), 46)


def length_profile() -> dict[str, Any]:
    out: dict[str, Any] = {"overall": {}, "by_class": {},
                           "histogram_bins": [round(float(b), 2) for b in LENGTH_BINS],
                           "histogram_counts": {}}
    for st in SUBTASKS:
        out["overall"][st], out["by_class"][st] = {}, {}
        for ed in EDITIONS:
            df = load_split(st, ed, "train")
            s = _texts(df)
            out["overall"][st][ed] = {"n": int(len(s)), **_length_stats(s)}
            counts, _ = np.histogram(s.str.len(), bins=LENGTH_BINS)
            out["histogram_counts"].setdefault(st, {})[ed] = [int(c) for c in counts]
            lab = _label_series(df)
            if lab is not None:
                out["by_class"][st][ed] = {
                    str(c): {"n": int((lab == c).sum()),
                             "char_median": float(s[lab == c].str.len().median()),
                             "word_median": float(s[lab == c].str.split().str.len().median())}
                    for c in sorted(lab.unique())
                }
    return out


# ---------------------------------------------------------------------------
# 4. provenance: the injected (non-Twitter-id) cohort
# ---------------------------------------------------------------------------

def provenance() -> dict[str, Any]:
    """
    Locate and characterise the non-Twitter-id cohort, and test whether it is
    genuinely a different population rather than an id-numbering artefact.
    """
    out: dict[str, Any] = {
        "_definition": "injected (= the marked items) = numeric id < 1e12 (short ids) or >= 1e18 (19-digit Twitter ids, all dated 2021); corpus ids have 15-16 digits",
        "by_split": {}, "per_class": {}, "text_profile": {}, "persistence": {},
    }

    for st in SUBTASKS:
        out["by_split"][st], out["per_class"][st], out["text_profile"][st] = {}, {}, {}
        for ed in EDITIONS:
            for sp in ("train", "test"):
                try:
                    df = load_split(st, ed, sp)
                except Exception:
                    continue
                inj = _is_injected(df)
                entry = {"n": int(len(df)), "injected_n": int(inj.sum()),
                         "injected_pct": _pct(int(inj.sum()), len(df))}
                if inj.any():
                    ids = _ids(df)[inj]
                    entry["id_range"] = [int(ids.min()), int(ids.max())]
                out["by_split"][st][f"{ed}_{sp}"] = entry

            df = load_split(st, ed, "train")
            lab, inj = _label_series(df), _is_injected(df)
            if lab is not None:
                g = pd.DataFrame({"label": lab, "inj": inj}).groupby("label")["inj"]
                out["per_class"][st][ed] = {
                    str(c): {
                        "n": int(g.size()[c]),
                        "injected_n": int(g.sum()[c]),
                        "injected_pct_of_class": round(float(100 * g.mean()[c]), 1),
                    } for c in g.size().index
                }
            # is the cohort stylistically distinguishable?
            s = _texts(df)
            prof = {}
            for name, mask in (("injected", inj), ("corpus", ~inj)):
                sub = s[mask]
                if len(sub) == 0:
                    continue
                prof[name] = {
                    "n": int(len(sub)),
                    "char_median": float(sub.str.len().median()),
                    "word_median": float(sub.str.split().str.len().median()),
                    "pct_url": _pct(int(sub.str.contains("http", regex=False).sum()), len(sub)),
                    "pct_placeholder": _pct(
                        int(sub.str.contains(r"\[@(?:PRE|POL|GRP|IND)\]", regex=True).sum()), len(sub)),
                }
            out["text_profile"][st][ed] = prof

    for st in SUBTASKS:
        a, b = load_split(st, "2025", "train"), load_split(st, "2026", "train")
        ia = set(_texts(a)[_is_injected(a)])
        ib = set(_texts(b)[_is_injected(b)])
        out["persistence"][st] = {
            "injected_texts_2025": len(ia), "injected_texts_2026": len(ib),
            "kept": len(ia & ib), "dropped": len(ia - ib), "added": len(ib - ia),
        }
    return out


# ---------------------------------------------------------------------------
# 5. duplicates, overlap, leakage
# ---------------------------------------------------------------------------

def _norm(s: pd.Series) -> pd.Series:
    """Case- and whitespace-insensitive form used for the 'near' duplicate count."""
    return s.str.strip().str.lower()


def duplication() -> dict[str, Any]:
    """
    Exact and normalised duplication, reported side by side.

    The distinction is not pedantry here: measured on this data, the 2026
    edition contains no exact duplicates at all while 2025 contains dozens, so
    an exact-match deduplication clearly ran between the editions. What
    survives in 2026 are case and whitespace variants ("Danke!" / "DANKE!"),
    which only the normalised count sees. Reporting one number alone would
    either hide the organisers' cleanup or overstate how clean the result is.
    """
    out: dict[str, Any] = {"within_split": [], "train_test": [], "train_trial": [],
                           "label_conflicts": []}
    for st in SUBTASKS:
        for ed in EDITIONS:
            tr = load_split(st, ed, "train")
            t, tn = _texts(tr), _norm(_texts(tr))
            out["within_split"].append({
                "subtask": st, "edition": ed, "split": "train", "n": int(len(tr)),
                "exact_duplicate_rows": int(len(t) - t.nunique()),
                "exact_duplicate_pct": _pct(len(t) - t.nunique(), len(t)),
                "normalised_duplicate_rows": int(len(tn) - tn.nunique()),
                "normalised_duplicate_pct": _pct(len(tn) - tn.nunique(), len(tn)),
                "distinct_texts": int(t.nunique()),
            })
            # a repeated text carrying two different labels is an
            # annotator-independent lower bound on label inconsistency
            lab = _label_series(tr)
            if lab is not None:
                for tag, key in ((t, "exact"), (tn, "normalised")):
                    g = pd.DataFrame({"t": tag, "l": lab}).groupby("t")["l"].nunique()
                    dup_texts = int((pd.DataFrame({"t": tag}).groupby("t").size() > 1).sum())
                    conflicted = int((g > 1).sum())
                    if key == "exact":
                        rec = {"subtask": st, "edition": ed}
                    rec[f"{key}_duplicated_texts"] = dup_texts
                    rec[f"{key}_conflicting_labels"] = conflicted
                    rec[f"{key}_conflict_pct_of_duplicated"] = _pct(conflicted, dup_texts)
                out["label_conflicts"].append(rec)
            for other in ("test", "trial"):
                try:
                    od = load_split(st, ed, other)
                except Exception:
                    continue
                o, on = _texts(od), _norm(_texts(od))
                out["train_test" if other == "test" else "train_trial"].append({
                    "subtask": st, "edition": ed, "n_other": int(len(od)),
                    "exact_overlap_n": len(set(o) & set(t)),
                    "exact_overlap_pct_of_other": _pct(len(set(o) & set(t)), len(od)),
                    "normalised_overlap_n": len(set(on) & set(tn)),
                    "normalised_overlap_pct_of_other": _pct(len(set(on) & set(tn)), len(od)),
                })
    return out


def cross_edition() -> dict[str, Any]:
    """Overlap in both directions, growth, label shift, and re-annotation."""
    out: dict[str, Any] = {"overlap": [], "growth": [], "reannotation": {}}
    for st in SUBTASKS:
        a, b = load_split(st, "2025", "train"), load_split(st, "2026", "train")
        ta, tb = set(_texts(a)), set(_texts(b))
        inter = ta & tb
        rec = {
            "subtask": st,
            "distinct_2025": len(ta), "distinct_2026": len(tb), "overlap_n": len(inter),
            "pct_of_2025_kept": _pct(len(inter), len(ta)),
            "pct_of_2026_carried": _pct(len(inter), len(tb)),
        }
        na, nb = set(_norm(_texts(a))), set(_norm(_texts(b)))
        rec["normalised_overlap_n"] = len(na & nb)
        try:
            te = load_split(st, "2025", "test")
            t_te, n_te = set(_texts(te)), set(_norm(_texts(te)))
            rec["test2025_in_train2026_n"] = len(t_te & tb)
            rec["test2025_in_train2026_pct"] = _pct(len(t_te & tb), len(t_te))
            rec["test2025_in_train2026_normalised_pct"] = _pct(len(n_te & nb), len(t_te))
        except Exception:
            pass
        out["overlap"].append(rec)

        la, lb = _label_series(a), _label_series(b)
        out["growth"].append({
            "subtask": st, "n_2025": int(len(a)), "n_2026": int(len(b)),
            "growth_pct": round(100 * (len(b) - len(a)) / len(a), 1),
            "label_space_2025": sorted(la.unique()) if la is not None else [],
            "label_space_2026": sorted(lb.unique()) if lb is not None else [],
            "label_space_changed": (sorted(la.unique()) != sorted(lb.unique()))
            if (la is not None and lb is not None) else None,
        })

        if la is None or lb is None:
            continue
        if sorted(la.unique()) != sorted(lb.unique()):
            out["reannotation"][st] = {
                "comparable": False,
                "reason": "label space changed between editions; a label change "
                          "cannot be separated from the redesign",
            }
            continue
        da = pd.DataFrame({"t": _texts(a), "l25": la}).drop_duplicates("t")
        db = pd.DataFrame({"t": _texts(b), "l26": lb}).drop_duplicates("t")
        m = da.merge(db, on="t")
        changed = m["l25"] != m["l26"]
        trans = pd.crosstab(m.loc[changed, "l25"], m.loc[changed, "l26"])
        per_class = {}
        for c in sorted(m["l25"].unique()):
            sel = m["l25"] == c
            per_class[str(c)] = {
                "shared_n": int(sel.sum()),
                "changed_n": int((changed & sel).sum()),
                "changed_pct": _pct(int((changed & sel).sum()), int(sel.sum())),
            }
        out["reannotation"][st] = {
            "comparable": True,
            "shared_texts": int(len(m)),
            "label_changed_n": int(changed.sum()),
            "label_changed_pct": _pct(int(changed.sum()), len(m)),
            "per_source_class": per_class,
            "transitions": {str(r): {str(c): int(trans.loc[r, c]) for c in trans.columns}
                            for r in trans.index},
        }
    return out


def cross_subtask() -> dict[str, Any]:
    """Do the three subtasks annotate the same tweets? (bears on Dimension 2)"""
    out: dict[str, Any] = {}
    for ed in EDITIONS:
        texts = {st: set(_texts(load_split(st, ed, "train"))) for st in SUBTASKS}
        pairs = {}
        for i, a in enumerate(SUBTASKS):
            for b in SUBTASKS[i + 1:]:
                inter = texts[a] & texts[b]
                pairs[f"{a}|{b}"] = {
                    "overlap_n": len(inter),
                    f"pct_of_{a}": _pct(len(inter), len(texts[a])),
                    f"pct_of_{b}": _pct(len(inter), len(texts[b])),
                }
        allthree = texts["c2a"] & texts["dbo"] & texts["vio"]
        union = texts["c2a"] | texts["dbo"] | texts["vio"]
        out[ed] = {
            "per_subtask_distinct": {st: len(texts[st]) for st in SUBTASKS},
            "pairwise": pairs,
            "in_all_three": len(allthree),
            "union_distinct_texts": len(union),
            "sum_of_splits": sum(len(texts[st]) for st in SUBTASKS),
            "redundancy_factor": round(sum(len(texts[st]) for st in SUBTASKS) / len(union), 2),
        }
    return out


# ---------------------------------------------------------------------------
# 5b. the DBO self-contradiction
# ---------------------------------------------------------------------------

def label_contradictions() -> dict[str, Any]:
    """
    Texts that appear twice in one training set under two different labels,
    and what the pipeline's deduplication currently does with them.

    Why this has its own section rather than a line in `duplication`: on DBO
    2025 it is not a rounding-error phenomenon. 86 of 87 contradicting texts
    sit in the injected cohort, every one of them appears exactly twice under
    two different ids, and 86 of the 87 contradictions are the same pair of
    labels (`agitation` vs `nothing`). The organisers removed the condition in
    2026, which supplies an external resolution for those texts that survive
    into the later edition, and measuring our kept label against it shows the
    deduplication is not choosing arbitrarily but wrongly.
    """
    out: dict[str, Any] = {}
    for st in SUBTASKS:
        for ed in EDITIONS:
            df = load_split(st, ed, "train")
            lab = _label_series(df)
            if lab is None:
                continue
            t = _texts(df)
            g = pd.DataFrame({"t": t, "l": lab}).groupby("t")["l"].nunique()
            conf = set(g[g > 1].index)
            rec: dict[str, Any] = {"contradicting_texts": len(conf)}
            if conf:
                sub = pd.DataFrame({"t": t, "l": lab, "inj": _is_injected(df)})
                sub = sub[sub["t"].isin(conf)]
                pairs = sub.groupby("t")["l"].apply(lambda s: " | ".join(sorted(set(s))))
                rec["rows_involved"] = int(len(sub))
                rec["in_injected_cohort_texts"] = int(sub.groupby("t")["inj"].any().sum())
                rec["label_pairs"] = {k: int(v) for k, v in pairs.value_counts().items()}
                cls = sub["l"].value_counts()
                tot = lab.value_counts()
                rec["share_of_class_affected_pct"] = {
                    str(c): _pct(int(cls.get(c, 0)), int(tot[c])) for c in tot.index
                }
                # what harness.py's dedup (normalised, keep="first") retains
                norm = _norm(t)
                keptmask = ~norm.duplicated(keep="first")
                kept = pd.DataFrame({"t": t, "l": lab})[keptmask]
                kc = kept[kept["t"].isin(conf)]
                rec["dedup_keeps"] = {str(k): int(v) for k, v in kc["l"].value_counts().items()}
                # resolve against the other edition where the label space allows it
                other = "2026" if ed == "2025" else "2025"
                try:
                    od = load_split(st, other, "train")
                    ol = _label_series(od)
                    if ol is not None and sorted(ol.unique()) == sorted(lab.unique()):
                        truth = (pd.DataFrame({"t": _texts(od), "o": ol})
                                 .drop_duplicates("t").set_index("t")["o"])
                        m = kc.assign(o=kc["t"].map(truth)).dropna(subset=["o"])
                        agree = (m["l"] == m["o"])
                        rec["resolution_against_other_edition"] = {
                            "edition_used": other,
                            "resolvable_texts": int(len(m)),
                            "kept_label_agrees": int(agree.sum()),
                            "kept_label_disagrees": int((~agree).sum()),
                            "agreement_pct": _pct(int(agree.sum()), len(m)),
                        }
                except Exception:
                    pass
            out[f"{st}_{ed}"] = rec
    out["_consequence"] = (
        "harness.py deduplicates with keep='first' on the normalised text. For the "
        "DBO 2025 contradictions that rule keeps the label the organisers later "
        "revised in every resolvable case, so 49 items enter training as 'nothing' "
        "that the 2026 edition calls 'agitation'. Resolving them from the 2026 "
        "labels would import later-edition information into a 2025 training set and "
        "is therefore not available to Dimension 3. The decision is recorded in "
        "THESIS_LOG Next Steps, not taken here."
    )
    return out


# ---------------------------------------------------------------------------
# 6. surface features
# ---------------------------------------------------------------------------

def surface_features() -> dict[str, Any]:
    pats = {
        "url": r"https?://",
        "user_mention": r"(?<!\[)@\w",
        "hashtag": r"#\w",
        "anonymisation_placeholder": r"\[@(?:PRE|POL|GRP|IND)\]",
        "digit_run": r"\d{3,}",
        "newline": r"\n",
        "uppercase_run": r"[A-ZÄÖÜ]{4,}",
    }
    out: dict[str, Any] = {}
    for st in SUBTASKS:
        out[st] = {}
        for ed in EDITIONS:
            s = _texts(load_split(st, ed, "train"))
            out[st][ed] = {"n": int(len(s))}
            for name, p in pats.items():
                out[st][ed][name + "_pct"] = _pct(int(s.str.contains(p, regex=True).sum()), len(s))
            out[st][ed]["placeholder_breakdown_pct"] = {
                ph: _pct(int(s.str.contains(re.escape(ph), regex=True).sum()), len(s))
                for ph in PLACEHOLDERS
            }
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

SECTIONS = {
    "inventory": inventory,
    "label_distributions": label_distributions,
    "length_profile": length_profile,
    "provenance": provenance,
    "duplication": duplication,
    "label_contradictions": label_contradictions,
    "cross_edition": cross_edition,
    "cross_subtask": cross_subtask,
    "surface_features": surface_features,
}


def build(sections: list[str] | None = None) -> dict[str, Any]:
    todo = sections or list(SECTIONS)
    out: dict[str, Any] = {
        "_schema": "data_profile/v1",
        "_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_note": "Measurement layer for Section 3.1. Figures live in src/data_figures.py.",
    }
    for name in todo:
        print(f"  [{name}] ...", flush=True)
        out[name] = SECTIONS[name]()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the GermEval data for Section 3.1.")
    ap.add_argument("--section", action="append", choices=list(SECTIONS),
                    help="run only these sections (repeatable)")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    print("Data profile")
    prof = build(args.section)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.section and args.out.exists():
        existing = json.loads(args.out.read_text())
        existing.update(prof)
        prof = existing
    args.out.write_text(json.dumps(prof, indent=2, ensure_ascii=False, default=str))
    print(f"\n  wrote {args.out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
