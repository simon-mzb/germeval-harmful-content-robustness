"""
verify_data_defects.py -- independent re-measurement of two defects in the
GermEval data that the Data chapter reports.

Both defects were first seen through `src/data_profile.py`, which uses pandas
and the project's own loader. A claim about someone else's dataset should not
rest on our own toolchain, so this script re-measures them with the Python
standard library alone: it opens the CSV files directly, parses them with the
`csv` module, and shares no code path with `data_loading.py`. If the two
disagree, the finding is ours, not the data's.

The defects:

  1. Self-contradicting labels. In dbo_train.csv (2025), 87 texts occur twice
     under two different labels. The pipeline deduplicates with keep="first",
     so one of the two labels silently wins.

  2. A second source inside the corpus. Part of the ST2 and ST3 data does not
     come from the Twitter corpus the task documentation describes. Two
     independent markers identify it: an id that is too short to be a Twitter
     snowflake id, and the anonymisation convention "@user" instead of the
     "[@PRE]/[@POL]/[@GRP]/[@IND]" scheme the readmes specify. The script
     checks that the two markers agree, which is what makes either usable.

Usage
-----
python -m src.verify_data_defects
python -m src.verify_data_defects --json results/data_defects_verification.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
_DATA = _ROOT / "data" / "codabench"
OUT_PATH = _ROOT / "results" / "data_defects_verification.json"

# Added material, read off the identifier: short ids (no platform assigns them)
# and 19-digit Twitter ids (all dated 2021, after the 2014-2016 collection). The corpus carries 15-16-digit
# ids. Kept as its own constant on purpose: this script shares no code path with data_profile.py.
SNOWFLAKE_MIN = 10 ** 12
TWITTER_2021_MIN = 10 ** 18

CORPUS_PLACEHOLDERS = ("[@PRE]", "[@POL]", "[@GRP]", "[@IND]")
FOREIGN_MENTION = "@user"

FILES = {
    ("c2a", "2025", "train"): "GermEval2025/data/c2a/c2a_train.csv",
    ("c2a", "2025", "test"):  "GermEval2025/data/c2a/c2a_test.csv",
    ("dbo", "2025", "train"): "GermEval2025/data/dbo/dbo_train.csv",
    ("dbo", "2025", "test"):  "GermEval2025/data/dbo/dbo_test.csv",
    ("vio", "2025", "train"): "GermEval2025/data/vio/vio_train.csv",
    ("vio", "2025", "test"):  "GermEval2025/data/vio/vio_test.csv",
    ("c2a", "2026", "train"): "GermEval2026/data/c2a/c2a_train_26.csv",
    ("dbo", "2026", "train"): "GermEval2026/data/dbo/dbo_train_26.csv",
    ("vio", "2026", "train"): "GermEval2026/data/vio/vio_train_26.csv",
}

# Totals stated in the 2025 subtask readmes, for reconciliation against the files.
STATED_TOTALS_2025 = {"c2a": 9822, "dbo": 9307, "vio": 10933}


def read_rows(rel: str) -> list[list[str]]:
    """Parse one CSV with the standard library. Returns data rows without the header."""
    text = (_DATA / rel).read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text), delimiter=";", quotechar='"'))
    return rows[1:]


def is_short_id(raw_id: str) -> bool:
    try:
        return int(raw_id) < SNOWFLAKE_MIN or int(raw_id) >= TWITTER_2021_MIN
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# defect 1: the same text under two labels
# ---------------------------------------------------------------------------

def check_contradictions(subtask: str, edition: str) -> dict[str, Any]:
    rows = read_rows(FILES[(subtask, edition, "train")])
    malformed = sum(1 for r in rows if len(r) != 3)

    labels_by_text: dict[str, set[str]] = collections.defaultdict(set)
    rows_by_text: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for row_id, text, label in ((r[0], r[1], r[2]) for r in rows if len(r) == 3):
        labels_by_text[text].add(label)
        rows_by_text[text].append((row_id, label))

    conflicting = {t: v for t, v in labels_by_text.items() if len(v) > 1}
    pair_counts = collections.Counter(" | ".join(sorted(v)) for v in conflicting.values())
    repeat_counts = collections.Counter(len(rows_by_text[t]) for t in conflicting)
    short_id_rows = sum(1 for t in conflicting for rid, _ in rows_by_text[t] if is_short_id(rid))

    # how much of each class is caught in a contradicting pair
    class_totals = collections.Counter(r[2] for r in rows if len(r) == 3)
    affected = collections.Counter(
        lbl for t in conflicting for _, lbl in rows_by_text[t]
    )

    return {
        "rows_parsed": len(rows),
        "malformed_rows": malformed,
        "rows_with_embedded_newline": sum(1 for r in rows if len(r) == 3 and "\n" in r[1]),
        "distinct_ids": len({r[0] for r in rows if len(r) == 3}),
        "distinct_texts": len(labels_by_text),
        "contradicting_texts": len(conflicting),
        "label_pairs": dict(pair_counts),
        "occurrences_per_contradicting_text": dict(repeat_counts),
        "contradicting_rows_with_short_id": short_id_rows,
        "contradicting_rows_total": 2 * len(conflicting),
        "share_of_class_in_a_contradiction_pct": {
            lbl: round(100 * affected.get(lbl, 0) / n, 2) for lbl, n in class_totals.items()
        },
        "examples": [
            {"text": t, "rows": rows_by_text[t]}
            for t in list(conflicting)[:3]
        ],
    }


# ---------------------------------------------------------------------------
# defect 2: two source datasets, two anonymisation conventions
# ---------------------------------------------------------------------------

def check_provenance_markers() -> dict[str, Any]:
    """
    Test whether the id marker and the anonymisation marker pick out the same
    rows. Neither is trustworthy alone; agreement is the evidence.
    """
    per_split = {}
    disagreements = 0
    for (subtask, edition, split), rel in FILES.items():
        rows = [r for r in read_rows(rel) if len(r) >= 2]
        short = [r for r in rows if is_short_id(r[0])]
        foreign = [r for r in rows if FOREIGN_MENTION in r[1].lower()]
        corpus_scheme = [r for r in rows
                         if any(p in r[1] for p in CORPUS_PLACEHOLDERS)]
        # the case that would break the identification: foreign convention on a
        # regular Twitter id
        foreign_with_long_id = [r for r in foreign if not is_short_id(r[0])]
        short_with_corpus_scheme = [r for r in short
                                    if any(p in r[1] for p in CORPUS_PLACEHOLDERS)]
        disagreements += len(foreign_with_long_id) + len(short_with_corpus_scheme)
        per_split[f"{subtask}_{edition}_{split}"] = {
            "n": len(rows),
            "short_id_rows": len(short),
            "foreign_mention_rows": len(foreign),
            "corpus_placeholder_rows": len(corpus_scheme),
            "foreign_mention_with_long_id": len(foreign_with_long_id),
            "short_id_with_corpus_placeholder": len(short_with_corpus_scheme),
        }
    return {
        "per_split": per_split,
        "total_marker_disagreements": disagreements,
        "markers_agree": disagreements == 0,
    }


# ---------------------------------------------------------------------------
# context: what the organisers' own baseline does, and whether totals reconcile
# ---------------------------------------------------------------------------

def check_organiser_baseline() -> dict[str, Any]:
    """The 2025 DBO baseline notebook, checked for a deduplication step."""
    nb_path = _DATA / "GermEval2025/baseline/dbo/dbo_baseline.ipynb"
    if not nb_path.exists():
        return {"available": False}
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    return {
        "available": True,
        "mentions_drop_duplicates": "drop_duplicates" in source,
        "mentions_duplicated": "duplicated" in source,
        "drops_id_column": "drop('id'" in source or 'drop("id"' in source,
    }


def reconcile_totals() -> dict[str, Any]:
    out = {}
    for subtask, stated in STATED_TOTALS_2025.items():
        train = len(read_rows(FILES[(subtask, "2025", "train")]))
        test = len(read_rows(FILES[(subtask, "2025", "test")]))
        injected = sum(
            1 for split in ("train", "test")
            for r in read_rows(FILES[(subtask, "2025", split)])
            if r and is_short_id(r[0])
        )
        diff = train + test - stated
        out[subtask] = {
            "stated_total": stated, "train": train, "test": test,
            "files_total": train + test, "difference": diff,
            "injected_rows": injected,
            "difference_explained_by_injected_cohort": diff == injected,
        }
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    result: dict[str, Any] = {
        "_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_method": "standard library only; no pandas and no project data loader",
        "contradictions": {},
    }
    for subtask in ("c2a", "dbo", "vio"):
        for edition in ("2025", "2026"):
            if (subtask, edition, "train") in FILES:
                result["contradictions"][f"{subtask}_{edition}"] = \
                    check_contradictions(subtask, edition)
    result["provenance_markers"] = check_provenance_markers()
    result["organiser_baseline"] = check_organiser_baseline()
    result["stated_total_reconciliation"] = reconcile_totals()

    dbo = result["contradictions"]["dbo_2025"]
    print("Verification of the reported data defects\n")
    print("  parsing")
    print(f"    dbo 2025 train rows parsed        {dbo['rows_parsed']:,}")
    print(f"    malformed rows                    {dbo['malformed_rows']}")
    print(f"    rows with an embedded newline     {dbo['rows_with_embedded_newline']}")
    print("\n  defect 1: contradicting labels")
    print(f"    contradicting texts               {dbo['contradicting_texts']}")
    print(f"    label pairs                       {dbo['label_pairs']}")
    print(f"    occurrences per such text         {dbo['occurrences_per_contradicting_text']}")
    print(f"    of which rows with a short id     {dbo['contradicting_rows_with_short_id']}"
          f" of {dbo['contradicting_rows_total']}")
    print(f"    share of each class affected      {dbo['share_of_class_in_a_contradiction_pct']}")
    print("\n  defect 2: two provenance markers")
    pm = result["provenance_markers"]
    print(f"    marker disagreements (all splits) {pm['total_marker_disagreements']}")
    print(f"    markers agree                     {pm['markers_agree']}")
    print("\n  context")
    print(f"    organiser baseline deduplicates   "
          f"{result['organiser_baseline'].get('mentions_drop_duplicates')}")
    print(f"    organiser baseline drops id       "
          f"{result['organiser_baseline'].get('drops_id_column')}")
    for st, r in result["stated_total_reconciliation"].items():
        print(f"    {st} stated {r['stated_total']:,} vs files {r['files_total']:,}"
              f"  diff {r['difference']:+,}"
              f"  {'= injected cohort' if r['difference_explained_by_injected_cohort'] else '(unexplained)'}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  wrote {args.json.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
