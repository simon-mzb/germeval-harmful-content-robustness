"""
e5_label_sample.py -- draw the manual inspection sample for Section 4.5, blind.

WHAT IT DOES. Draws two class-matched strata of 72 items each from the 2025 DBO
pool and writes them as two outwardly identical coding files, so that a single
human coder can assign a class to every item without knowing which stratum a
file holds, which item is a repeat, or what the gold label is.

  stratum IRR  the 278 irreducible items -- all six components wrong
  stratum CTL  items where at least one but not all six components are wrong

WHY CTL IS NOT "ALL SIX RIGHT". That set cannot be
drawn class-matched at all: it holds 0 `subversive` and 4 `agitation` items
against the 7 and 19 the allocation needs, and 4,420 of its 4,492 items are
`nothing`. A draw from it would compare the easy majority class against a
minority-heavy error set, and the difference would be large by construction and
silent about label quality. CTL as defined holds difficulty far closer, so the
contrast is consensus errors against ordinary errors, which is the question
Section 4.5 asks. It does NOT hold difficulty equal, which is why the k
distribution of the drawn control group is written into the artefact: if the
control group sits mostly at k = 1, the difference still carries some difficulty, and
that is reported rather than re-drawn.

PRE-REGISTERED BEFORE THE DRAW (thesis Appendix B). Seeds, strata, allocation,
the repeat count, the analysis and the two power figures were fixed before this
module was run.

Usage
-----
python -m src.e5_label_sample --help
python -m src.e5_label_sample            # draws; refuses to overwrite
"""

from __future__ import annotations

import csv
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src import e3_combination as e3
from src.e5_irreducible import COMPONENTS
from src.harness import measuring_main_guard

SUBTASK = "dbo"
EDITION = "2025"
N_PER_STRATUM = 72
N_REPEATS = 20
MIN_PAIR_DISTANCE = 25
SEED_DRAW, SEED_ORDER, SEED_REPEATS, SEED_ASSIGN = 42, 43, 44, 45

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "e5_sample"
OUT = OUT_DIR / "dbo_sample_key.json"
NEWLINE_MARK = " ⏎ "


def allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation with largest-remainder rounding, summing to n."""
    total = sum(counts.values())
    raw = {c: v * n / total for c, v in counts.items()}
    out = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: (-(raw[c] - out[c]), c))[:n - sum(out.values())]:
        out[c] += 1
    assert sum(out.values()) == n, out
    return out


def k_wrong_per_item(recs: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    base = recs[COMPONENTS[0]]
    classes = [str(c) for c in base["classes"]]
    y = np.array([classes.index(str(v)) for v in base["y_true"]])
    ids = np.asarray([str(i) for i in base["ids"]])
    pred = np.stack([np.asarray(recs[c]["proba"]) for c in COMPONENTS]).argmax(2)
    return (pred != y).sum(0), y, ids, classes


def draw_stratum(rng, mask, y, ids, classes, alloc) -> list[dict[str, Any]]:
    out = []
    for cls in sorted(alloc):
        i = classes.index(cls)
        pool = np.flatnonzero(mask & (y == i))
        if len(pool) < alloc[cls]:
            raise SystemExit("stratum has {} {} items, allocation needs {}".format(
                len(pool), cls, alloc[cls]))
        for j in rng.choice(pool, size=alloc[cls], replace=False):
            out.append({"index": int(j), "id": ids[j], "gold": cls})
    return out


def place_repeats(rng, uniques: list[dict], repeat_idx) -> tuple[list[dict], int]:
    """Insert each repeated item a second time, at least MIN_PAIR_DISTANCE rows away.

    Rejection sampling does not work here: a uniform permutation of 92 rows keeps
    all 20 pairs 25 apart with vanishing probability, and 1000 attempts found no
    ordering at all. So the placement is CONSTRUCTIVE. Each duplicate is inserted
    at a position drawn uniformly from the two admissible regions -- at least 25
    before or at least 25 after its twin -- and an insertion never shortens a gap
    that already holds: inserting between two occurrences widens theirs, and
    inserting outside them leaves it unchanged. The result therefore satisfies the
    constraint by construction, with no retries and no bias towards the orderings
    a rejection loop would have happened to accept. Drawing from BOTH regions
    matters: allowing only "later" would mean an item in the last rows could never
    be the first of a pair, which is a pattern a coder could notice.
    """
    final = list(uniques)
    for k in rng.permutation(repeat_idx):
        twin = uniques[int(k)]
        i = next(pos for pos, r in enumerate(final) if r["id"] == twin["id"])
        allowed = list(range(0, max(i - MIN_PAIR_DISTANCE + 1, 0)))
        allowed += list(range(i + MIN_PAIR_DISTANCE, len(final) + 1))
        if not allowed:
            raise SystemExit("no admissible position for the repeat of {}".format(twin["id"]))
        final.insert(int(rng.choice(allowed)), dict(twin))
    return final, 0


def build_file(rng_repeat, rng_order, drawn: list[dict[str, Any]]) -> tuple[list[dict], int]:
    uniques = [dict(drawn[int(k)]) for k in rng_order.permutation(len(drawn))]
    repeat_idx = rng_repeat.choice(len(uniques), size=N_REPEATS, replace=False)
    ordered, retries = place_repeats(rng_order, uniques, repeat_idx)
    seen: dict[str, int] = {}
    for pos, r in enumerate(ordered, start=1):
        r["row"] = "r{:03d}".format(pos)
        r["occurrence"] = 2 if r["id"] in seen else 1
        seen.setdefault(r["id"], pos)
    # the constraint, asserted rather than trusted
    first: dict[str, int] = {}
    for pos, r in enumerate(ordered):
        if r["id"] in first:
            assert pos - first[r["id"]] >= MIN_PAIR_DISTANCE, (r["id"], pos, first[r["id"]])
        first[r["id"]] = pos
    return ordered, retries


GUIDELINE = (Path(__file__).resolve().parents[1] / "data" / "codabench"
             / "elaborated_info_2025_tasks.txt")
LABELS = ("Subversive", "Agitation", "Criticism", "Nothing")


def guideline_text() -> tuple[list[str], list[str]]:
    """The organisers' own class definitions and examples, quoted verbatim.

    Taken from the shared task's material and never from this thesis's prose: the
    coder is asked to apply the scheme the original annotators applied, and a
    paraphrase of it in Chapter 3 is a different yardstick. Extracted by the
    label names rather than by line number so the quotation cannot drift.
    """
    lines = GUIDELINE.read_text(encoding="utf-8").splitlines()
    # Scope to this subtask's own section: the file documents all three 2025
    # subtasks, and "Nothing (" introduces a class in two of them.
    starts = [i for i, ln in enumerate(lines) if ln.startswith("# Subtask ")]
    here = next(i for i in starts if "Democratic Basic Order" in lines[i])
    after = next((i for i in starts if i > here), len(lines))
    lines = lines[here:after]
    defs, examples = [], []
    for label in LABELS:
        hit = [ln.strip() for ln in lines if ln.startswith(label + " (")]
        if len(hit) != 1:
            raise SystemExit("the 2025 guideline no longer has exactly one {} definition "
                             "({} found)".format(label, len(hit)))
        defs.append(hit[0])
    for label in LABELS:
        hit = [ln for ln in lines if ln.startswith(label.lower() + "\t")]
        if len(hit) != 1:
            raise SystemExit("the 2025 guideline no longer has exactly one {} example "
                             "({} found)".format(label.lower(), len(hit)))
        examples.append(hit[0].split("\t", 1)[1].strip())
    return defs, examples


INSTRUCTIONS = """# Coding sheet -- attacks on the democratic basic order (DBO), 2025

You are assigning ONE of four classes to each text, the same four the original
annotators used. This is not a judgement of anyone's label: you do not see the
stored label, and you are not asked whether it is right. Assign the class you
would have assigned.

## How to work

1. Open `{file}` in a spreadsheet (Numbers, Excel, LibreOffice). Every row is one
   text and the `label` column is empty for you to fill.
2. Enter exactly one of: `subversive`, `agitation`, `criticism`, `nothing`.
   If you genuinely cannot decide, enter `unclear` -- it is counted as its own
   share and is NOT quietly dropped.
3. Work top to bottom and do not skip a row. Nothing may be excluded afterwards.
4. A line break inside a text is shown as `{mark}` so that every record stays on
   one line. `[@PRE]`, `[@POL]`, `[@GRP]` and `[@IND]` are the corpus's
   anonymisation placeholders for press, police, groups and individuals.
5. Some texts appear twice. That is deliberate and measures how stable the coding
   is; code the second occurrence as you find it, without looking back.

Expect about {minutes} minutes. Save the file under the same name when you are done.

## The four classes, in the shared task's own words

{definitions}

## The shared task's own example for each class

{examples}

## What is measured afterwards

How often your class matches the stored one, with a confidence interval; which
classes are confused for which, descriptively; and how often the repeated texts
got the same class twice. Nothing else. The comparison the two files allow is
fixed in THESIS_LOG NS 124 (z14) and (z16), written before this draw.
"""


def write_instructions(path: Path, file_name: str, minutes: int) -> None:
    defs, examples = guideline_text()
    path.write_text(INSTRUCTIONS.format(
        file=file_name, mark=NEWLINE_MARK.strip(), minutes=minutes,
        definitions="\n\n".join("- **{}**".format(d) for d in defs),
        examples="\n\n".join("- *{}* -- {}".format(lbl.lower(), ex)
                              for lbl, ex in zip(LABELS, examples))),
        encoding="utf-8")


def main(argv=None) -> int:
    args = measuring_main_guard(__doc__, OUT, argv)
    if args is None:
        return 0
    from src.component_store import ComponentStore
    from src.data_loading import load_split

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        recs = e3.load_subtask(SUBTASK, COMPONENTS, ComponentStore())
    k_wrong, y, ids, classes = k_wrong_per_item(recs)

    irr = k_wrong == len(COMPONENTS)
    ctl = (k_wrong >= 1) & (k_wrong < len(COMPONENTS))
    counts = {c: int(((y == classes.index(c)) & irr).sum()) for c in classes}
    alloc = allocate(counts, N_PER_STRATUM)

    rng = np.random.default_rng(SEED_DRAW)
    strata = {"IRR": draw_stratum(rng, irr, y, ids, classes, alloc),
              "CTL": draw_stratum(rng, ctl, y, ids, classes, alloc)}

    text_by_id = {}
    split = load_split(SUBTASK, EDITION, "train")
    for i, t in zip(split["id"].astype(str), split["description"].astype(str)):
        text_by_id[i] = t

    rng_assign = np.random.default_rng(SEED_ASSIGN)
    names = ["IRR", "CTL"] if rng_assign.random() < 0.5 else ["CTL", "IRR"]
    assignment = {"set_a": names[0], "set_b": names[1]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key: dict[str, Any] = {}
    # PER-FILE STREAMS, not one shared seed. With a single `default_rng(SEED_ORDER)`
    # per file both files inherit the same permutation of 72 indices and the same
    # 20 repeat slots, so their repeat positions coincide row for row -- checked
    # after the first draw and found identical (r036, r046, r056, ...). Two files
    # that are supposed to be indistinguishable must not share a structure, so
    # each takes its own stream. Nothing had been coded when this was corrected.
    for idx, (letter, stratum) in enumerate(
            (("a", assignment["set_a"]), ("b", assignment["set_b"]))):
        rows, retries = build_file(np.random.default_rng([SEED_REPEATS, idx]),
                                   np.random.default_rng([SEED_ORDER, idx]), strata[stratum])
        first = {}
        for r in rows:
            first.setdefault(r["id"], r["row"])
        path = OUT_DIR / "dbo_sample_set_{}.csv".format(letter)
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, quoting=csv.QUOTE_ALL)
            w.writerow(["row", "label", "text"])
            for r in rows:
                text = text_by_id[r["id"]].replace("\r\n", "\n").replace("\n", NEWLINE_MARK)
                w.writerow([r["row"], "", text])
        key["set_{}".format(letter)] = {
            "file": path.name,
            "stratum": stratum,
            "order_retries": retries,
            "rows": [{"row": r["row"], "id": r["id"], "gold": r["gold"],
                      "k_wrong": int(k_wrong[r["index"]]),
                      "occurrence": r["occurrence"],
                      "pair_of": first[r["id"]] if r["occurrence"] == 2 else None}
                     for r in rows],
        }

    for letter in ("a", "b"):
        write_instructions(OUT_DIR / "CODING_INSTRUCTIONS_set_{}.md".format(letter),
                           "dbo_sample_set_{}.csv".format(letter), 140)

    ctl_rows = key["set_a" if assignment["set_a"] == "CTL" else "set_b"]["rows"]
    ctl_k = {}
    for r in ctl_rows:
        if r["occurrence"] == 1:
            ctl_k[str(r["k_wrong"])] = ctl_k.get(str(r["k_wrong"]), 0) + 1

    payload = {
        "experiment": "E5 manual label inspection sample (Section 4.5)",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": e3._git_commit(),
        "pre_registration": "THESIS_LOG NS 124 (z14), control stratum and power figures (z16)",
        "subtask": SUBTASK, "edition": EDITION,
        "strata": {
            "IRR": "all six components wrong (the irreducible set)",
            "CTL": "at least one but not all six components wrong",
        },
        "population_sizes": {"IRR": int(irr.sum()), "CTL": int(ctl.sum()),
                             "all_six_right": int((k_wrong == 0).sum())},
        "allocation_per_class": alloc,
        "n_per_stratum": N_PER_STRATUM, "n_repeats_per_file": N_REPEATS,
        "min_pair_distance": MIN_PAIR_DISTANCE,
        "seeds": {"draw": SEED_DRAW, "order": SEED_ORDER,
                  "repeats": SEED_REPEATS, "assign": SEED_ASSIGN},
        "assignment": assignment,
        "control_k_wrong_distribution": dict(sorted(ctl_k.items())),
        "newline_replacement": NEWLINE_MARK,
        "ids_sha256": {s: hashlib.sha256(
            "\n".join(r["id"] for r in sorted(strata[s], key=lambda r: r["id"])).encode()
        ).hexdigest() for s in strata},
        "analysis_fixed_before_coding": [
            "agreement with gold per file, 95 % Wilson interval",
            "the complement as the contested share; `unclear` stays in the denominator",
            "descriptive confusion of assigned against gold class, no per-class claim",
            "stability: share of the 20 repeat pairs coded identically, with its interval",
            "if both files are coded: the difference of the two contested shares with a 95 % "
            "interval; the interval half-width is at most 16.3 pp and a difference of about "
            "22 pp is what 80 % power detects -- these are different quantities",
        ],
        "key": key,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote {} and the two coding files".format(OUT))
    print("assignment (sidecar only): {}".format(assignment))
    print("control k_wrong distribution: {}".format(payload["control_k_wrong_distribution"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
