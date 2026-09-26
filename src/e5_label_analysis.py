"""
e5_label_analysis.py -- score the manual coding of the Section 4.5 sample, exactly as pre-registered.

WHAT IT DOES. Reads the two coded files `dbo_sample_set_a.csv` / `dbo_sample_set_b.csv`
(by explicit name, never by glob), joins them to the sidecar `dbo_sample_key.json`, and
computes what the pre-registration (thesis Appendix B) fixed BEFORE the coding,
and nothing else:

  * agreement with gold over the 72 distinct items of each file, 95 % Wilson interval
    (the uncorrected form: the finite-population form has not been checked at an
    original, so the wider interval is reported, as the pre-registration says);
  * its complement as the contested share; `unclear` stays IN the denominator and counts
    as non-agreement;
  * a descriptive confusion of the assigned class against the gold class, no per-class claim;
  * stability: the share of the 20 repeat pairs of each file coded identically, with its
    Wilson interval -- the weaker quantity, NOT inter-rater reliability (McDonald,
    Schoenebeck & Forte 2019);
  * the difference of the two contested shares (IRR minus CTL) with a 95 % interval, and
    BESIDE it, as a separate quantity, the difference that 80 % power detects at 72 per
    group. Half-width and detectable difference are two different figures and are never
    reported as one.

No significance test, no per-class statement. `control_k_wrong_distribution` is copied
from the sidecar because it is a limitation the text must carry.

DECISIONS TAKEN HERE (each with its counterfactual, so it can be checked):
  1. AGREEMENT IS SCORED ON THE FIRST OCCURRENCE of each of the 72 items. Counterfactual:
     scoring the second occurrence of the 20 repeated items would mix in the coder's
     memory of his own first answer, which is the recognition effect the pre-registration
     warns about. The second occurrence enters only the stability figure.
  2. THE DIFFERENCE INTERVAL IS NEWCOMBE'S HYBRID-SCORE INTERVAL (method 10, built from the
     two Wilson intervals). The pre-registered 16.3 pp is the half-width of the plain
     Wald interval at p = 0.5 and 72 per group (1.96 * sqrt(2 * 0.25 / 72)); Wald is known
     to be too narrow at small n and near the boundaries, and the per-file intervals here
     are Wilson, so the difference uses the interval built from them. The Wald interval is
     stored beside it so the 16.3 remains checkable, and the actual half-width of each is
     recorded. Counterfactual: reporting the Wald interval would match the design figure
     digit for digit and be the less trustworthy of the two.

INPUT FORMATS. The coded files come out of Apple Numbers: a title line with
the table name BEFORE the header, `;` as the separator, empty padding columns after
`text`, CRLF line ends, no quotes. The files as drawn were comma-separated and quoted. The
reader recognises both by their header and REFUSES anything else; it does not guess. A
text may itself contain `;`, so the text is everything after the second `;` minus exactly
as many trailing `;` as the header has padding columns. The coder's originals are the raw
evidence and are never overwritten: a normalised copy (comma, quoted, LF) is written
beside them, and only after key and label survived the round trip unchanged.

GUARDS. Refuses if the coding directory holds any `dbo_sample_set_*.csv` other than the two
named files and their `.coded_backup.csv` copies, because a glob such as
`dbo_sample_set_*.csv` would read a backup as a third file and double every count. A backup
that differs from its original is reported.

Usage
-----
python -m src.e5_label_analysis --help
python -m src.e5_label_analysis          # writes results/e5_label_analysis.json; refuses to overwrite
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.e5_irreducible import wilson
from src.harness import measuring_main_guard

RESULTS = Path(__file__).resolve().parents[1] / "results"
SAMPLE_DIR = RESULTS / "e5_sample"
KEY = SAMPLE_DIR / "dbo_sample_key.json"
OUT = RESULTS / "e5_label_analysis.json"
FILES = {"set_a": "dbo_sample_set_a.csv", "set_b": "dbo_sample_set_b.csv"}
CLASSES = ["agitation", "criticism", "nothing", "subversive"]
LABELS = CLASSES + ["unclear"]
NEWLINE_MARK = " ⏎ "
Z = 1.96
Z_POWER = 0.8416212335729143  # Phi^-1(0.80)
N_PAIRS = 20
N_ITEMS = 72


class CodingFileError(Exception):
    """A coded file that cannot be read without guessing."""


def coding_files_present(directory: Path = SAMPLE_DIR) -> list[str]:
    """Every dbo_sample_set_*.csv, and refuse a stray one (a glob would double the counts)."""
    found = sorted(p.name for p in directory.glob("dbo_sample_set_*.csv"))
    allowed = set(FILES.values()) | {n.replace(".csv", ".coded_backup.csv") for n in FILES.values()}
    stray = [n for n in found if n not in allowed]
    if stray:
        raise CodingFileError(
            "more coding files than the two named ones: {} -- a glob would read them as "
            "additional data. Move them out or rename them.".format(stray))
    return found


def read_coded(path: Path) -> tuple[list[dict[str, str]], str]:
    """Rows of {row, label, text} from either format, and the name of the format found."""
    raw = path.read_bytes().decode("utf-8")
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    # Format 1: as drawn -- comma, every field quoted, header first.
    if lines and lines[0] == '"row","label","text"':
        rows = list(csv.DictReader(raw.replace("\r\n", "\n").splitlines(keepends=True)))
        return [{"row": r["row"], "label": r["label"], "text": r["text"]} for r in rows], "as-drawn"
    # Format 2: Numbers export -- a title line, then a `;` header with padding columns.
    head = next((i for i, l in enumerate(lines) if l.startswith("row;label;text")), None)
    if head is None or head > 1:
        raise CodingFileError(
            "{}: neither the as-drawn header nor a Numbers-export header (`row;label;text`) "
            "was found in the first two lines; refusing to guess a format".format(path.name))
    header = lines[head]
    pad = header.count(";") - 2
    if header != "row;label;text" + ";" * pad:
        raise CodingFileError("{}: the header carries something besides padding after `text`".format(path.name))
    rows = []
    for n, l in enumerate(lines[head + 1:], start=head + 2):
        parts = l.split(";", 2)
        if len(parts) != 3:
            raise CodingFileError("{}: line {} has no label/text separator".format(path.name, n))
        tail = parts[2]
        if pad and (len(tail) < pad or set(tail[-pad:]) != {";"}):
            raise CodingFileError("{}: line {} does not end in the {} padding columns".format(path.name, n, pad))
        text = tail[:len(tail) - pad]
        # Numbers quotes a field that contains a `"` and doubles the inner quotes
        # (ordinary CSV quoting). Found by comparing every text against the corpus: 20 of
        # the 184 rows carried the wrapping quotes and the doubling.
        if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
            text = text[1:-1].replace('""', '"')
        rows.append({"row": parts[0], "label": parts[1], "text": text})
    return rows, "numbers-export(pad={})".format(pad)


def check_rows(name: str, rows: list[dict[str, str]], key_rows: list[dict]) -> None:
    """Keys r001.. in order and identical to the sidecar; every label in the allowed set."""
    want = [k["row"] for k in key_rows]
    got = [r["row"] for r in rows]
    if got != want:
        raise CodingFileError("{}: row keys differ from the sidecar (first difference at position {})".format(
            name, next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))))
    bad = [(r["row"], r["label"]) for r in rows if r["label"] not in LABELS]
    if bad:
        raise CodingFileError("{}: labels outside {}: {}".format(name, LABELS, bad[:5]))


def write_normalised(name: str, rows: list[dict[str, str]], key_rows: list[dict]) -> Path:
    """Comma, quoted, LF -- beside the original, never over it -- then re-read and compare."""
    out = SAMPLE_DIR / "normalised_{}".format(FILES[name])
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writerow(["row", "label", "text"])
        for r in rows:
            w.writerow([r["row"], r["label"], r["text"]])
    back, fmt = read_coded(out)
    assert fmt == "as-drawn"
    check_rows(name + " (normalised)", back, key_rows)
    if [(r["row"], r["label"], r["text"]) for r in back] != [(r["row"], r["label"], r["text"]) for r in rows]:
        raise CodingFileError("{}: the normalised copy does not round-trip".format(name))
    return out


def newcombe(k1: int, n1: int, k2: int, n2: int) -> list[float]:
    """Newcombe (1998) hybrid-score interval for p1 - p2, built from the two Wilson intervals."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return [d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]


def wald_diff(k1: int, n1: int, k2: int, n2: int) -> list[float]:
    p1, p2 = k1 / n1, k2 / n2
    h = Z * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return [p1 - p2 - h, p1 - p2 + h]


def detectable_difference(p_control: float, n: int, z_alpha: float = Z, z_power: float = Z_POWER) -> float:
    """Smallest difference with 80 % power at alpha 0.05 two-sided, n per group.

    The standard two-proportion form, variance pooled under the null and unpooled under the
    alternative: n = (z_a * sqrt(2 pbar qbar) + z_b * sqrt(p1 q1 + p2 q2))^2 / delta^2, solved for delta
    by bisection with p1 = p2 + delta. The pre-registration (z16) states 21.5 / 23.0 / 23.5 / 23.0 pp at
    control rates 0.2 / 0.3 / 0.4 / 0.5 without naming its formula; this one gives 21.4 / 22.8 / 23.1 /
    22.6, within half a point of them, and the module records both sets side by side rather than
    claiming an exact match.
    """
    p2 = p_control
    lo, hi = 0.0, 1.0 - p2
    for _ in range(200):
        d = (lo + hi) / 2
        p1 = p2 + d
        pbar = (p1 + p2) / 2
        need = (z_alpha * math.sqrt(2 * pbar * (1 - pbar))
                + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / n
        if d * d < need:
            lo = d
        else:
            hi = d
    return (lo + hi) / 2


PREREGISTERED_DETECTABLE = {"0.2": 0.215, "0.3": 0.230, "0.4": 0.235, "0.5": 0.230}


def score_file(name: str, rows: list[dict[str, str]], key_rows: list[dict]) -> dict[str, Any]:
    by_key = {k["row"]: k for k in key_rows}
    first = [r for r in rows if by_key[r["row"]]["occurrence"] == 1]
    if len(first) != N_ITEMS:
        raise CodingFileError("{}: {} first occurrences, expected {}".format(name, len(first), N_ITEMS))
    agree = sum(1 for r in first if r["label"] == by_key[r["row"]]["gold"])
    conf = {a: {g: 0 for g in CLASSES} for a in LABELS}
    for r in first:
        conf[r["label"]][by_key[r["row"]]["gold"]] += 1
    pairs = [(r["row"], by_key[r["row"]]["pair_of"]) for r in rows if by_key[r["row"]]["occurrence"] == 2]
    if len(pairs) != N_PAIRS:
        raise CodingFileError("{}: {} repeat pairs, expected {}".format(name, len(pairs), N_PAIRS))
    label = {r["row"]: r["label"] for r in rows}
    same = sum(1 for second, firstrow in pairs if label[second] == label[firstrow])
    unclear = sum(1 for r in first if r["label"] == "unclear")
    return {
        "n_items": len(first),
        "agree": agree,
        "agreement": agree / len(first),
        "agreement_wilson95": wilson(agree, len(first)),
        "contested": len(first) - agree,
        "contested_share": (len(first) - agree) / len(first),
        "contested_wilson95": wilson(len(first) - agree, len(first)),
        "unclear_in_first_occurrences": unclear,
        "confusion_assigned_by_gold": conf,
        "stability_pairs": len(pairs),
        "stability_identical": same,
        "stability": same / len(pairs),
        "stability_wilson95": wilson(same, len(pairs)),
        "rows_coded": len(rows),
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    args = measuring_main_guard(__doc__, OUT, argv)
    if args is None:
        return 0
    from src.data_loading import load_split

    key = json.loads(KEY.read_text(encoding="utf-8"))
    present = coding_files_present()
    text_by_id = {str(i): str(t) for i, t in zip(*[load_split("dbo", "2025", "train")[c] for c in ("id", "description")])}

    out: dict[str, Any] = {
        "experiment": "E5 manual label inspection, Section 4.5",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pre_registration": "THESIS_LOG NS 124 (z14) point 8, (z16) point 3",
        "interval_note": "per-file intervals are uncorrected 95 % Wilson intervals (the finite-population form "
                         "is unchecked at an original); the difference interval is Newcombe's hybrid score",
        "files_seen_in_directory": present,
        "inputs": {},
        "files": {},
    }
    for name, fname in FILES.items():
        path = SAMPLE_DIR / fname
        rows, fmt = read_coded(path)
        kd = key["key"][name]
        check_rows(name, rows, kd["rows"])
        backup = SAMPLE_DIR / fname.replace(".csv", ".coded_backup.csv")
        norm = write_normalised(name, rows, kd["rows"])
        expected = {k["row"]: text_by_id[k["id"]].replace("\r\n", "\n").replace("\n", NEWLINE_MARK) for k in kd["rows"]}
        text_mismatch = [r["row"] for r in rows if r["text"] != expected[r["row"]]]
        if text_mismatch:
            raise CodingFileError(
                "{}: {} row(s) whose text differs from the corpus after normalisation ({}); "
                "either the parse is wrong or the coder saw an altered text -- refusing".format(
                    name, len(text_mismatch), text_mismatch[:8]))
        out["inputs"][name] = {
            "file": fname, "format": fmt, "sha256": sha(path), "rows": len(rows),
            "backup_identical": backup.exists() and backup.read_bytes() == path.read_bytes(),
            "normalised_copy": norm.name, "normalised_sha256": sha(norm),
            "rows_whose_text_differs_from_the_corpus": text_mismatch,
        }
        res = score_file(name, rows, kd["rows"])
        res["stratum"] = kd["stratum"]
        out["files"][name] = res

    irr = next(v for v in out["files"].values() if v["stratum"] == "IRR")
    ctl = next(v for v in out["files"].values() if v["stratum"] == "CTL")
    n = N_ITEMS
    diff = irr["contested_share"] - ctl["contested_share"]
    out["group_comparison"] = {
        "quantity": "contested share of the irreducible items (IRR) minus contested share of the control items (CTL)",
        "difference": diff,
        "difference_newcombe95": newcombe(irr["contested"], n, ctl["contested"], n),
        "difference_wald95": wald_diff(irr["contested"], n, ctl["contested"], n),
        "design_worst_case_wald_half_width": Z * math.sqrt(2 * 0.25 / n),
        "detectable_difference_80pct_power_at_observed_control_rate": detectable_difference(ctl["contested_share"], n),
        "detectable_difference_80pct_power_at_control_rate": {
            str(p): detectable_difference(p, n) for p in (0.2, 0.3, 0.4, 0.5)},
        "detectable_difference_as_stated_in_the_pre_registration_z16": PREREGISTERED_DETECTABLE,
        "these_are_two_different_quantities": "the interval half-width says how precisely the difference is "
                                              "estimated; the detectable difference says what a test at 80 % power "
                                              "could have found. Neither is the other.",
    }
    dist = key["control_k_wrong_distribution"]
    out["control_k_wrong_distribution"] = dist
    out["control_k_wrong_mean"] = sum(int(k) * v for k, v in dist.items()) / sum(dist.values())
    out["limitations_carried"] = [
        "the coder is the author of the thesis, not an independent annotator",
        "set_a was always the second file coded and therefore carries more practice and more fatigue than set_b; "
        "this cannot be corrected, only named",
        "the control group is closer in difficulty than an all-correct group but does not hold it equal "
        "(control_k_wrong_distribution)",
        "single-coder stability is not inter-rater reliability (McDonald, Schoenebeck & Forte 2019)",
        "no per-class statement is supported at these sample sizes, least of all `nothing` at n = 3",
    ]
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote {}".format(args.out))
    for name, v in out["files"].items():
        print("{} ({}): agreement {}/{} = {:.4f} {} | stability {}/{}".format(
            name, v["stratum"], v["agree"], v["n_items"], v["agreement"],
            [round(x, 4) for x in v["agreement_wilson95"]], v["stability_identical"], v["stability_pairs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
