"""
e5_examples.py -- pick the three example items of Section 4.5 by a rule fixed before any item was read.

THE RULE (written down and fixed before a single row of the coding files was read). Each case is the FIRST row, in the order of the coding file (r001 upward), among
the first occurrences of the items (`occurrence` = 1 in the sidecar) that meets its condition:

  (i)   set_b (the irreducible items), and the coder's class equals the gold class:  a real model error
  (ii)  set_b (the irreducible items), and the coder's class differs from gold:      a disputed label
  (iii) set_a (the control items),     and the coder's class differs from gold:      the contrast to (ii)

WHY THIS RULE AND NOT A CHOSEN EXAMPLE. A hand-picked example is the cherry-picking a reader suspects first;
"the first in the drawn order" is checkable by anyone with the two files, and the order itself was fixed by
a seed before the coding. (iii) is the control item with a differing class, and not simply the first control
row, because then only the stratum varies against (ii), not also the condition.

WHAT THE ARTEFACT HOLDS. Row, id, file, gold class, the class of each of the six components (the argmax of
its stored calibrated probabilities), the coder's class, the number of wrong components, and a SHA-256 of
the item's text and its length. It does NOT hold the text: the corpus is not redistributed by this repository
and the three texts are printed in the thesis only; the verifier reads the text from the corpus by id and
checks it against the hash and against the printed passage.

Usage
-----
python -m src.e5_examples --help
python -m src.e5_examples          # writes results/e5_examples.json; refuses to overwrite
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src import e3_combination as e3
from src.e5_irreducible import COMPONENTS
from src.e5_label_analysis import FILES, KEY, NEWLINE_MARK, SAMPLE_DIR, check_rows, read_coded
from src.harness import measuring_main_guard

OUT = Path(__file__).resolve().parents[1] / "results" / "e5_examples.json"

CASES = [
    ("i", "set_b", lambda coder, gold: coder == gold,
     "irreducible item, the coder's class equals the gold class: a real model error"),
    ("ii", "set_b", lambda coder, gold: coder != gold,
     "irreducible item, the coder's class differs from the gold class: a disputed label"),
    ("iii", "set_a", lambda coder, gold: coder != gold,
     "control item, the coder's class differs from the gold class: the contrast to (ii)"),
]


def item_text(load_split, item_id: str) -> str:
    split = load_split("dbo", "2025", "train")
    row = split[split["id"].astype(str) == item_id]
    if len(row) != 1:
        raise SystemExit("id {} is not exactly once in the 2025 dbo train split".format(item_id))
    return str(row["description"].iloc[0])


def main(argv=None) -> int:
    args = measuring_main_guard(__doc__, OUT, argv)
    if args is None:
        return 0
    from src.component_store import ComponentStore
    from src.data_loading import load_split

    key = json.loads(KEY.read_text(encoding="utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        recs = e3.load_subtask("dbo", COMPONENTS, ComponentStore())
    base = recs[COMPONENTS[0]]
    classes = [str(c) for c in base["classes"]]
    ids = [str(i) for i in base["ids"]]
    index = {i: n for n, i in enumerate(ids)}
    proba = {c: np.asarray(recs[c]["proba"]) for c in COMPONENTS}

    coded = {}
    for name, fname in FILES.items():
        rows, _ = read_coded(SAMPLE_DIR / fname)
        check_rows(name, rows, key["key"][name]["rows"])
        coded[name] = {r["row"]: r["label"] for r in rows}

    out_cases: list[dict[str, Any]] = []
    for tag, file_name, cond, description in CASES:
        chosen = None
        for k in key["key"][file_name]["rows"]:          # file order, r001 upward
            if k["occurrence"] != 1:
                continue
            if cond(coded[file_name][k["row"]], k["gold"]):
                chosen = k
                break
        if chosen is None:
            raise SystemExit("no row meets case ({})".format(tag))
        j = index[chosen["id"]]
        preds = {c: classes[int(proba[c][j].argmax())] for c in COMPONENTS}
        text = item_text(load_split, chosen["id"]).replace("\r\n", "\n").replace("\n", NEWLINE_MARK)
        out_cases.append({
            "case": tag, "description": description, "file": file_name,
            "stratum": key["key"][file_name]["stratum"],
            "row": chosen["row"], "id": chosen["id"], "gold": chosen["gold"],
            "coder": coded[file_name][chosen["row"]],
            "predictions": preds,
            "k_wrong": int(sum(p != chosen["gold"] for p in preds.values())),
            "k_wrong_in_sidecar": chosen["k_wrong"],
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_chars": len(text),
        })
        assert out_cases[-1]["k_wrong"] == chosen["k_wrong"], (tag, out_cases[-1]["k_wrong"], chosen["k_wrong"])

    art = {
        "experiment": "E5 example items, Section 4.5",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "first row in coding-file order among first occurrences that meets the condition; fixed "
                "before any row was read (THESIS_LOG NS 124 z24)",
        "components_in_order": COMPONENTS,
        "cases": out_cases,
    }
    args.out.write_text(json.dumps(art, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote {}".format(args.out))
    for c in out_cases:
        print("({}) {} {} gold={} coder={} k_wrong={}".format(c["case"], c["file"], c["row"], c["gold"], c["coder"], c["k_wrong"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
