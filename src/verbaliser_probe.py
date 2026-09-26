"""
verbaliser_probe.py -- can the LLM components' label sets be read off as
probabilities at all? (a pre-flight check of the E2 LLM arm)

Why this exists
---------------
The pipeline design promises that the generative component is restricted to the
verbalised labels of the subtask, in the sense of the verbalisers of Schick &
Schuetze (2021), and that "the probabilities it assigns to those label tokens
are read off and normalised" (3.2, subsec:llm). The combination layer consumes
that as a row of an (n_items, n_classes) probability matrix like every other component's.

That sentence contains a hidden assumption: that a label *is* a token. This
module measures whether it is, and the answer is no for every multi-class
subtask on at least one of the two tokenisers.

The criterion that matters is NOT "are the first tokens distinct"
--------------------------------------------------------------
First-token distinctness holds for every candidate set on both tokenisers, and
it is close to worthless. What decides whether a first-token reading is sound is
whether the first token is *specific to its label*. Measured here:

    LLaeMmlein, VIO 6-way German : Verherrlichung -> ['_Ver','herr','lichung']
    LLaeMmlein, DBO German       : Hetze -> ['_H','etze'], Umsturz -> ['_Um','sturz']
    LLaeMmlein, DBO English      : criticism -> ['_c','rit','ic','ism'], nothing -> ['_n',...]

`_Ver`, `_H`, `_Um`, `_c` and `_n` are generic German prefixes and bare letters.
Reading P(`_Ver`) as P(glorification) charges that label with the mass of every
Ver- continuation the model might produce. It biases the estimate *upward* on
precisely the thinnest class in the study (VIO glorification, n = 27 in 2026 and
n = 12 in the overlap-free residual), which is the one place the project
has repeatedly said an inflated number must not be allowed to appear.

The two tokenisers are mirror images, which is itself a result: LLaeMmlein
(32 000, German-only) handles German verbalisers well and English label names
badly; Qwen 2.5 (151 643, multilingual) does the reverse. A verbaliser set
chosen to tokenise cleanly for one model is therefore the wrong set for the
other, and no single choice of words escapes the problem.

What the measurement decided
----------------------------
Full-sequence scoring instead of first-token probability: each candidate label
is scored as the length-normalised log-likelihood of its whole token sequence
given the prompt, and the label set is normalised over. This is independent of
how any tokeniser happens to split a word, so the design stops being hostage to
a tokenisation accident, and it keeps the probability interface exactly as written.

The residual risk it does NOT remove is surface-form competition (`holtzman2021`,
EMNLP 2021): likelihood still depends on the surface string, so verbaliser words
must be comparable in kind. That is why the German sets here are single common
nouns rather than a mix of nouns and phrases.

Note the scope honestly rather than importing that paper's severity: it studies
open multiple choice, where the candidate strings differ from question to
question. Here the label set is small, fixed and identical for every item of a
subtask, so the competition is a constant per-class prior rather than per-item
noise -- and the fine-tuned arm is trained to emit the verbaliser on top of that.
The untrained few-shot component is where the residual actually bites.

Usage
-----
    python -m src.verbaliser_probe            # writes results/verbaliser_probe.json

Tokenisers only. No weights are downloaded and `trust_remote_code` is never set.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from src.harness import measuring_main_guard

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Tokenisers of the two LLM components. The ids are the ones e2_matrix.yaml
# pins under components.llm_llammlein / components.llm_qwen_fewshot.
MODELS = {
    "llammlein_7b": "LSX-UniWue/LLaMmlein_7B",
    "qwen25_14b": "Qwen/Qwen2.5-14B-Instruct",
}

# "raw" = the label strings exactly as the CSVs carry them; "de" = the German
# verbaliser words. Both are measured rather than argued about.
VERBALISER_SETS: dict[str, dict[str, str]] = {
    "c2a_raw": {"True": "True", "False": "False"},
    "c2a_de": {"True": "Ja", "False": "Nein"},
    "dbo_raw": {
        "agitation": "agitation", "criticism": "criticism",
        "nothing": "nothing", "subversive": "subversive",
    },
    "dbo_de": {
        "agitation": "Hetze", "criticism": "Kritik",
        "nothing": "Nichts", "subversive": "Subversion",
    },
    "vio25_raw": {"True": "True", "False": "False"},
    "vio25_de": {"True": "Ja", "False": "Nein"},
    "vio26_raw": {
        "call2violence": "call2violence", "glorification": "glorification",
        "nothing": "nothing", "other": "other",
        "propensity": "propensity", "support": "support",
    },
    "vio26_de": {
        "call2violence": "Gewaltaufruf", "glorification": "Verherrlichung",
        "nothing": "Nichts", "other": "Sonstiges",
        "propensity": "Gewaltbereitschaft", "support": "Unterstützung",
    },
}

# A first token shorter than this, or one that is a bare prefix rather than a
# word, cannot carry a label's probability on its own. The threshold is a
# reporting aid, not a decision rule -- the decision (full-sequence scoring) is
# already taken and does not depend on where the line sits.
GENERIC_PREFIX_MAX_CHARS = 4


def _tokenise(tok, word: str) -> dict[str, Any]:
    """Bare and with a leading space; the latter is what a verbaliser after a
    prompt's answer marker actually meets."""
    out: dict[str, Any] = {}
    for variant, text in (("bare", word), ("space", " " + word)):
        ids = tok.encode(text, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
        out[variant] = {
            "n_tokens": len(ids),
            "first_id": int(ids[0]) if ids else None,
            "first_piece": pieces[0] if pieces else None,
            "pieces": pieces,
        }
    return out


def _summarise(per_label: dict[str, Any], variant: str) -> dict[str, Any]:
    firsts = [per_label[lab][variant]["first_id"] for lab in per_label]
    n_tokens = [per_label[lab][variant]["n_tokens"] for lab in per_label]
    # A first piece is called generic when the label needs more tokens than one
    # and the first of them is a short fragment rather than the word itself.
    generic = [
        lab for lab in per_label
        if per_label[lab][variant]["n_tokens"] > 1
        and len(str(per_label[lab][variant]["first_piece"]).lstrip("▁Ġ"))
        <= GENERIC_PREFIX_MAX_CHARS
    ]
    return {
        "all_single_token": all(n == 1 for n in n_tokens),
        "first_token_distinct": len(set(firsts)) == len(firsts),
        "max_tokens": max(n_tokens),
        "labels_with_generic_first_piece": generic,
        "first_token_reading_sound": all(n == 1 for n in n_tokens) and not generic,
    }


def run() -> dict[str, Any]:
    from transformers import AutoTokenizer

    report: dict[str, Any] = {}
    for key, hf_id in MODELS.items():
        print(f"\n=== {key}  ({hf_id}) ===")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  UNAVAILABLE: {type(exc).__name__}: {exc}")
            report[key] = {"hf_id": hf_id, "error": f"{type(exc).__name__}: {exc}"}
            continue

        entry: dict[str, Any] = {
            "hf_id": hf_id,
            "vocab_size": int(tok.vocab_size),
            "sets": {},
        }
        print(f"  vocab_size = {tok.vocab_size}")

        for set_name, mapping in VERBALISER_SETS.items():
            per_label = {lab: _tokenise(tok, word) for lab, word in mapping.items()}
            entry["sets"][set_name] = {
                "words": mapping,
                "bare": _summarise(per_label, "bare"),
                "space": _summarise(per_label, "space"),
                "detail": per_label,
            }
            s = entry["sets"][set_name]["space"]
            verdict = "sound" if s["first_token_reading_sound"] else "UNSOUND"
            print(
                f"  {set_name:<11} first-token reading: {verdict:<8}"
                f" max {s['max_tokens']} tok"
                + (f"  generic first piece: {s['labels_with_generic_first_piece']}"
                   if s["labels_with_generic_first_piece"] else "")
            )
        report[key] = entry
    return report


def main() -> int:
    args = measuring_main_guard(__doc__, RESULTS_DIR / "verbaliser_probe.json")
    if args is None:
        return 0
    payload = {
        "experiment": "E2 LLM arm pre-flight",
        "label": "verbaliser tokenisation under the two LLM components' tokenisers",
        "created": date.today().isoformat(),
        "question": (
            "Can a label's probability be read off a single token, as 3.2's "
            "wording implies and D7 consumes?"
        ),
        "verdict": (
            "No for every multi-class subtask on at least one tokeniser. First-token "
            "distinctness holds everywhere and is not the relevant property; the "
            "relevant property is whether the first token is specific to its label, "
            "and it is not. The arm therefore scores full label sequences with length "
            "normalisation instead of reading a single token. See llm_protocol in "
            "configs/e2_matrix.yaml and THESIS_LOG #54."
        ),
        "anchor": "THESIS_LOG Next Steps #45, #54",
        "models": run(),
    }
    out = args.out
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
