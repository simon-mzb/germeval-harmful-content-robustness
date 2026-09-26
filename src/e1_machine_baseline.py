"""
Measure the two reproduced E1 baselines on THIS machine and store them as a
machine-keyed artefact.

Why this exists. `baseline_comparison` withholds a delta whose two sides were
not produced on the same machine. The TML arm is Darwin/arm64; the encoder and
LLM arms come from a rented Linux GPU machine. Without a Linux-side E1
reference the comparison is withheld on exactly the subtasks where the headline
component wins -- correctly, but Chapter 4 loses the comparison where it matters
most. So the reference stops being one file and becomes a set with one entry per
machine, and this is what writes an entry.

It is deliberately NOT a second implementation of the E1 protocol: the protocol
lives in `e1_platform_check.run`, and this module only composes the artefact
around it. Two copies of a protocol drift.

Usage (on each machine that produces components):
    python -m src.e1_machine_baseline

**Do not force a device here.** DBO is pure scikit-learn and never sees
one; C2A encodes through SentenceBERT, `baselines.py` passes no device, and the
library therefore picks -- mps on a Mac, cuda on the GPU machine. That is the
right behaviour and not an oversight: the reference has to be served the way the
arm it will be compared against is served, and the machine key is OS plus
architecture, so a CPU-forced baseline beside a CUDA encoder run would be a
device mismatch *inside* one machine that no guard can see. The resolved device
is recorded in `env_pin.torch_auto_device` and printed below.

Refuses by default to write an entry for a machine that already has one --
`load_baselines` raises on a duplicate, so writing one would break the summary
rather than extend it. `--force` overwrites the entry for THIS machine only.
"""

from __future__ import annotations

import argparse
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

from src.e1_platform_check import HOLDOUT_FRAC, RESULTS, SEED, run
from src.harness import env_pin, save_results

# Descriptive labels, not measurements: they name which implementation was run,
# and there is no artefact they could be derived from. Kept beside the protocol
# they describe so a new baseline cannot acquire a stale name.
BASELINE_NAMES = {
    "dbo": "DBO-TF-IDF-SVC",
    "c2a": "C2A-GradBoost-SentenceBERT",
    "vio": "VIO-Qwen2.5-32B-Ollama",
}

# The default a bare call measures. vio is opt-in (`--subtask vio`) because it needs
# a served qwen2.5:32b; a bare call must not start depending on Ollama.
SUBTASKS = ("dbo", "c2a")
ALL_SUBTASKS = ("dbo", "c2a", "vio")

VIO_DEVIATIONS = [
    "evaluated on our own stratified 80/20 holdout of the deduplicated 2025 train pool, "
    "as the c2a and dbo reproductions are, not on the official test set",
    "options.seed = 123 + trial index added so the run is repeatable and a retry still "
    "resamples; the notebook's top-level seed field is not read by Ollama",
    "served model is the Ollama library tag qwen2.5:32b (quantisation under `serving`); "
    "the organisers' quantisation is not stated",
    "Python urllib client instead of R httr; the body is built field for field from the "
    "notebook, str_escape on prompt and modelfile included, prompts read from the Rmd itself",
    "items unparsed after ten trials are set to false, as the organisers did after "
    "their manual re-attempt",
]


def machine_slug(platform_string: str) -> str:
    """
    'Linux/x86_64' -> 'linux-x86-64'. Filename-safe, lossless enough to read.

    (Not 'linux-x86_64': the character class strips the underscore along with
    the slash. Harmless -- nothing resolves a baseline by the slug -- but a
    docstring that shows the wrong filename sends a reader looking for a file
    that does not exist, which is what it did.)

    The slug is for humans reading `ls`; nothing resolves a baseline by it. The
    machine of record is always the `env_pin.platform` inside the artefact, so a
    renamed file changes nothing about which machine it counts as.
    """
    return re.sub(r"[^a-z0-9]+", "-", platform_string.lower()).strip("-")


def artefact_path(subtask: str, platform_string: str,
                  results_dir: Path = RESULTS) -> Path:
    return Path(results_dir) / f"e1_{subtask}_baseline__{machine_slug(platform_string)}.json"


def existing_machines(subtask: str, results_dir: Path = RESULTS) -> dict[str, str]:
    """Machines that already have an entry for this subtask -> the file holding it."""
    from src.e2_summary import load_baselines
    entry = load_baselines(Path(results_dir))[subtask]
    return {m: rec["source"] for m, rec in entry["by_machine"].items()}


def this_machine() -> str:
    """
    The machine string without importing anything heavy.

    `env_pin()` produces the same value, but it imports torch, lightgbm and
    SentenceBERT to do it, and that cannot happen before the runs -- see the
    ordering note in `main`. This is the identical expression from
    `harness.env_pin`, and `main` asserts the two agree before writing.
    """
    return f"{platform.system()}/{platform.machine()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="overwrite the entry for this machine if one exists")
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--subtask", nargs="+", choices=ALL_SUBTASKS, default=list(SUBTASKS),
                    help="default: dbo c2a. vio needs --ollama-url serving qwen2.5:32b")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    here = this_machine()

    # Decide what to measure BEFORE measuring it. The C2A baseline fits
    # SentenceBERT and costs minutes; discovering afterwards that this machine
    # already has an entry would spend them for nothing.
    todo = []
    for st in args.subtask:
        have = existing_machines(st, results_dir)
        if here in have and not args.force:
            print(f"{st.upper():<4} SKIP: {here} already has an entry "
                  f"({have[here]}). Re-measure with --force only if you mean "
                  f"to replace it.")
            continue
        todo.append(st)

    if not todo:
        print(f"\nnothing to do: {here} is already in the baseline set.")
        _report(results_dir)
        return

    vio_extra = {}
    if "vio" in todo:
        # Fail before any measuring if the model is not served, and measure what the
        # organisers' request fields actually do on this server.
        from src.baselines import VIO_OLLAMA_MODEL, VIOBaselineOllama
        probe_model = VIOBaselineOllama(args.ollama_url)
        serving = probe_model.serving_info()
        if not serving["served"]:
            raise SystemExit(f"{VIO_OLLAMA_MODEL} is not served at {args.ollama_url}")
        vio_extra = {"serving": serving,
                     "request_semantics_probe": probe_model.probe_request_semantics()}
        print(f"VIO  serving {serving}\n     probe {vio_extra['request_semantics_probe']}")
    calls_log = Path(results_dir) / f"e1_vio_ollama_calls__{machine_slug(here)}.jsonl"
    full = {}
    for st in todo:
        kw = {"ollama_url": args.ollama_url, "log_path": calls_log} if st == "vio" else {}
        full[st] = run(st, bootstrap=True, **kw)
    measured = {st: full[st]["metrics"] for st in todo}

    # env_pin ONCE, and only after every model run. It imports lightgbm, torch
    # and SentenceBERT; a process that then loads SentenceBERT again brings up a
    # second OpenMP runtime and dies with SIGSEGV. The rule is therefore that
    # env_pin comes after ALL model runs, not merely before the next one. Calling it between the DBO and C2A runs of this
    # very module segfaults, which is how the wording was corrected.
    from src.component_store import compute_data_rules_id
    from src.e2_summary import ORGANIZER

    pin = env_pin()
    if pin["platform"] != here:
        raise RuntimeError(
            f"machine string disagrees: pre-check {here!r}, env_pin "
            f"{pin['platform']!r}. The artefact would be keyed under a machine "
            "the pre-check never tested for duplicates.")
    rules_id = compute_data_rules_id()
    print(f"\nmachine {pin['platform']}, torch serves "
          f"{pin.get('torch_auto_device')} (cuda_available="
          f"{pin['cuda_available']})")

    for st in todo:
        metrics = measured[st]
        target = ORGANIZER[st]
        payload = {
            "subtask": st,
            "baseline": BASELINE_NAMES[st],
            "edition": "2025",
            "eval_set": "own_stratified_holdout_80_20",
            "holdout_frac": HOLDOUT_FRAC,
            "seed": SEED,
            "organizer_target_macro_f1": target,
            "organizer_eval_set": "official_test_gold_unavailable",
            "delta_pp_vs_organizer": metrics["macro_f1"] * 100 - target,
            **metrics,
            "env_pin": pin,
            "data_rules_id": rules_id,
            "produced_by": "src/e1_machine_baseline.py",
            "protocol_source": ("src/e1_platform_check.run (mirrors "
                                "notebooks/01_baseline_repro.ipynb)"),
            "anchor": "THESIS_LOG Next Steps #67: E1 references are keyed by machine",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if st == "vio":
            payload.update({
                "protocol_source": ("src/e1_platform_check.run + baselines.VIOBaselineOllama "
                                    "(mirrors data/codabench/GermEval2025/baseline/vio/vio_baseline.Rmd)"),
                "model_info": full[st].get("model_info"),
                "calls_log": calls_log.name,
                "deviations": VIO_DEVIATIONS,
                "anchor": "THESIS_LOG Next Steps #15 / roadmap B14 (E1b)",
                **vio_extra,
            })
        out = artefact_path(st, here, results_dir)
        save_results(payload, out)
        print(f"{st.upper():<4} {metrics['macro_f1']*100:.4f} on {here} -> {out.name}")

    _report(results_dir)


def _report(results_dir: Path) -> None:
    """
    Reload the set through the function the summary uses.

    This is what makes the run usable rather than merely written: a duplicate
    machine or a missing env_pin raises here, on the machine, instead of in the
    notebook three steps later.
    """
    from src.e2_summary import load_baselines
    reloaded = load_baselines(results_dir)
    print("\nbaseline set now:")
    for st in ALL_SUBTASKS:
        entries = sorted(reloaded[st]["by_machine"].items())
        if not entries:
            print(f"  {st:<4} (none)")
        for m, rec in entries:
            print(f"  {st:<4} {m:<20} {rec['reproduced']:.4f}  ({rec['source']})")


if __name__ == "__main__":
    main()
