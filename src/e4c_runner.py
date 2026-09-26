"""
e4c_runner.py -- Dimension 3: the 2025-trained components, scored once on 2026.

WHAT IT RUNS. For each component and subtask: ONE final fit on the full
deduplicated 2025 train pool, at the configuration `e4_config` froze for that
subtask, with epoch selection and temperature scaling from the usual inner
split -- the same fit a fold runs, on all of the 2025 data. Then one scoring pass
over the frozen 2026 evaluation pool (`results/e4c_eval_pool.json`: 2026 train
minus every 2025 train text). Nothing is tuned on 2026
(`dimension3.retraining_on_2026: false`).

WHY A FINAL FIT AND NOT THE E2 FOLD MODELS. E2 never kept a model -- no adapter
was saved -- and even if it had, five fold models trained on 80 % of the
pool are an ensemble, a different system from the one E2 reports. One fit on the
whole pool is the standard reading of "the 2025-trained component".

THE LABEL SPACES. C2A and DBO keep their label set across editions. VIO does
not: 2026 splits the positive class five ways, and `e4_protocol.e4c.vio_mapping`
collapses them against `nothing` exactly as 3.1 states. Because a mapping is an
assumption, the artefact also carries, for every ORIGINAL 2026 class, the share
the binary model flags as positive -- a figure that needs no mapping for the five
positive classes at all.

WHAT IT KEEPS. Probabilities in the component store under `__e4c__` (so the
combination layer can be applied to 2026 unchanged), the run artefact under
results/e4c/, and -- for the encoder and the fine-tuned LLM -- the LoRA adapter
with its temperature under results/component_store/models/e4c/ (gitignored;
fetched explicitly), so later inference needs no re-training.

Usage
-----
    python -m src.e4c_runner                                   # describe the plan; runs nothing
    python -m src.e4c_runner --go --arm llm --device cuda      # llm_llammlein, all three subtasks
    python -m src.e4c_runner --go --component tml_svm --subtask dbo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src import e4_config, imbalance
from src.component_store import ComponentStore, compute_config_id, run_key
from src.data_loading import load_split
from src.e2_runner import (PRIMARY_SEED, TML_COMPONENTS, _git_commit, component_api,
                           compute_data_rules_id, family_of, load_matrix, load_pool)
from src.harness import (apply_data_quality_rules, env_pin, pooled_oof_report,
                         resolve_classes, save_results, tier_for_n)

_ROOT = Path(__file__).parent.parent
POOL_PATH = _ROOT / "results" / "e4c_eval_pool.json"
OUT_DIR = _ROOT / "results" / "e4c"
ADAPTER_DIR = _ROOT / "results" / "component_store" / "models" / "e4c"
SUBTASKS = ("c2a", "dbo", "vio")
ARMS = {"tml": list(TML_COMPONENTS), "encoder": ["encoder_1b"],
        "llm": ["llm_llammlein"], "fewshot": ["llm_qwen_fewshot"]}


def e4c_protocol(matrix: dict[str, Any]) -> dict[str, Any]:
    block = imbalance.protocol(matrix)
    status = str(block.get("status", ""))
    if not status.startswith("confirmed"):
        raise RuntimeError("REFUSING E4c: e4_protocol.status is {!r}".format(status))
    if "e4c" not in block:
        raise RuntimeError("REFUSING E4c: e4_protocol carries no e4c block")
    return block["e4c"]


def eval_pool(subtask: str, pool_path: Path | None = None):
    """The frozen 2026 pool, rebuilt from the raw split and checked against its hash."""
    frozen = json.loads(Path(pool_path or POOL_PATH).read_text(encoding="utf-8"))
    entry = next(p for p in frozen["pools"] if p["subtask"] == subtask)
    df, _ = apply_data_quality_rules(load_split(subtask, "2026", "train"),
                                     dedup=True, drop_overlap=False)
    wanted = set(entry["ids"])
    pool = df[df["id"].astype(str).isin(wanted)].reset_index(drop=True)
    digest = hashlib.sha256(
        "\n".join(sorted(str(i) for i in pool["id"].tolist())).encode("utf-8")).hexdigest()
    if len(pool) != entry["n_pool"] or digest != entry["ids_sha256"]:
        raise ValueError(
            "the 2026 pool rebuilt for {} ({} items) does not match the frozen one "
            "({} items, hash {}...). Scoring a different pool than the one frozen "
            "before any model saw it would void Dimension 3's out-of-sample claim."
            .format(subtask, len(pool), entry["n_pool"], entry["ids_sha256"][:12]))
    return pool, entry


def map_labels(subtask: str, labels, classes_2025: list, protocol: dict) -> np.ndarray:
    """2026 gold labels in the 2025 label space; VIO through the declared mapping."""
    labels = list(labels)
    if subtask == "vio":
        mapping = protocol["vio_mapping"]
        unknown = sorted({str(v) for v in labels} - set(mapping))
        if unknown:
            raise ValueError("VIO 2026 labels with no declared mapping: {}".format(unknown))
        mapped = [bool(mapping[str(v)]) for v in labels]
    else:
        mapped = labels
    known = {str(c) for c in classes_2025}
    stray = sorted({str(v) for v in mapped} - known)
    if stray:
        raise ValueError("{} 2026 labels outside the 2025 label set {}: {}".format(
            subtask, sorted(known), stray))
    by_str = {str(c): c for c in classes_2025}
    # NOT dtype=object: an object array of bools is target type "unknown" to
    # scikit-learn, so f1_score would raise on c2a and vio -- after the fit,
    # which on a rented GPU is the paid part. The natural dtype
    # matches the 2025 training labels (bool for the binary subtasks, str for dbo).
    return np.asarray([by_str[str(v)] for v in mapped])


def source_class_detection(original, proba, classes) -> dict[str, Any]:
    """Per ORIGINAL 2026 class: support, tier, and the share predicted positive."""
    pred = np.asarray([classes[i] for i in proba.argmax(axis=1)])
    positive = next(c for c in classes if str(c) == "True")
    original = np.asarray([str(v) for v in original])
    out = {}
    for c in sorted(set(original)):
        mask = original == c
        n = int(mask.sum())
        out[c] = {"n": n, "tier": tier_for_n(n),
                  "share_flagged_positive": float(np.mean(pred[mask] == positive))}
    return out


def _save_adapter(model, component: str, subtask: str, adapter_dir: Path) -> dict | None:
    """The trained weights and their temperature, for later inference without re-training.

    The few-shot LLM trains nothing, but its `model_` still has
    `save_pretrained`, which would write the untouched Qwen base weights --
    29.5 GB per subtask, identical across subtasks. For `llm_fewshot` only the
    temperature is saved: it is the one thing that cell fitted.
    """
    fewshot = family_of(component) == "llm_fewshot"
    if not fewshot and (getattr(model, "model_", None) is None
                        or not hasattr(model.model_, "save_pretrained")):
        return None
    path = Path(adapter_dir) / "{}_{}".format(subtask, component)
    path.mkdir(parents=True, exist_ok=True)
    if not fewshot:
        model.model_.save_pretrained(str(path))
    (path / "e4c_calibration.json").write_text(json.dumps({
        "temperature": float(model.temperature_), "classes": [str(c) for c in model.classes_],
        "component": component, "subtask": subtask}, indent=2), encoding="utf-8")
    files = {}
    for f in sorted(path.iterdir()):
        if f.is_file():
            files[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    try:
        shown = str(path.resolve().relative_to(_ROOT.resolve()))
    except ValueError:              # an adapter_dir outside the repository (tests)
        shown = str(path)
    out = {"path": shown, "sha256": files}
    if fewshot:
        out["note"] = "few-shot: no trained weights; the base model is not copied (h14)"
    return out


def run_cell(component: str, subtask: str, *, device: str | None = None,
             bootstrap_n: int = 1000, store: ComponentStore | None = None,
             out_dir: Path = OUT_DIR, adapter_dir: Path = ADAPTER_DIR,
             force: bool = False, pool_path: Path | None = None) -> dict[str, Any]:
    matrix = load_matrix()
    protocol = e4c_protocol(matrix)
    fam = family_of(component)
    fixed = (None if fam == "llm_fewshot"
             else e4_config.fixed_params_for(component, subtask, "2025", "e4a", matrix=matrix))
    e4 = {"experiment": "E4C", "condition": "none", "fixed_params": fixed}
    config_id = compute_config_id(component, matrix, e4=e4)
    out_path = Path(out_dir) / "e4c_{}_{}.json".format(subtask, component)
    if out_path.exists() and not force:
        prior = json.loads(out_path.read_text(encoding="utf-8"))
        if prior.get("meta", {}).get("config_id") != config_id:
            raise RuntimeError("REFUSING to skip {}: its config_id is {}, the matrix now "
                               "gives {}".format(out_path.name,
                                                 prior.get("meta", {}).get("config_id"), config_id))
        print("  [skip] {} exists and matches config {}".format(out_path.name, config_id))
        return prior

    train_df = load_pool(subtask, "2025")
    classes = resolve_classes(train_df["label"].values)
    pool, entry = eval_pool(subtask, pool_path)
    y_true = map_labels(subtask, pool["label"].values, classes, protocol)
    print("\n=== E4c / {} / {}: fit on 2025 (n={}), score 2026 (n={}) ===".format(
        subtask, component, len(train_df), len(pool)))

    api = component_api(component, subtask=subtask, device=device,
                        grid=[fixed] if fixed is not None else None)
    t0 = time.time()
    model = api["make_train_fn"](classes, PRIMARY_SEED)(train_df)
    fit_seconds = time.time() - t0
    t1 = time.time()
    proba = np.asarray(api["predict_proba_fn"](model, pool, classes), dtype=np.float64)
    score_seconds = time.time() - t1

    report = pooled_oof_report(y_true, proba, classes, bootstrap=True,
                               bootstrap_n=bootstrap_n)
    detection = (source_class_detection(pool["label"].values, proba, classes)
                 if subtask == "vio" else None)

    # ⚠️ ORDER: the store entry and the artefact are written BEFORE the adapter.
    # The adapter is an optional side output; if it ran first, a full disk
    # during its save would destroy hours of paid scoring. If the save fails, the artefact on disk says `adapter: pending` and a
    # retry skips the cell instead of re-scoring it.
    store = store if store is not None else ComponentStore()
    key = run_key(subtask, "2026", component, PRIMARY_SEED, variant="e4c")
    store.save(key, proba=proba, y_true=y_true, ids=pool["id"].values, classes=classes,
               fold=np.full(len(pool), -1), meta={
                   "subtask": subtask, "edition": "2026", "component": component,
                   "seed": PRIMARY_SEED, "n_splits": 0,
                   "aggregation": "one final fit on 2025, scored once on 2026",
                   "uncalibrated": False, "config_id": config_id,
                   "data_rules_id": compute_data_rules_id(),
                   "matrix_version": matrix["meta"]["version"],
                   "experiment": "E4c", "e4": e4, "trained_on": "2025",
                   "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    result = {
        "experiment": "E4c", "package": "B10", "git_commit": _git_commit(),
        "subtask": subtask, "component": component,
        "meta": {"config_id": config_id, "e4": e4, "family": fam,
                 "matrix_version": matrix["meta"]["version"],
                 "data_rules_id": compute_data_rules_id(),
                 "train_edition": "2025", "eval_edition": "2026",
                 "n_train": int(len(train_df)), "n_eval": int(len(pool)),
                 "eval_pool_sha256": entry["ids_sha256"],
                 "label_mapping": protocol["vio_mapping"] if subtask == "vio" else None,
                 "classes": [str(c) for c in classes],
                 "bootstrap_n": bootstrap_n, "env_pin": env_pin(),
                 "created": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "primary": report,
        "source_class_detection": detection,
        "fit_record": model.record,
        "adapter": {"status": "pending"},
        "component_store_key": key,
        "fit_seconds": fit_seconds, "score_seconds": score_seconds,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_results(result, out_path)
    result["adapter"] = _save_adapter(model, component, subtask, adapter_dir)
    save_results(result, out_path)
    print("  2026 macro-F1 {:.4f}  (fit {:.0f}s, score {:.0f}s)".format(
        report["macro_f1"], fit_seconds, score_seconds))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="actually run; without it, describe")
    ap.add_argument("--arm", choices=sorted(ARMS), default=None)
    ap.add_argument("--component", nargs="+", default=None)
    ap.add_argument("--subtask", nargs="+", choices=list(SUBTASKS), default=list(SUBTASKS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    components = ARMS[args.arm] if args.arm else (args.component or [])
    if not components:
        print((__doc__ or "").strip())
        print("\nNothing selected: pass --arm or --component (and --go to run).")
        return 0
    jobs = [(c, st) for c in components for st in args.subtask]
    print("E4c: {} cell(s)".format(len(jobs)))
    for c, st in jobs:
        print("  - {} / {}".format(st, c))
    if not args.go:
        print("\n(describe only -- add --go to run)")
        return 0
    for c, st in jobs:
        run_cell(c, st, device=args.device, bootstrap_n=args.bootstrap_n, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
