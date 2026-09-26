"""
component_store.py -- persistent store for component outputs.

The combination layer (E3) must compare soft voting, the confidence cascade
and the LLM-as-judge **on identical component outputs**; otherwise a difference
between strategies could be a difference between two training runs of the same
component. Section 3.2 states that components are trained once and reused, and
this module is what makes that literally true across runs: every E2 run
writes its pooled out-of-fold probability matrix here, and E3 reads it back
without retraining anything.

Layout
------
results/component_store/
    <subtask>_<edition>/
        <component>__seed<seed>.npz     out-of-fold probabilities + gold + ids
        <component>__seed<seed>.json    metadata sidecar (human-readable)
    models/                             optional fitted artefacts, gitignored

The .npz files are small (n_items x n_classes float32, a few hundred kB per
run) and are the inputs to every E3 number. Fitted model artefacts are not
kept: TF-IDF vocabularies and tree ensembles run to tens of megabytes and are
cheap to refit on CPU.

Usage
-----
from src.component_store import ComponentStore

store = ComponentStore()
store.save(key, proba=..., y_true=..., ids=..., classes=..., fold=..., meta=...)
rec = store.load(key)
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ROOT = Path(__file__).parent.parent / "results" / "component_store"
MATRIX_PATH = Path(__file__).parent.parent / "configs" / "e2_matrix.yaml"

# Which protocol block and which variance row govern a component family. Part
# of the config identity below: a change to the family protocol changes what a
# stored run means, a change to another family's protocol does not.
_FAMILY_PROTOCOL = {
    "tml": "tml_protocol",
    "encoder": "encoder_protocol",
    "llm_finetune": "llm_protocol",
    "llm_fewshot": "llm_fewshot_protocol",
}


def parse_run_key(key: str) -> dict[str, Any]:
    """Split a store key into cell, component, variant (None for E2) and seed.

    ⚠️ WHY THIS EXISTS. Selecting runs by SUFFIX (keep every key ending in
    `__seed42`) is wrong once E4 entries exist: an E4 key
    (`encoder_1b__e4a-none__seed42`) ends the same way and would be paired
    against E2 silently. Readers ask this function
    instead of pattern-matching the name.
    """
    cell, _, stem = key.partition("/")
    parts = stem.split("__")
    if len(parts) < 2 or not parts[-1].startswith("seed"):
        raise ValueError("not a component-store key: {!r}".format(key))
    return {"cell": cell, "component": parts[0],
            "variant": "__".join(parts[1:-1]) or None,
            "seed": int(parts[-1][len("seed"):])}


def run_key(subtask: str, edition: str, component: str, seed: int,
            variant: str | None = None) -> str:
    """Canonical identifier for one component run. Used as the file stem.

    `variant` names an E4 cell (`e4a-focal_loss`, `e4b`). E2 keys carry none, so
    every stored E2 key is unchanged, and an E4 run can never overwrite the E2
    run of the same component it is compared against.
    """
    if variant:
        return "{}_{}/{}__{}__seed{}".format(subtask.lower(), edition, component,
                                             variant, seed)
    return "{}_{}/{}__seed{}".format(subtask.lower(), edition, component, seed)


def load_matrix() -> dict[str, Any]:
    import yaml

    with open(MATRIX_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# The data-quality rules live in harness.py, not in the matrix, so a change to
# them moves every stored probability without moving `config_id`. A change to
# the contradiction rule once changed the DBO training data without any stored
# component noticing; `data_rules_id` closes that gap.
_DATA_RULE_FUNCTIONS = (
    "_drop_contradicting_labels",
    "_drop_within_edition_duplicates",
    "_drop_train_test_overlap",
    "_drop_cross_edition_overlap",
    "apply_data_quality_rules",
)


def compute_data_rules_id() -> str:
    """
    Identity of the data-quality rules that produced a stored matrix.

    Hashes the *code* of the functions listed above, not the file: hashing
    harness.py wholesale would invalidate the store on any unrelated edit, and
    a check which fires on everything stops being usable.
    Docstrings are stripped before hashing for the same reason, so rewording
    the rationale of a rule does not invalidate runs the rule still produces
    identically. Renaming or removing one of these functions is itself a change
    of the rules and raises rather than silently hashing less.
    """
    import ast
    import inspect

    from src import harness

    source = inspect.getsource(harness)
    tree = ast.parse(source)
    wanted = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name in _DATA_RULE_FUNCTIONS:
            body = node.body
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                body = body[1:]          # drop the docstring
            stripped = ast.FunctionDef(
                name=node.name, args=node.args, body=body or [ast.Pass()],
                decorator_list=[], returns=None, type_params=[],
            )
            wanted[node.name] = ast.unparse(ast.fix_missing_locations(stripped))

    missing = set(_DATA_RULE_FUNCTIONS) - set(wanted)
    if missing:
        raise RuntimeError(
            "data-quality rules not found in harness.py: {}. Renaming or removing "
            "a rule changes what every stored component was trained on; update "
            "_DATA_RULE_FUNCTIONS deliberately rather than letting the hash drift."
            .format(", ".join(sorted(missing)))
        )
    canonical = json.dumps(wanted, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _machine() -> str:
    """
    The machine that produced a stored run, in `env_pin`'s format.

    The store already carried two identities -- `config_id` for the matrix and
    `data_rules_id` for the code that shaped the training data -- and neither
    covers the hardware. The encoder and LLM components were produced on a
    rented Linux GPU machine while the classical components ran on a Darwin/arm64
    laptop, so E3 would align components from two architectures with nothing
    in the files saying so. A reported figure has to be attributable to the
    machine that produced it, and the sidecars are where Chapter 4's figures are
    read from, so the field belongs at the point of writing rather than at the point
    of reporting -- what is reported from it is a separate, later decision.
    """
    import platform

    return f"{platform.system()}/{platform.machine()}"


def compute_config_id(component: str, matrix: dict[str, Any] | None = None,
                      *, e4: dict[str, Any] | None = None) -> str:
    """
    Configuration identity of one component under the current e2_matrix.yaml.

    Without it, the store could hold runs from two matrix generations
    distinguishable only by a `created` timestamp, so E3 could consume mixed
    generations with no way to notice. The identity
    hashes exactly the parts of the matrix that determine a stored output:
    the component's own block (with an `inherits` parent resolved), its
    family's protocol block, the fold structure, and — for the LLM families —
    the verbaliser section, which decides what a stored probability *is*.
    Deliberately excluded: reporting-side settings (bootstrap_n, ci_level,
    reporting tiers), which can change without touching any stored matrix,
    and YAML comments, which never reach the parsed structure. So a v1.3 →
    v1.4 edit that only adds LLM blocks leaves every TML identity unchanged,
    which is the property that makes the check usable at all.
    """
    matrix = matrix if matrix is not None else load_matrix()
    block = dict(matrix["components"][component])
    parent = block.get("inherits")
    payload: dict[str, Any] = {
        "component": component,
        "block": block,
        "inherited_from": dict(matrix["components"][parent]) if parent else None,
    }
    family = block.get("family") or (payload["inherited_from"] or {}).get("family")
    proto_key = _FAMILY_PROTOCOL.get(family)
    payload["protocol"] = matrix.get(proto_key) if proto_key else None
    payload["cv"] = {
        "cv_folds_e2": matrix["protocol"].get("cv_folds_e2"),
        "cv_folds_e4a": matrix["protocol"].get("cv_folds_e4a"),
        "seed_base": matrix["protocol"].get("seed_base"),
        "aggregation": matrix["protocol"].get("aggregation"),
        "variance": matrix.get("variance", {}).get(family),
    }
    if family in ("llm_finetune", "llm_fewshot"):
        payload["verbalisers"] = matrix.get("verbalisers")
    # E4 cells (2026-09-12). An E4 run is the same component under a different
    # condition and a fixed configuration, so its identity is the E2 payload
    # PLUS what varies. The key is added only when `e4` is given, which is what
    # keeps every stored E2 identity byte-identical: the payload an E2 run hashes
    # has no "e4" key at all, not an "e4": null one.
    if e4 is not None:
        experiment = str(e4["experiment"]).lower()
        payload["e4"] = {
            "experiment": experiment,
            "condition": e4.get("condition", "none"),
            "fixed_params": e4.get("fixed_params"),
            "protocol": (matrix.get("e4_protocol") or {}).get(experiment),
            "applicability": matrix.get("imbalance_applicability"),
        }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


class ComponentStore:
    """Read/write access to the component store."""

    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------

    def _npz(self, key: str) -> Path:
        return self.root / (key + ".npz")

    def _json(self, key: str) -> Path:
        return self.root / (key + ".json")

    def exists(self, key: str) -> bool:
        return self._npz(key).exists() and self._json(key).exists()

    def list_runs(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            str(p.relative_to(self.root).with_suffix("")).replace("\\", "/")
            for p in self.root.rglob("*.npz")
        )

    # -- write ---------------------------------------------------------------

    def save(
        self,
        key: str,
        *,
        proba: np.ndarray,
        y_true,
        ids,
        classes,
        fold,
        meta: dict[str, Any],
    ) -> Path:
        """
        Persist one run: the pooled out-of-fold probabilities and everything
        needed to interpret them without re-reading the source data.

        Probabilities are stored as float32. The loss is ~1e-7 per entry, far
        below any effect the combination layer could resolve, and it halves
        the size of the stored artefact.
        """
        # `meta` is free-form, and `load_many` refuses a set whose entries
        # disagree on n_splits -- so an entry written WITHOUT one is unusable.
        # Fail at write time rather than months later at read time. `machine`
        # is not listed because save() stamps it itself.
        required = ("subtask", "edition", "component", "seed", "n_splits",
                    "config_id", "data_rules_id")
        missing = [k for k in required if meta.get(k) is None]
        if missing:
            raise ValueError(
                "ComponentStore.save({!r}): meta is missing {} -- every stored run "
                "must carry these, because load_many compares them across "
                "components before E3 is allowed to combine anything.".format(
                    key, ", ".join(missing)))

        path = self._npz(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        proba = np.asarray(proba, dtype=np.float32)
        classes = np.asarray([str(c) for c in classes], dtype=object)
        y_true = np.asarray([str(v) for v in y_true], dtype=object)

        np.savez_compressed(
            path,
            proba=proba,
            y_true=y_true.astype("U"),
            ids=np.asarray(ids, dtype=np.int64),
            classes=classes.astype("U"),
            fold=np.asarray(fold, dtype=np.int16),
        )

        sidecar = dict(meta)
        sidecar.update({
            "key": key,
            "n_items": int(proba.shape[0]),
            "n_classes": int(proba.shape[1]),
            "classes": [str(c) for c in classes],
            "npz": path.name,
            "machine": _machine(),
        })
        self._json(key).write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    # -- read ----------------------------------------------------------------

    def load(self, key: str) -> dict[str, Any]:
        """
        Load one run. Returns proba, y_true, ids, classes, fold and the
        metadata sidecar.

        The gold labels are stored as strings, because a store shared between
        a boolean subtask (C2A, VIO) and a four-class one (DBO) cannot keep a
        single dtype. Callers that need the original dtype should map through
        the classes list.
        """
        if not self.exists(key):
            raise FileNotFoundError(
                "No component-store entry {!r} under {}".format(key, self.root)
            )
        with np.load(self._npz(key), allow_pickle=False) as z:
            rec = {
                "proba": z["proba"].astype(np.float64),
                "y_true": z["y_true"].tolist(),
                "ids": z["ids"],
                "classes": z["classes"].tolist(),
                "fold": z["fold"],
            }
        rec["meta"] = json.loads(self._json(key).read_text(encoding="utf-8"))
        return rec

    def load_many(self, keys, *, check_generation: bool = True) -> dict[str, dict[str, Any]]:
        """
        Load several runs, keyed by run key. Used by E3 to align components.

        `check_generation` (default ON, so E3 cannot forget it) asserts that
        every loaded sidecar carries a `config_id` and a `data_rules_id`, and
        that both match the current identities: the configuration computed
        from e2_matrix.yaml for that component, and the code of the
        data-quality rules in harness.py. The second was added on 2026-08-18
        after a rule change altered the DBO training data while every stored
        config_id stayed put, which is the same failure one level down.
        A mismatch means the stored run was produced under a configuration the
        matrix no longer describes — the mixed-generation failure of audit
        item D — and raises rather than warns, because a combination-layer
        number built on mixed generations is wrong, not merely undocumented.
        Pass check_generation=False only for exploratory reading, never in a
        path that produces a reported number.
        """
        recs = {k: self.load(k) for k in keys}
        if check_generation:
            matrix = load_matrix()
            rules_now = compute_data_rules_id()
            current: dict[tuple, str] = {}
            problems = []
            for k, rec in recs.items():
                component = rec["meta"].get("component")
                stored = rec["meta"].get("config_id")
                # An E4 sidecar is checked against the E4 identity it was
                # written under; checked against the bare component it would
                # always mismatch, and the only way through would be
                # check_generation=False, which is the escape hatch this
                # function exists to keep out of reported numbers.
                e4 = rec["meta"].get("e4")
                ident = (component, json.dumps(e4, sort_keys=True))
                if ident not in current:
                    current[ident] = compute_config_id(component, matrix, e4=e4)
                if stored is None:
                    problems.append("{}: sidecar carries no config_id "
                                    "(older generation, re-stamp or re-run)".format(k))
                elif stored != current[ident]:
                    problems.append("{}: config_id {} != current {} "
                                    "(matrix drifted since this run)".format(
                                        k, stored, current[ident]))
                stored_rules = rec["meta"].get("data_rules_id")
                if stored_rules is None:
                    problems.append("{}: sidecar carries no data_rules_id "
                                    "(pre-2026-08-18 generation, re-run)".format(k))
                elif stored_rules != rules_now:
                    problems.append("{}: data_rules_id {} != current {} "
                                    "(the data-quality rules changed since this "
                                    "run)".format(k, stored_rules, rules_now))
            # config_id and data_rules_id alone let through two mismatches:
            # a 2-fold encoder stored beside 5-fold TML runs, and a Linux
            # entry beside Darwin ones. Neither is visible in the data --
            # pooled out-of-fold probabilities have the SAME SHAPE whatever the
            # fold count, so the entries look like peers. E3 would then soft-vote
            # a component trained on ~50% of the data per fold against ones
            # trained on ~80%, understating it for a reason that has nothing to
            # do with combination, and SQ2 is exactly the question "does
            # combining beat the best single component?".
            splits = {k: rec["meta"].get("n_splits") for k, rec in recs.items()}
            distinct = {v for v in splits.values() if v is not None}
            if len(distinct) > 1:
                problems.append(
                    "mixed fold counts across the loaded set: {} -- these are not "
                    "comparable, and pooled probabilities hide it because the shape "
                    "is identical either way".format(
                        ", ".join("{}={}".format(k, v) for k, v in sorted(splits.items()))))
            missing_splits = [k for k, v in splits.items() if v is None]
            if missing_splits:
                problems.append(
                    "sidecar carries no n_splits, so fold agreement cannot be "
                    "checked: {}".format(", ".join(sorted(missing_splits))))

            # A WARNING, NOT AN ERROR. Every TML entry is Darwin/arm64 and every
            # GPU entry Linux/x86_64, by design, so raising would block E3
            # outright. A fold-count mismatch is genuine INCOMPARABILITY:
            # probabilities from models trained on different amounts of data.
            # A machine difference is a PROVENANCE FACT, and the C2A baseline
            # reproduced bit-identically across Darwin, Windows and Linux.
            # The reader is owed the machine, not a refusal;
            # `e2_summary._warn_on_mixed_folds` treats it the same way.
            machines = {k: rec["meta"].get("machine") for k, rec in recs.items()}
            distinct_m = {v for v in machines.values() if v is not None}
            if len(distinct_m) > 1:
                warnings.warn(
                    "component store: this set spans several machines ({}). "
                    "Expected when the GPU components ran on another machine -- but every "
                    "cross-component claim built on it must say so.".format(
                        ", ".join("{}={}".format(k, v) for k, v in sorted(machines.items()))),
                    RuntimeWarning, stacklevel=2)

            if problems:
                raise RuntimeError(
                    "Component-store generation check failed:\n  "
                    + "\n  ".join(problems))
        return recs

    # -- optional model artefacts -------------------------------------------

    def model_path(self, key: str, fold: int) -> Path:
        return self.root / "models" / (key.replace("/", "__") + "__fold{}.joblib".format(fold))

    def save_model(self, key: str, fold: int, model: Any) -> Path:
        """
        Persist a fitted component. Off by default: for the classical models a
        refit costs seconds, while the artefacts are large enough to bloat the
        repository. Turned on for components whose refit is expensive.
        """
        import joblib

        path = self.model_path(key, fold)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path, compress=3)
        return path

    def load_model(self, key: str, fold: int) -> Any:
        import joblib

        return joblib.load(self.model_path(key, fold))
