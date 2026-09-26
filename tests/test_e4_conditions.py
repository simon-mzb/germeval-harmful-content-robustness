"""test_e4_conditions.py -- the E4a/E4b machinery, on CPU, before any pod pays for it.

WHY THIS EXISTS. Until 2026-09-12 no line of code implemented an imbalance
condition: the protocol named them and sized their runs, and every trainable component
ran exactly one objective. E4a is the thesis's Dimension 1 and the largest paid
arm left, so the plumbing is tested here rather than discovered on an A40.

WHAT IT GUARDS, in the order the defects would cost:
  * the E2 path is untouched: every stored E2 artefact still validates under
    the identity the live code computes, and `none` reproduces the model's own
    loss on both neural families;
  * each condition does what 3.3 will say it does -- balanced weights average
    to one, oversampling reaches the majority count and drops nothing, focal
    loss reduces to cross-entropy at gamma 0 -- and a condition that is silently
    ignored changes the loss, so it is caught;
  * an E4 cell cannot collide with the E2 cell it is compared against: its own
    config_id, store key, artefact name and checkpoint directory;
  * the pre-registration holds mechanically: no E4 run and no derivation while
    `e4_protocol.status` is proposed, a frozen configuration is required, and
    the modal rule breaks ties the way the matrix states and refuses the rest;
  * one real E4a cell end to end for TML and for the LLM (tiny random Llama, the
    real tokeniser), and one encoder fit under focal loss on ModernGBERT-134M.

Run: .venv/bin/python -m tests.test_e4_conditions
"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import e2_runner, e4_config, imbalance
from src import encoder_components as enc
from src import llm_components as llm
from src import tml_components as tml
from src.component_store import ComponentStore, compute_config_id, run_key
from src.e2_runner import load_pool, run_component
from src.harness import resolve_classes

_PASS, _FAIL = 0, 0


def ok(m):
    global _PASS
    _PASS += 1
    print("  \033[32mPASS\033[0m  " + m)


def bad(m):
    global _FAIL
    _FAIL += 1
    print("  \033[31mFAIL\033[0m  " + m)


def check(label, cond):
    (ok if cond else bad)(label)


def raises(label, fn, exc=Exception, needle=None):
    try:
        fn()
    except exc as e:
        if needle is None or needle in str(e):
            ok(label)
        else:
            bad("{} (raised, but without {!r}: {})".format(label, needle, e))
        return
    except SystemExit as e:
        (ok if exc is SystemExit else bad)("{} (SystemExit {})".format(label, e.code))
        return
    bad(label + " (did not raise)")


_REAL_LOAD_MATRIX = e2_runner.load_matrix


def confirmed_matrix():
    # Reads through the ORIGINAL loader: the tests patch e2_runner.load_matrix
    # with this very function, and a lookup through the module attribute would
    # recurse until the interpreter dies without a traceback.
    m = copy.deepcopy(_REAL_LOAD_MATRIX())
    m["e4_protocol"]["status"] = "confirmed (test)"
    return m


def stratified_head(subtask, edition, per_class):
    df = load_pool(subtask, edition)
    parts = []
    for c in sorted(set(df["label"].astype(str))):
        sub = df[df["label"].astype(str) == c]
        parts.append(sub.head(min(per_class, len(sub))))
    out = pd.concat(parts).sample(frac=1.0, random_state=0).reset_index(drop=True)
    return out


def imbalanced_head(subtask, edition, counts):
    """A slice with a DELIBERATE imbalance. A balanced slice makes every class
    weight 1.0 and every oversampling draw empty, so a condition that does
    nothing and a condition that works look identical -- which is how the first
    run of this suite passed a TML cell that could not have shown a difference."""
    df = load_pool(subtask, edition)
    parts = [df[df["label"].astype(str) == c].head(n) for c, n in counts.items()]
    return pd.concat(parts).sample(frac=1.0, random_state=0).reset_index(drop=True)


# ---------------------------------------------------------------------------

def units():
    print("\n--- the condition definitions")
    y = np.array([0] * 90 + [1] * 8 + [2] * 2)
    w = imbalance.balanced_weights(y, 4)
    check("balanced weights average exactly 1 over the items (scale kept)",
          abs(float(w[y].mean()) - 1.0) < 1e-12)
    check("a class absent from the slice gets weight 1.0, not 0", w[3] == 1.0)
    check("rarer class weighs more", w[2] > w[1] > w[0])

    idx = imbalance.oversample_indices(y, seed=7)
    counts = np.bincount(y[idx])
    check("oversampling brings every present class to the majority count",
          list(counts) == [90, 90, 90])
    check("the natural slice is a prefix: nothing dropped, order kept",
          list(idx[:len(y)]) == list(range(len(y))))
    check("oversampling is deterministic in its seed",
          np.array_equal(idx, imbalance.oversample_indices(y, seed=7))
          and not np.array_equal(idx, imbalance.oversample_indices(y, seed=8)))
    check("a balanced slice is returned unchanged",
          list(imbalance.oversample_indices(np.array([0, 1, 0, 1]), 0)) == [0, 1, 2, 3])

    import torch
    torch.manual_seed(0)
    logits = torch.randn(32, 4)
    target = torch.randint(0, 4, (32,))
    ce = torch.nn.functional.cross_entropy(logits, target)
    check("gamma 0, no weight == torch cross-entropy",
          torch.allclose(imbalance.classification_loss(logits, target), ce, atol=1e-6))
    check("focal loss (gamma 2) is below cross-entropy",
          float(imbalance.classification_loss(logits, target, gamma=2.0)) < float(ce))
    wt = torch.tensor([0.5, 1.0, 2.0, 4.0])
    manual = (torch.nn.functional.cross_entropy(logits, target, reduction="none")
              * wt[target]).mean()
    check("weighted loss is the plain mean of w_i * l_i, not torch's weighted mean",
          torch.allclose(imbalance.classification_loss(logits, target, weight=wt),
                         manual, atol=1e-6))

    m = e2_runner.load_matrix()
    check("focal gamma is read from the matrix (2.0)", imbalance.focal_gamma(m) == 2.0)
    raises("focal loss refused for the SVM (D4)",
           lambda: imbalance.check_applicable("focal_loss", "tml_svm", "tml", m),
           ValueError, "does not define")
    raises("an unknown condition refused",
           lambda: imbalance.check_applicable("smote", "encoder_1b", "encoder", m),
           ValueError, "unknown")
    for cond in ("class_weighting", "focal_loss", "random_oversampling"):
        imbalance.check_applicable(cond, "llm_llammlein", "llm_finetune", m)
    ok("all three conditions defined for the fine-tuned LLM")


def identity():
    print("\n--- identity: E2 untouched, E4 distinct")
    store = ComponentStore()
    from src.component_store import parse_run_key
    # E2 runs only: once E4a cells are
    # fetched, the store also holds 3-fold entries, and load_many rightly refuses
    # a set that mixes them with 5-fold E2 runs.
    keys = [k for k in store.list_runs() if parse_run_key(k)["variant"] is None]
    arts = sorted(Path("results/e2").glob("e2_*_2025_*.json"))
    # The nine TML artefacts predate the run-file `config_id` (K5, 2026-08-27);
    # their identity lives in the store sidecars, which load_many checks below.
    mismatched, carrying = [], 0
    for p in arts:
        a = json.loads(p.read_text(encoding="utf-8"))
        stored = a["meta"].get("config_id")
        if stored is None:
            continue
        carrying += 1
        if stored != compute_config_id(a["component"]):
            mismatched.append(p.name)
    check("all {} E2 artefacts that carry a run-file config_id still match the live "
          "code".format(carrying), carrying >= 9 and not mismatched)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store.load_many(keys)
        ok("load_many validates all {} stored E2 runs".format(len(keys)))
    except Exception as e:
        bad("load_many refuses the stored E2 runs: {}".format(e))

    base = compute_config_id("encoder_1b")
    ids = {c: compute_config_id("encoder_1b", e4={"experiment": "E4A", "condition": c,
                                                  "fixed_params": {"lora_r": 8, "lr": 1e-4}})
           for c in imbalance.CONDITIONS}
    check("four E4a conditions -> four distinct ids, none equal to E2",
          len(set(ids.values())) == 4 and base not in ids.values())
    check("the fixed configuration is part of the E4 identity",
          compute_config_id("encoder_1b", e4={"experiment": "E4A", "condition": "none",
                                              "fixed_params": {"lora_r": 16, "lr": 1e-4}})
          != ids["none"])
    m = e2_runner.load_matrix()
    mm = copy.deepcopy(m)
    mm["e4_protocol"]["status"] = "confirmed"
    check("flipping e4_protocol.status moves no E4 id",
          compute_config_id("encoder_1b", mm, e4={"experiment": "E4A", "condition": "none",
                                                  "fixed_params": {"lora_r": 8, "lr": 1e-4}})
          == ids["none"])
    mm["e4_protocol"]["e4a"]["conditions"]["focal_loss"]["gamma"] = 1.0
    check("editing an E4a condition moves the E4 id and NOT the E2 id",
          compute_config_id("encoder_1b", mm, e4={"experiment": "E4A", "condition": "focal_loss",
                                                  "fixed_params": {"lora_r": 8, "lr": 1e-4}})
          != ids["focal_loss"] and compute_config_id("encoder_1b", mm) == base)
    check("E2 store key format unchanged",
          run_key("dbo", "2025", "encoder_1b", 42) == "dbo_2025/encoder_1b__seed42")
    check("E4 store key cannot collide with E2",
          run_key("dbo", "2025", "encoder_1b", 42, variant="e4a-none")
          != run_key("dbo", "2025", "encoder_1b", 42))


def modal_rule():
    print("\n--- the frozen configuration rule")

    def fold(sel, scores):
        return {"selected": sel, "grid_scores": {json.dumps(k, sort_keys=True): v
                                                 for k, v in scores}}
    a, b = {"lr": 1e-4}, {"lr": 2e-4}
    r = e4_config.modal_configuration(
        [fold(a, [(a, .5), (b, .4)])] * 3 + [fold(b, [(a, .4), (b, .5)])] * 2)
    check("a clear mode wins (3 of 5)", r["params"] == a and not r["count_tie"])
    r = e4_config.modal_configuration(
        [fold(a, [(a, .50), (b, .49)]), fold(b, [(a, .40), (b, .60)])])
    check("a count tie is broken by the higher MEAN selection score",
          r["params"] == b and r["count_tie"])
    r8a, r16a = {"lora_r": 8, "lr": 2e-4}, {"lora_r": 16, "lr": 1e-4}
    r = e4_config.modal_configuration(
        [fold(r16a, [(r16a, .70), (r8a, .40)]), fold(r8a, [(r16a, .60), (r8a, .50)])])
    check("#82's tie-break first: the smaller lora_r wins a count tie even with the "
          "lower mean score", r["params"] == r8a and r["tie_broken_by_capacity"])
    raises("a tie surviving both is refused, not broken by list order",
           lambda: e4_config.modal_configuration(
               [fold(a, [(a, .5), (b, .5)]), fold(b, [(a, .5), (b, .5)])]),
           ValueError, "refuses")

    import contextlib
    import io
    real = _REAL_LOAD_MATRIX()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = e4_config.main([])
        rc_write = e4_config.main(["--write"])
    printed = buf.getvalue()
    if real["e4_protocol"]["status"].startswith("confirmed"):
        ok("e4_protocol is confirmed; the proposed-state refusals no longer apply")
    else:
        check("bare e4_config under a PROPOSED protocol derives nothing and exits 0",
              rc == 0 and "NOT DERIVED" in printed and "\nE4A:" not in printed)
        check("--write under a proposed protocol is refused", rc_write == 1)


def refusals(tmp: Path):
    print("\n--- the runner refuses what it must")

    def proposed_matrix():
        m = copy.deepcopy(_REAL_LOAD_MATRIX())
        m["e4_protocol"]["status"] = "proposed (test)"
        return m

    # Patched, not read off the real matrix: the real status is confirmed since
    # 2026-09-12, and a gate test that depends on the live status stops testing
    # the gate the moment the protocol is confirmed -- which is what happened.
    orig = e2_runner.load_matrix
    e2_runner.load_matrix = proposed_matrix
    try:
        raises("E4a refused while e4_protocol.status is proposed",
               lambda: run_component("dbo", "2025", "tml_svm", experiment="E4A",
                                     condition="class_weighting", n_splits=3,
                                     out_dir=tmp, store=ComponentStore(tmp / "s"),
                                     ckpt_root=None),
               RuntimeError, "status")
    finally:
        e2_runner.load_matrix = orig
    e2_runner.load_matrix = confirmed_matrix
    orig_path = e4_config.OUT_PATH
    e4_config.OUT_PATH = tmp / "absent.json"
    try:
        raises("E4a at the wrong fold count refused",
               lambda: run_component("dbo", "2025", "tml_svm", experiment="E4A",
                                     condition="class_weighting", n_splits=5,
                                     out_dir=tmp, ckpt_root=None),
               ValueError, "folds")
        raises("E4a with no frozen configuration refused",
               lambda: run_component("dbo", "2025", "tml_svm", experiment="E4A",
                                     condition="class_weighting", n_splits=3,
                                     out_dir=tmp, ckpt_root=None),
               FileNotFoundError, "FROZEN")
        raises("E4b with a condition refused",
               lambda: run_component("dbo", "2025", "encoder_1b", experiment="E4B",
                                     condition="focal_loss", n_splits=5,
                                     out_dir=tmp, ckpt_root=None),
               ValueError, "none")
    finally:
        e2_runner.load_matrix = orig
        e4_config.OUT_PATH = orig_path
    raises("E2 with a condition refused",
           lambda: run_component("dbo", "2025", "tml_svm", condition="focal_loss",
                                 out_dir=tmp, ckpt_root=None),
           ValueError, "E4a")
    raises("the few-shot component refuses any E4 condition",
           lambda: e2_runner.component_api("llm_qwen_fewshot", subtask="dbo",
                                           condition="class_weighting"),
           ValueError, "prompt_coverage")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = e2_runner.main(["--experiment", "e4a", "--subtask", "dbo", "--edition", "2026",
                             "--component", "llm_llammlein", "--condition", "none",
                             "class_weighting", "--dry-run"])
    listed = [l for l in buf.getvalue().splitlines() if l.startswith("  - ")]
    check("CLI takes several conditions, so a subtask can be split across pods",
          rc == 0 and len(listed) == 2 and listed[1].endswith("class_weighting"))
    raises("CLI refuses focal loss for the SVM before a dry run can list it",
           lambda: e2_runner.main(["--experiment", "e4a", "--subtask", "dbo",
                                   "--component", "tml_svm", "--condition",
                                   "focal_loss", "--dry-run"]),
           SystemExit)


def _freeze(tmp: Path, entries: dict) -> Path:
    path = tmp / "e4_fixed_configurations.json"
    payload = {"protocol_digest": e4_config.protocol_digest(confirmed_matrix()),
               "e4a": {}, "e4b": {}}
    for (component, cell), params in entries.items():
        payload["e4a"].setdefault(component, {})[cell] = {"params": params}
        payload["e4b"][component] = {"params": params}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def tml_cell(tmp: Path):
    print("\n--- a real E4a cell: tml_svm on a stratified DBO slice")
    svm_params = dict(tml.GRIDS["tml_svm"][0])
    df = imbalanced_head("dbo", "2025", {"nothing": 150, "criticism": 50,
                                         "agitation": 25, "subversive": 15})
    classes = resolve_classes(df["label"].values)

    fit_none = tml._fit_component("tml_svm", df, classes, seed=42, n_jobs=1,
                                  grid=[svm_params])
    fit_cw = tml._fit_component("tml_svm", df, classes, seed=42, n_jobs=1,
                                grid=[svm_params], condition="class_weighting")
    fit_os = tml._fit_component("tml_svm", df, classes, seed=42, n_jobs=1,
                                grid=[svm_params], condition="random_oversampling")
    check("class weighting changes the fitted SVM (not silently ignored)",
          not np.allclose(fit_none.estimator_.coef_, fit_cw.estimator_.coef_))
    check("oversampling trains on more rows than the natural fit slice",
          fit_os.record["n_fit_trained"] > fit_os.record["n_fit"]
          and fit_none.record["n_fit_trained"] == fit_none.record["n_fit"])
    check("the vectoriser is fitted on the natural slice under oversampling",
          fit_os.record["n_features"] == fit_none.record["n_features"])
    raises("TML refuses focal loss inside the component too",
           lambda: tml._fit_component("tml_svm", df, classes, seed=42, n_jobs=1,
                                      grid=[svm_params], condition="focal_loss"),
           ValueError, "the protocol defines")

    freeze = _freeze(tmp, {("tml_svm", "dbo_2025"): svm_params})
    store = ComponentStore(tmp / "store")
    orig_m, orig_p, orig_pool = e2_runner.load_matrix, e4_config.OUT_PATH, e2_runner.load_pool
    e2_runner.load_matrix = confirmed_matrix
    e4_config.OUT_PATH = freeze
    e2_runner.load_pool = lambda st, ed: df.copy()
    try:
        res = run_component("dbo", "2025", "tml_svm", experiment="E4A",
                            condition="random_oversampling", n_splits=3, n_jobs=1,
                            bootstrap_n=50, store=store, out_dir=tmp / "out",
                            ckpt_root=tmp / "ckpt")
    finally:
        e2_runner.load_matrix, e4_config.OUT_PATH, e2_runner.load_pool = orig_m, orig_p, orig_pool

    out = tmp / "out" / "e4a_dbo_2025_tml_svm__random_oversampling.json"
    check("artefact lands under its own E4a name", out.exists())
    check("experiment E4a, package B8, condition recorded",
          res["experiment"] == "E4a" and res["package"] == "B8"
          and res["meta"]["imbalance_condition"] == "random_oversampling")
    check("ONE seed and no D3 replicates (D5)",
          [r["seed"] for r in res["repeats"]] == [42] and res["seed_replicates"] == [])
    check("meta carries the one frozen cell, not the E2 grid",
          res["meta"]["grid"] == [svm_params] and res["meta"]["e4"]["fixed_params"] == svm_params)
    check("every fold trained on the oversampled slice",
          all(f["n_fit_trained"] > f["n_fit"] for f in res["repeats"][0]["folds_detail"]))
    key = run_key("dbo", "2025", "tml_svm", 42, variant="e4a-random_oversampling")
    check("store key carries the variant", store.exists(key))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store.load_many([key])
        ok("load_many validates the E4 sidecar under its E4 identity")
    except Exception as e:
        bad("load_many refuses a fresh E4 sidecar: {}".format(e))
    check("checkpoints were cleaned under the E4 stem",
          not any((tmp / "ckpt").glob("dbo_2025_tml_svm__e4a-random_oversampling_*")))
    check("the E2 artefact of the same component is untouched",
          Path("results/e2/e2_dbo_2025_tml_svm.json").exists())


def llm_losses_and_cell(tmp: Path):
    print("\n--- LLM: the loss paths, then a real E4a cell (tiny Llama, real tokeniser)")
    import torch
    from tests.test_llm_cell import COMPONENT, tiny_model_dir, tiny_pool

    original = llm.MODEL_IDS[COMPONENT]
    llm.MODEL_IDS[COMPONENT] = str(tiny_model_dir(tmp))
    try:
        tok = llm._prepare_tokeniser(llm.MODEL_IDS[COMPONENT])
        df = tiny_pool(per_class=12)
        classes = resolve_classes(df["label"].values)
        label_ids = llm._label_sequences(tok, "dbo", classes)
        prompts = [tok("Beispieltext Nummer {}".format(i))["input_ids"] for i in range(8)]
        y = np.array([0, 1, 2, 3, 0, 0, 1, 3])
        gold = [label_ids[i] for i in y]
        batch = next(llm._training_batches(prompts, gold, tok.pad_token_id, 8, 0,
                                           shuffle=False, item_class=y))
        model = llm._build_model(COMPONENT, {"lr": 1e-3}, 0, "cpu")
        model.eval()
        with torch.no_grad():
            hf = model(**{k: v for k, v in batch.items() if k != "item_class"}).loss
            none = llm._batch_loss(model, dict(batch), "none", None, 0.0)
            ours = imbalance.token_loss(
                model(input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"]).logits, batch["labels"])
            cw = torch.tensor(imbalance.balanced_weights(y, 4), dtype=torch.float32)
            weighted = llm._batch_loss(model, dict(batch), "class_weighting", cw, 0.0)
            focal = llm._batch_loss(model, dict(batch), "focal_loss", None, 2.0)
        check("token_loss reproduces HuggingFace's masked loss exactly (none path)",
              torch.allclose(hf, ours, atol=1e-5) and torch.allclose(hf, none, atol=1e-6))
        check("class weighting changes the LLM loss (not silently ignored)",
              not torch.allclose(weighted, none, atol=1e-6))
        check("focal loss is below the plain loss", float(focal) < float(none))
        plain = next(llm._training_batches(prompts, gold, tok.pad_token_id, 8, 0, shuffle=False))
        check("item_class rides along without changing the batch tensors",
              all(torch.equal(plain[k], batch[k]) for k in plain))

        fixed = {"lr": 1e-3}
        freeze = _freeze(tmp, {(COMPONENT, "dbo_2025"): fixed})
        store = ComponentStore(tmp / "llm_store")
        orig_m, orig_p, orig_pool = e2_runner.load_matrix, e4_config.OUT_PATH, e2_runner.load_pool
        e2_runner.load_matrix = confirmed_matrix
        e4_config.OUT_PATH = freeze
        e2_runner.load_pool = lambda st, ed: df.copy()
        try:
            res = run_component("dbo", "2025", COMPONENT, experiment="E4A",
                                condition="class_weighting", n_splits=3,
                                device="cpu", max_epochs=1, bootstrap_n=50,
                                store=store, out_dir=tmp / "llm_out",
                                ckpt_root=tmp / "llm_ckpt")
        finally:
            e2_runner.load_matrix, e4_config.OUT_PATH, e2_runner.load_pool = orig_m, orig_p, orig_pool
        folds = res["repeats"][0]["folds_detail"]
        # TML has no seed replicate in E2 either, so only a NEURAL cell can show
        # that E4 skips them -- a mutation that restored them survived the TML
        # check alone.
        check("LLM E4a cell: primary seed only, and NO D3 replicates (D5)",
              [r["seed"] for r in res["repeats"]] == [42] and res["seed_replicates"] == [])
        check("LLM E4a cell ran 3 folds under class weighting with recorded weights",
              len(folds) == 3 and all(f["imbalance_condition"] == "class_weighting"
                                      and f["class_weights"] for f in folds))
        check("LLM E4a artefact named for its condition",
              (tmp / "llm_out" / "e4a_dbo_2025_{}__class_weighting.json".format(COMPONENT)).exists())
    finally:
        llm.MODEL_IDS[COMPONENT] = original


def encoder_loss_and_fit():
    print("\n--- encoder: the loss path on ModernGBERT-134M, then one focal-loss fit")
    import torch
    from transformers import AutoTokenizer

    comp = "encoder_134m"
    tok = AutoTokenizer.from_pretrained(enc.MODEL_IDS[comp])
    model = enc._build_model(comp, {"lora_r": 8, "lr": 1e-4}, 4, 0)
    model.eval()
    enc_ = tok(["erster Text", "zweiter, etwas längerer Text", "drei", "vier vier"],
               padding=True, return_tensors="pt")
    labels = torch.tensor([0, 1, 2, 3])
    batch = {"input_ids": enc_["input_ids"], "attention_mask": enc_["attention_mask"],
             "labels": labels}
    with torch.no_grad():
        hf = enc._batch_loss(model, batch, "none", None, 0.0)
        ours = imbalance.classification_loss(
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits,
            labels)
        cw = torch.tensor([0.25, 1.0, 2.0, 4.0])
        weighted = enc._batch_loss(model, batch, "class_weighting", cw, 0.0)
    check("encoder: classification_loss reproduces the model's own loss (none path)",
          torch.allclose(hf, ours, atol=1e-5))
    check("encoder: class weighting changes the loss (not silently ignored)",
          not torch.allclose(weighted, hf, atol=1e-6))

    df = stratified_head("dbo", "2025", 10)
    df["label"] = df["label"].astype(str)
    classes = resolve_classes(df["label"].values)
    fitted = enc._fit_component(comp, df, classes, seed=42, device="cpu", max_epochs=1,
                                grid=[{"lora_r": 8, "lr": 1e-4}], condition="focal_loss")
    check("encoder fit under focal loss records gamma 2.0 and its condition",
          fitted.record["imbalance_condition"] == "focal_loss"
          and fitted.record["focal_gamma"] == 2.0
          and fitted.record["n_fit_trained"] == fitted.record["n_fit"])


def e4c_cells(tmp: Path):
    print("\n--- E4c: the 2026 pool, the VIO mapping, and cells end to end")
    import hashlib
    from src import e4c_runner

    protocol = e4c_runner.e4c_protocol(_REAL_LOAD_MATRIX())
    y = e4c_runner.map_labels("vio", ["nothing", "call2violence", "other", "glorification"],
                              [False, True], protocol)
    check("VIO mapping collapses every positive 2026 class against nothing (3.1)",
          list(y) == [False, True, True, True])
    # 2026-09-14: map_labels returned an OBJECT array of bools, which scikit-learn
    # types as "unknown", so f1_score raised on c2a and vio after the fit. The dbo
    # cell below never saw it because dbo's labels are strings. Mutation: restore
    # dtype=object in map_labels and both checks turn red.
    from sklearn.metrics import f1_score
    from sklearn.utils.multiclass import type_of_target
    y_c2a = e4c_runner.map_labels("c2a", [True, False, True], [False, True], protocol)
    check("mapped binary labels are a binary target, not 'unknown' (vio, c2a)",
          type_of_target(y) == "binary" and type_of_target(y_c2a) == "binary")
    try:
        f1_score(y_c2a, np.array([True, False, False]), labels=[False, True], average="macro")
        ok("macro-F1 accepts the mapped binary labels")
    except ValueError as exc:
        bad("macro-F1 accepts the mapped binary labels ({})".format(exc))
    raises("an unmapped VIO label is refused",
           lambda: e4c_runner.map_labels("vio", ["nothing", "praise"], [False, True], protocol),
           ValueError, "no declared mapping")
    raises("a DBO label outside the 2025 set is refused",
           lambda: e4c_runner.map_labels("dbo", ["nothing", "insult"],
                                         ["agitation", "criticism", "nothing", "subversive"],
                                         protocol),
           ValueError, "outside")
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
    det = e4c_runner.source_class_detection(["nothing", "support", "support", "other"],
                                            proba, [False, True])
    check("per-source-class detection needs no mapping for the positive classes",
          det["support"]["share_flagged_positive"] == 1.0
          and det["other"]["share_flagged_positive"] == 0.0 and det["nothing"]["n"] == 1)

    # A small pool file in the frozen format, for a real scoring pass on the laptop.
    real_pool, _ = e4c_runner.eval_pool("dbo")
    sub = pd.concat([real_pool[real_pool["label"] == c].head(n) for c, n in
                     {"nothing": 60, "criticism": 20, "agitation": 10, "subversive": 5}.items()])
    ids = [str(i) for i in sub["id"].tolist()]
    entry = {"subtask": "dbo", "n_pool": len(ids), "ids": ids,
             "ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()}
    small = tmp / "pool_small.json"
    small.write_text(json.dumps({"pools": [entry]}), encoding="utf-8")
    bad = tmp / "pool_bad.json"
    bad.write_text(json.dumps({"pools": [dict(entry, ids_sha256="0" * 64)]}), encoding="utf-8")
    raises("a 2026 pool that does not match its frozen hash is refused",
           lambda: e4c_runner.eval_pool("dbo", bad), ValueError, "does not match")

    train = imbalanced_head("dbo", "2025", {"nothing": 120, "criticism": 40,
                                            "agitation": 20, "subversive": 12})
    svm = dict(tml.GRIDS["tml_svm"][0])
    freeze = _freeze(tmp, {("tml_svm", "dbo_2025"): svm})
    store = ComponentStore(tmp / "e4c_store")
    orig_pool, orig_p = e4c_runner.load_pool, e4_config.OUT_PATH
    e4c_runner.load_pool = lambda st, ed: train.copy()
    e4_config.OUT_PATH = freeze
    try:
        res = e4c_runner.run_cell("tml_svm", "dbo", bootstrap_n=50, store=store,
                                  out_dir=tmp / "e4c_out", adapter_dir=tmp / "adapters",
                                  pool_path=small)
    finally:
        e4c_runner.load_pool, e4_config.OUT_PATH = orig_pool, orig_p
    check("E4c TML cell: scored on exactly the frozen pool, artefact under its own name",
          res["meta"]["n_eval"] == len(ids)
          and (tmp / "e4c_out" / "e4c_dbo_tml_svm.json").exists())
    check("E4c cell trained at the frozen configuration, one fit, no adapter for TML",
          res["fit_record"]["selected_params"] == svm and res["adapter"] is None)
    key = run_key("dbo", "2026", "tml_svm", 42, variant="e4c")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = store.load_many([key])[key]
        check("E4c store entry validates under its E4c identity and is keyed to 2026",
              rec["meta"]["edition"] == "2026" and len(rec["ids"]) == len(ids))
    except Exception as e:
        bad("load_many refuses the E4c sidecar: {}".format(e))

    # 2026-09-15: `_save_adapter` wrote the whole Qwen base
    # model for the few-shot LLM (29.5 GB per subtask), and it ran BEFORE the store
    # entry and the artefact, so a full disk during that save destroyed the paid
    # vio scoring. Mutations: drop the llm_fewshot guard, or move the save back
    # above `store.save`, and a check below turns red.
    class _Weights:
        def save_pretrained(self, path):
            (Path(path) / "model.safetensors").write_bytes(b"base weights")

    class _Fitted:
        model_, temperature_, classes_ = _Weights(), 1.3, ["a", "b"]

    fs = e4c_runner._save_adapter(_Fitted(), "llm_qwen_fewshot", "dbo", tmp / "fs_adapters")
    ft = e4c_runner._save_adapter(_Fitted(), "llm_llammlein", "dbo", tmp / "ft_adapters")
    check("E4c few-shot keeps its temperature and never copies the base model (h14)",
          fs is not None and set(fs["sha256"]) == {"e4c_calibration.json"}
          and not (tmp / "fs_adapters" / "dbo_llm_qwen_fewshot" / "model.safetensors").exists()
          and "model.safetensors" in ft["sha256"])

    def _disk_full(*a, **k):
        raise OSError("Disk quota exceeded (os error 122)")

    order_store = ComponentStore(tmp / "e4c_order_store")
    orig_save = e4c_runner._save_adapter
    e4c_runner._save_adapter = _disk_full
    e4c_runner.load_pool = lambda st, ed: train.copy()
    e4_config.OUT_PATH = freeze
    try:
        try:
            e4c_runner.run_cell("tml_svm", "dbo", bootstrap_n=50, store=order_store,
                                out_dir=tmp / "e4c_order_out", adapter_dir=tmp / "order_adapters",
                                pool_path=small)
            bad("a failing adapter save is not swallowed")
        except OSError:
            ok("a failing adapter save is not swallowed")
        art = tmp / "e4c_order_out" / "e4c_dbo_tml_svm.json"
        check("a failing adapter save leaves the store entry and the artefact on disk (h14)",
              order_store.exists(key) and art.exists()
              and json.loads(art.read_text(encoding="utf-8"))["adapter"] == {"status": "pending"})
        e4c_runner._save_adapter = orig_save
        again = e4c_runner.run_cell("tml_svm", "dbo", bootstrap_n=50, store=order_store,
                                    out_dir=tmp / "e4c_order_out", adapter_dir=tmp / "order_adapters",
                                    pool_path=small)
        check("the retry after a failed adapter save skips the cell instead of re-scoring it",
              again["adapter"] == {"status": "pending"})
    finally:
        e4c_runner._save_adapter = orig_save
        e4c_runner.load_pool, e4_config.OUT_PATH = orig_pool, orig_p

    from tests.test_llm_cell import COMPONENT, tiny_model_dir
    original_id, original_epochs = llm.MODEL_IDS[COMPONENT], llm.MAX_EPOCHS
    llm.MODEL_IDS[COMPONENT] = str(tiny_model_dir(tmp))
    llm.MAX_EPOCHS = 1
    freeze = _freeze(tmp, {(COMPONENT, "dbo_2025"): {"lr": 1e-3}})
    e4c_runner.load_pool = lambda st, ed: train.assign(label=train["label"].astype(str)).copy()
    e4_config.OUT_PATH = freeze
    try:
        res = e4c_runner.run_cell(COMPONENT, "dbo", device="cpu", bootstrap_n=50,
                                  store=ComponentStore(tmp / "e4c_llm_store"),
                                  out_dir=tmp / "e4c_llm_out", adapter_dir=tmp / "llm_adapters",
                                  pool_path=small)
        adapter = res["adapter"]
        check("E4c LLM cell saves its LoRA adapter with the temperature beside it",
              adapter is not None and "e4c_calibration.json" in adapter["sha256"]
              and any(n.startswith("adapter_") for n in adapter["sha256"]))
    finally:
        e4c_runner.load_pool, e4_config.OUT_PATH = orig_pool, orig_p
        llm.MODEL_IDS[COMPONENT], llm.MAX_EPOCHS = original_id, original_epochs


def store_isolation(tmp: Path):
    print("\n--- E4 entries never enter an E2 reader (found 2026-09-13)")
    from src import combination_headroom as ch
    from src import significance as sig
    from src.component_store import parse_run_key

    check("parse_run_key: an E2 key has no variant",
          parse_run_key("dbo_2025/encoder_1b__seed42")
          == {"cell": "dbo_2025", "component": "encoder_1b", "variant": None, "seed": 42})
    check("parse_run_key: an E4 key carries its variant",
          parse_run_key("dbo_2025/encoder_1b__e4a-none__seed42")["variant"] == "e4a-none")
    raises("parse_run_key refuses a name that is not a store key",
           lambda: parse_run_key("dbo_2025/encoder_1b"), ValueError, "not a component-store key")

    # A store holding two E2 runs and an E4 COPY of one of them, named as the
    # runner names it. The copy is the dangerous case: same items, same classes,
    # so nothing but the key tells it apart.
    src_dir = Path("results/component_store/dbo_2025")
    root = tmp / "iso_store"
    (root / "dbo_2025").mkdir(parents=True)
    for comp in ("tml_svm", "encoder_1b"):
        for ext in ("npz", "json"):
            shutil.copy(src_dir / "{}__seed42.{}".format(comp, ext),
                        root / "dbo_2025" / "{}__seed42.{}".format(comp, ext))
    for ext in ("npz", "json"):
        shutil.copy(src_dir / "encoder_1b__seed42.{}".format(ext),
                    root / "dbo_2025" / "encoder_1b__e4a-none__seed42.{}".format(ext))

    cells = sig.compare_store(ComponentStore(root), n_bootstrap=50)
    pairs = cells.get("dbo_2025", [])
    check("significance pairs E2 runs only (1 pair, no E4 key in it)",
          len(pairs) == 1 and all("e4a" not in p["a"] and "e4a" not in p["b"] for p in pairs))
    orig = ch.STORE
    ch.STORE = root
    try:
        comps = ch._load_cell("dbo")[0]
    finally:
        ch.STORE = orig
    check("combination_headroom loads E2 runs only (no second encoder)",
          sorted(comps) == ["encoder_1b", "tml_svm"])


def main() -> int:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    print("\n=== E4 conditions and cells, on CPU ===")
    tmp = Path(tempfile.mkdtemp(prefix="e4_test_"))
    try:
        units()
        identity()
        modal_rule()
        refusals(tmp)
        tml_cell(tmp)
        llm_losses_and_cell(tmp)
        encoder_loss_and_fit()
        e4c_cells(tmp)
        store_isolation(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n  {} passed, {} failed".format(_PASS, _FAIL))
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
