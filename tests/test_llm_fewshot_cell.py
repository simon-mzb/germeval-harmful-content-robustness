"""test_llm_fewshot_cell.py -- the in-context half of run_component, on CPU.

WHY THIS EXISTS. The same argument `test_llm_cell.py` makes, one component
along: a GPU run buys a 14B model on a rented card to find out what a
few-shot row costs and how well it calibrates, not to discover that a draw
returned eleven examples or that a prompt lost its instruction to truncation.
It should not reach a GPU untested when the whole path runs on CPU in
seconds.

HOW IT RUNS WITHOUT A 29 GB DOWNLOAD. Identical trick to test_llm_cell: a tiny
randomly initialised LlamaForCausalLM built from a config, with the REAL
LLaeMmlein tokeniser (already cached, tokeniser only). `MODEL_IDS` is pointed at
it. Random weights mean nothing about accuracy is asserted and nothing could be;
what is asserted is the machinery.

WHAT IT GUARDS, in the order the defects would bite:
  * `prompt_coverage` really does hold the example COUNT constant while varying
    composition -- if it did not, E4a's only few-shot cell would measure prompt
    budget instead of coverage, and no artefact would say so;
  * `none` really is proportional, i.e. it does NOT quietly guarantee the rare
    class, because the reference condition is the whole contrast;
  * the examples come from the fold TRAINING portion and nothing else -- the
    property `example_source` states and the one that would be a silent leak;
  * the in-context block does not eat the instruction: the item text is what
    truncation cuts, exactly as in the zero-shot arm, and both prompts share
    ONE implementation of that rule;
  * the probability is a proper distribution over the declared class order and
    the seed moves the DRAW rather than any weight;
  * run_component drives the family end to end with the replicate shape.

Run: .venv/bin/python -m tests.test_llm_fewshot_cell
"""
from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import e2_runner, llm_components as llm, llm_fewshot as fs
from src.component_store import ComponentStore, compute_config_id
from src.e2_runner import load_pool, run_component
from src.harness import make_cv_splits, resolve_classes

_PASS, _FAIL = 0, 0

COMPONENT = "llm_qwen_fewshot"
SUBTASK, EDITION = "dbo", "2025"    # four classes, and the one where the rare
                                    # class makes the imbalance contrast visible
N_SPLITS = 2
N_EXAMPLES = 8                      # smaller than the declared 12 so a tiny
                                    # pool can still be drawn from; the COUNT
                                    # invariant is what is asserted, not the 12


def ok(m):
    global _PASS
    _PASS += 1
    print("  \033[32mPASS\033[0m  " + m)


def bad(m):
    global _FAIL
    _FAIL += 1
    print("  \033[31mFAIL\033[0m  " + m)


def tiny_model_dir(tmp: Path) -> Path:
    """~1 M parameters, the REAL tokeniser. See test_llm_cell for the reasoning."""
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(llm.MODEL_IDS["llm_llammlein"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=4096, pad_token_id=tok.pad_token_id,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    out = tmp / "tiny_llama"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def tiny_pool(per_class=16):
    """Hand-stratified: `subversive` is 0.8 % of DBO and a random slice drops it."""
    df = load_pool(SUBTASK, EDITION)
    parts = []
    for c in sorted(set(df["label"].astype(str))):
        sub = df[df["label"].astype(str) == c]
        parts.append(sub.head(min(per_class, len(sub))))
    out = pd.concat(parts).sample(frac=1.0, random_state=0).reset_index(drop=True)
    out["label"] = out["label"].astype(str)
    return out


def check_one_resident_model(tmp: Path, df, classes) -> None:
    """A CELL LOADS THE MODEL ONCE, not once per fold.

    This is the check that was missing when the first real B6 cell died. The
    few-shot arm touches no weight, so `_fit_component` used to build a fresh
    copy for every fold and every seed replicate -- seven for a cell. On CPU with a
    ~1 M-parameter Llama that is free, which is exactly why every suite passed;
    on the A40 the second 34.4 GB copy hit "44.09 GiB is allocated by PyTorch"
    inside `Module._apply` and took the run down after fold 0.

    Counting CONSTRUCTIONS rather than measuring memory is what makes this
    meaningful on a laptop: the defect is arithmetic on the number of copies, and
    that number is machine-independent.
    """
    import transformers

    real = transformers.AutoModelForCausalLM.from_pretrained
    calls = {"n": 0}

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    fs._evict_resident()
    transformers.AutoModelForCausalLM.from_pretrained = counting
    try:
        fitted = [
            fs._fit_component(COMPONENT, df, classes, subtask=SUBTASK,
                              seed=s, device="cpu", n_examples=N_EXAMPLES)
            for s in (42, 43, 44)
        ]
        assert calls["n"] == 1, (
            "three fits built {} models. A cell is 5 folds + 2 replicates, so "
            "this is seven resident copies of a 29.5 GB model on a 44.4 GB "
            "card".format(calls["n"]))
        assert fitted[0].model_ is fitted[1].model_ is fitted[2].model_, (
            "the fits hold different model objects, so the old copies stay "
            "reachable and no allocator can reclaim them")
        ok("a cell loads the few-shot model ONCE across three fits, and all "
           "three share the one resident copy")

        # A different model ID must evict rather than hand back the stale one --
        # `--arm llm` runs two components in one process (27.0 + 29.5 GB), and
        # the CPU suites swap MODEL_IDS to a tiny directory.
        first = fitted[0].model_
        second_dir = tmp / "tiny_llama_2"
        shutil.copytree(fs.MODEL_IDS[COMPONENT], second_dir)
        swapped = fs.MODEL_IDS[COMPONENT]
        fs.MODEL_IDS[COMPONENT] = str(second_dir)
        try:
            other = fs._fit_component(COMPONENT, df, classes, subtask=SUBTASK,
                                      seed=42, device="cpu",
                                      n_examples=N_EXAMPLES)
            assert calls["n"] == 2, (
                "a different model ID did not reload: the slot is keyed on "
                "something coarser than the resolved ID, so a suite that points "
                "MODEL_IDS at a tiny model would silently score the wrong one")
            assert other.model_ is not first, (
                "the resident slot handed back the previous model for a "
                "different ID")
            ok("a different model ID evicts and reloads, so the slot never "
               "holds two models or the wrong one")
        finally:
            fs.MODEL_IDS[COMPONENT] = swapped
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = real
        fs._evict_resident()


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    print("\n=== the few-shot half of run_component, on CPU (tiny random Llama, "
          "real tokeniser) ===")
    original_id = fs.MODEL_IDS[COMPONENT]
    original_n = fs.N_EXAMPLES
    original_pool = e2_runner.load_pool
    try:
        fs.MODEL_IDS[COMPONENT] = str(tiny_model_dir(tmp))
        df = tiny_pool()
        classes = resolve_classes(df["label"].values)
        tok = llm._prepare_tokeniser(fs.MODEL_IDS[COMPONENT])
        label_ids = llm._label_sequences(tok, SUBTASK, classes)

        # --- one resident model per cell (the B6 OOM, 2026-09-04) ---------
        try:
            check_one_resident_model(tmp, df, classes)
        except AssertionError as e:
            bad("resident model: {}".format(e))
        except Exception as e:
            bad("resident model raised {}: {}".format(type(e).__name__, e))

        # --- the imbalance contrast: composition varies, budget does not ---------
        try:
            assert len(classes) == 4, "the slice lost a class: {}".format(classes)
            ref = fs.select_examples(df, classes, n_examples=N_EXAMPLES,
                                     strategy="none", seed=42)
            cov = fs.select_examples(df, classes, n_examples=N_EXAMPLES,
                                     strategy="prompt_coverage", seed=42)
            assert len(ref) == len(cov) == N_EXAMPLES, (
                "the two D4 cells drew {} and {} examples -- the contrast would "
                "measure prompt BUDGET, not coverage".format(len(ref), len(cov)))
            cov_counts = cov["label"].astype(str).value_counts().to_dict()
            assert len(cov_counts) == len(classes), (
                "prompt_coverage missed a class that the pool has: {}".format(
                    cov_counts))
            assert max(cov_counts.values()) - min(cov_counts.values()) <= 1, (
                "prompt_coverage is not equal-per-class: {}".format(cov_counts))
            ok("prompt_coverage covers all {} classes at the identical example "
               "count ({} vs {})".format(len(classes), len(ref), len(cov)))
        except AssertionError as e:
            bad("D4 contrast: {}".format(e))
        except Exception as e:
            bad("D4 contrast raised {}: {}".format(type(e).__name__, e))

        # --- the reference cell must NOT guarantee the rare class ---------
        # If `none` reliably contained a subversive example, both E4a cells would
        # be the same prompt and E4a's few-shot row would measure nothing. The
        # real pool is 0.8 % subversive; this hand-stratified slice is far
        # richer, so the assertion is the weaker, honest one: the draw follows
        # the frame it is given rather than repairing it.
        try:
            skewed = pd.concat([
                df[df["label"] == "nothing"],
                df[df["label"] == "subversive"].head(1),
            ]).reset_index(drop=True)
            seen = set()
            for s in range(30):
                drawn = fs.select_examples(skewed, classes, n_examples=4,
                                           strategy="none", seed=s)
                seen.update(drawn["label"].astype(str))
            share = sum(1 for s in range(30)
                        if "subversive" in set(
                            fs.select_examples(skewed, classes, n_examples=4,
                                               strategy="none", seed=s
                                               )["label"].astype(str)))
            assert share < 30, (
                "every one of 30 proportional draws contained the rare class; "
                "the reference cell is repairing the imbalance it exists to "
                "represent")
            ok("the `none` draw follows the natural distribution -- the rare "
               "class appeared in {}/30 draws, not 30/30".format(share))
        except AssertionError as e:
            bad("reference cell: {}".format(e))
        except Exception as e:
            bad("reference cell raised {}: {}".format(type(e).__name__, e))

        # --- the examples come from the training portion, and only that ----
        try:
            splits = list(make_cv_splits(df, n_splits=N_SPLITS, seed=42))
            tr_idx, va_idx = splits[0]
            train_df = df.iloc[tr_idx].reset_index(drop=True)
            val_texts = set(df.iloc[va_idx]["description"].astype(str))
            drawn = fs.select_examples(train_df, classes, n_examples=N_EXAMPLES,
                                       strategy="prompt_coverage", seed=42)
            leaked = val_texts & set(drawn["description"].astype(str))
            assert not leaked, (
                "{} in-context example(s) came from the fold's VALIDATION part "
                "-- llm_fewshot_protocol.example_source forbids it and no "
                "artefact would have shown it".format(len(leaked)))
            ok("no in-context example is drawn from the fold's validation part")
        except AssertionError as e:
            bad("example source: {}".format(e))
        except Exception as e:
            bad("example source raised {}: {}".format(type(e).__name__, e))

        # --- the prompt keeps its instruction; the ITEM text is what is cut --
        try:
            ex = fs.select_examples(df, classes, n_examples=4, strategy="none",
                                    seed=42)
            render = lambda body: fs.build_fewshot_prompt(
                body, SUBTASK, classes, ex, definitions=False)
            p = render("ein Beispieltext")
            verb = llm.verbalisers_for(SUBTASK)
            for c in classes:
                assert verb[str(c)] in p and str(c) in p, (
                    "the label block lost {}".format(c))
            assert llm._INSTRUCTION[SUBTASK] in p, "the instruction is missing"
            assert p.count("Beitrag:") == len(ex) + 1, (
                "expected {} example turns plus the item, found {}".format(
                    len(ex), p.count("Beitrag:") - 1))
            assert p.rstrip().endswith("Antwort:"), "the prompt does not end open"

            long_text = "sehr langer Text " * 4000
            ids, stats = llm._encode_prompts(
                tok, [long_text], SUBTASK, classes, 512,
                max(len(x) for x in label_ids), render=render)
            assert stats["text_truncated"] == 1, "the over-long item was not cut"
            assert stats["prompt_tokens_max"] <= 512, (
                "the assembled prompt is {} tokens against a 512 budget -- the "
                "decode/re-tokenise round trip grew it".format(
                    stats["prompt_tokens_max"]))
            cut = tok.decode(ids[0])
            assert llm._INSTRUCTION[SUBTASK][:20] in cut, (
                "truncation ate the instruction: the budget is spent on the "
                "item text, never on the template")
            ok("the few-shot prompt carries instruction, label block and {} "
               "example turns, and truncation cuts the item text alone "
               "({} template tokens)".format(len(ex), stats["template_overhead_tokens"]))
        except AssertionError as e:
            bad("prompt: {}".format(e))
        except Exception as e:
            bad("prompt raised {}: {}".format(type(e).__name__, e))

        # --- the seed moves the DRAW, and nothing else --------------------
        try:
            a = fs.select_examples(df, classes, n_examples=N_EXAMPLES,
                                   strategy="none", seed=42)
            b = fs.select_examples(df, classes, n_examples=N_EXAMPLES,
                                   strategy="none", seed=43)
            a2 = fs.select_examples(df, classes, n_examples=N_EXAMPLES,
                                    strategy="none", seed=42)
            assert list(a["description"]) == list(a2["description"]), \
                "the same seed drew different examples"
            assert list(a["description"]) != list(b["description"]), (
                "seeds 42 and 43 drew the identical set, so `draws: 3` would "
                "report zero variance by construction")
            ok("the draw is deterministic in the seed and moves with it -- "
               "which is what `draws: {}` measures".format(fs.DRAWS))
        except AssertionError as e:
            bad("draw variance: {}".format(e))
        except Exception as e:
            bad("draw variance raised {}: {}".format(type(e).__name__, e))

        # --- run_component end to end -------------------------------------
        res = None
        try:
            e2_runner.load_pool = lambda st, ed: df
            fs.N_EXAMPLES = N_EXAMPLES
            store = ComponentStore(root=tmp / "store")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = run_component(SUBTASK, EDITION, COMPONENT,
                                    n_splits=N_SPLITS, device="cpu",
                                    bootstrap_n=20, store=store,
                                    out_dir=tmp / "out", ckpt_root=tmp / "ck")
            assert res["meta"]["family"] == "llm_fewshot", res["meta"]["family"]
            assert res["package"] == "B6", res["package"]
            assert res["meta"]["config_id"] == compute_config_id(COMPONENT)
            assert len(res["repeats"]) == 1 and res["repeats"][0]["seed"] == 42
            assert [r["model_seed"] for r in res["seed_replicates"]] == [43, 44]
            # A replicate must carry the diagnostics that explain what it COST,
            # not only what it scored. B6 cell 1 produced two
            # replicates of identical work at 1215 s and 4371 s and the artefact
            # could not say why: `tokenisation_predict` was on every fold record
            # and on no replicate record. Same boundary that dropped
            # `fit_diagnosis` in #81b.
            for r in res["seed_replicates"]:
                tp = r.get("tokenisation_predict")
                assert isinstance(tp, dict) and tp.get("prompt_tokens_mean"), (
                    "replicate seed {} carries no tokenisation_predict, so its "
                    "runtime cannot be attributed to prompt length".format(
                        r["model_seed"]))
                assert r.get("prompt_skeleton_tokens"), (
                    "replicate seed {} does not record its example-block size, "
                    "which is what differs between replicate seeds".format(
                        r["model_seed"]))
            rec = store.load("{}_{}/{}__seed42".format(SUBTASK, EDITION, COMPONENT))
            assert np.allclose(rec["proba"].sum(axis=1), 1.0, atol=1e-5)
            assert rec["classes"] == [str(c) for c in classes]
            ok("run_component drives the few-shot family end to end: D3 shape, "
               "B6 stamp, store sidecar with the right identity")
        except AssertionError as e:
            bad("run_component: {}".format(e))
        except Exception as e:
            bad("run_component raised {}: {}".format(type(e).__name__, e))

        # --- the fold record says what 3.3 has to state -------------------
        try:
            rec = res["repeats"][0]["folds_detail"][0]
            for field in ("training", "example_selection", "n_examples",
                          "example_class_counts", "temperature", "tokenisation",
                          "verbalisers", "probability_interface",
                          "n_unused_selection", "selection_slice_note"):
                assert field in rec, "the fold record omits {}".format(field)
            assert rec["training"] == "none"
            assert rec["trainable_params"] == 0, (
                "a few-shot fold reports {} trainable parameters -- something "
                "was adapted that should not have been".format(
                    rec["trainable_params"]))
            assert rec["probability_interface"] == "verbaliser_sequence_score", (
                "the two LLM components must reach D7 through the SAME "
                "interface or their rows are not comparable")
            assert rec["n_unused_selection"] > 0, (
                "the unused selection slice is reported as empty, so 3.3 cannot "
                "state its cost")
            ok("the fold record states training=none, 0 trainable parameters, "
               "{} unused selection rows and the shared probability interface"
               .format(rec["n_unused_selection"]))
        except AssertionError as e:
            bad("fold record: {}".format(e))
        except Exception as e:
            bad("fold record raised {}: {}".format(type(e).__name__, e))

        # --- the POST-CAMPAIGN path, against a real FEW-SHOT artefact ------
        # This is the shape e2_summary has never seen: `selected_params` is
        # None, `selection_grid` is EMPTY and `selected_epoch` is None, because
        # nothing is tuned here. Those are exactly the fields a table builder
        # indexes into without asking, and a break in them is discovered AFTER
        # the B6 money is spent -- the argument test_encoder_cell has made
        # since B4, one arm over.
        try:
            from src import e2_summary, significance

            runs = e2_summary.collect_runs(tmp / "out")
            assert runs, "collect_runs found no artefact to read"
            summary = e2_summary.summary_table(runs)
            assert len(summary) >= 1 and "macro_f1" in summary.columns, summary.columns
            assert summary["component"].isin([COMPONENT]).any(), (
                "the few-shot row is missing from the summary table")
            per_class = e2_summary.per_class_table(runs)
            assert len(per_class) >= len(classes), (
                "per_class_table produced {} rows for {} classes".format(
                    len(per_class), len(classes)))
            e2_summary.baseline_comparison(summary)
            assert rec["selected_params"] is None and rec["selection_grid"] == [], (
                "this check is only meaningful while the few-shot record really "
                "carries an empty selection")
            significance.compare_store(store, n_bootstrap=25)
            ok("the post-campaign path reads a real few-shot artefact -- empty "
               "selection and all -- through summary, per-class, baseline "
               "comparison and the significance layer")
        except AssertionError as e:
            bad("post-campaign path: {}".format(e))
        except Exception as e:
            bad("post-campaign path raised {}: {}".format(type(e).__name__, e))

    finally:
        fs.MODEL_IDS[COMPONENT] = original_id
        fs.N_EXAMPLES = original_n
        e2_runner.load_pool = original_pool
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 50)
    print("  {} passed, {} failed".format(_PASS, _FAIL))
    print("=" * 50)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
