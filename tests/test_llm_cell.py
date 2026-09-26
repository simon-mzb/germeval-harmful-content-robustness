"""test_llm_cell.py -- the LLM half of run_component, on CPU, with no weights.

WHY THIS EXISTS. B5 is a paid pod whose whole job is to find out whether the
generative arm runs end to end. Sending untested plumbing to that pod wastes the
pod: the questions B5 exists to answer are about the MODEL (does a 7B LoRA
fine-tune fit and converge, what does a fold cost, does calibration survive),
not about whether a tensor is the wrong shape. `test_encoder_cell.py` made the
same argument for B4 and found real defects on CPU.

HOW IT RUNS WITHOUT A 27 GB DOWNLOAD. A tiny randomly initialised
LlamaForCausalLM is built from a config and saved to a temp directory, and
MODEL_IDS is pointed at it. The TOKENISER is the real LLaeMmlein one -- it is
already cached (4.5 MB, tokeniser only) and it is the half that matters here,
because the whole probability interface exists to survive how that tokeniser
splits `Subversion`. The weights are random, so nothing about accuracy is
asserted and nothing about accuracy could be: what is asserted is the machinery.

WHAT IT ACTUALLY GUARDS, in the order the defects would bite:
  * the verbaliser really is multi-token under the real tokeniser -- if that
    ever stops being true this test is guarding a problem that no longer exists,
    and the sequence-scoring design would deserve re-reading rather than trust;
  * the scored probability is a proper distribution over the DECLARED class
    order, and it changes when the model changes;
  * the training loss is masked to the label tokens and to nothing else, which
    is llm_protocol.objective and is invisible in any output;
  * right-padding cannot alter a score -- the property that makes batching safe
    under causal attention;
  * run_component produces the replicate shape (one partition seed, two replicates)
    and a store sidecar carrying the LLM config_id.

Run: .venv/bin/python -m tests.test_llm_cell
"""
from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import e2_runner, llm_components as llm
from src.component_store import ComponentStore, compute_config_id
from src.e2_runner import load_pool, run_component
from src.harness import resolve_classes

_PASS, _FAIL = 0, 0

COMPONENT = "llm_llammlein"
SUBTASK, EDITION = "dbo", "2025"    # the four-class path, and the one whose
                                    # verbalisers tokenise worst
N_SPLITS = 2
MAX_EPOCHS = 2
GRID = [{"lr": 1.0e-3}, {"lr": 2.0e-3}]   # a --grid override, not a matrix change


def ok(m):
    global _PASS
    _PASS += 1
    print("  \033[32mPASS\033[0m  " + m)


def bad(m):
    global _FAIL
    _FAIL += 1
    print("  \033[31mFAIL\033[0m  " + m)


def tiny_model_dir(tmp: Path) -> Path:
    """A ~1 M-parameter LlamaForCausalLM with the REAL LLaeMmlein tokeniser.

    The vocabulary has to be the real one or the verbaliser token ids would
    index nothing, so the embedding is sized from the tokeniser rather than
    chosen. Everything else is as small as transformers will accept.
    """
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(llm.MODEL_IDS[COMPONENT])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=1024, pad_token_id=tok.pad_token_id,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    out = tmp / "tiny_llama"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def tiny_pool(per_class=16):
    """Stratified by hand: `subversive` is 0.8% of DBO, so a random slice would
    usually drop it and turn the only four-class test into a three-class one."""
    df = load_pool(SUBTASK, EDITION)
    parts = []
    for c in sorted(set(df["label"].astype(str))):
        sub = df[df["label"].astype(str) == c]
        parts.append(sub.head(min(per_class, len(sub))))
    out = pd.concat(parts).sample(frac=1.0, random_state=0).reset_index(drop=True)
    out["label"] = out["label"].astype(str)
    return out


def main() -> int:
    import torch

    tmp = Path(tempfile.mkdtemp())
    print("\n=== the LLM half of run_component, on CPU (tiny random Llama, "
          "real tokeniser) ===")
    original_id = llm.MODEL_IDS[COMPONENT]
    original_grid = llm.GRIDS[COMPONENT]
    original_pool = e2_runner.load_pool
    try:
        llm.MODEL_IDS[COMPONENT] = str(tiny_model_dir(tmp))
        df = tiny_pool()
        classes = resolve_classes(df["label"].values)
        tok = llm._prepare_tokeniser(llm.MODEL_IDS[COMPONENT])
        label_ids = llm._label_sequences(tok, SUBTASK, classes)

        # --- the premise the whole design rests on ------------------------
        try:
            lengths = [len(x) for x in label_ids]
            assert len(classes) == 4, "the slice lost a class: {}".format(classes)
            assert max(lengths) > 1, (
                "every DBO verbaliser is a single token under this tokeniser, so "
                "the #54 measurement no longer holds and the sequence-scoring "
                "design should be re-read rather than trusted: {}".format(
                    dict(zip(classes, lengths))))
            firsts = [x[0] for x in label_ids]
            assert len(set(firsts)) <= len(firsts)
            ok("the DBO verbalisers are multi-token ({}), which is why single-token "
               "read-off was abandoned".format(dict(zip([str(c) for c in classes], lengths))))
        except AssertionError as e:
            bad("verbaliser premise: {}".format(e))

        # --- the prompt ---------------------------------------------------
        try:
            p = llm.build_prompt("ein Beispieltext", SUBTASK, classes)
            for c in classes:
                assert llm.verbalisers_for(SUBTASK)[str(c)] in p, \
                    "the prompt omits the verbaliser for {}".format(c)
                assert str(c) in p, (
                    "the prompt omits the organiser's English string for {} -- #54 "
                    "requires the mapping to be auditable in the prompt".format(c))
            assert "ein Beispieltext" in p and p.rstrip().endswith("Antwort:")
            assert "Beitrag:" in p
            ok("the prompt carries every verbaliser beside its organiser label and "
               "ends on the answer slot")
        except AssertionError as e:
            bad("prompt: {}".format(e))

        model = llm._build_model(COMPONENT, {"lr": 1e-3}, seed=0, device="cpu")
        prompts, tok_stats = llm._encode_prompts(
            tok, llm._texts(df)[:6], SUBTASK, classes,
            llm.MAX_SEQ_LEN, max(len(x) for x in label_ids))

        # --- the probability interface ------------------------------------
        try:
            proba = llm._label_proba(model, prompts, label_ids,
                                     tok.pad_token_id, "cpu", 3)
            assert proba.shape == (len(prompts), len(classes)), proba.shape
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
            assert (proba > 0).all() and np.isfinite(proba).all()
            ok("scored probabilities are a proper distribution over the {} declared "
               "classes".format(len(classes)))
        except AssertionError as e:
            bad("probability interface: {}".format(e))

        # --- the SCORE ITSELF, not merely its shape ------------------------
        # ⚠️ Review 6 mutated `start + t - 1` to `start + t` and separately
        # dropped the length normalisation, and every check in this file stayed
        # green: a wrong score is still a positive number whose row sums to one.
        # The assertions above can see contract, not arithmetic. So the value is
        # pinned against a reference written from the DEFINITION -- one sequence
        # at a time, no padding, no batching, no shared code with the thing
        # under test.
        try:
            def reference(prompt, lab, normalise=True):
                seq = torch.tensor([prompt + lab])
                with torch.no_grad():
                    lp = torch.log_softmax(
                        model(input_ids=seq).logits.float(), dim=-1)[0]
                start = len(prompt)
                # token t of the label is predicted by the logits at t-1
                total = sum(float(lp[start + t - 1, lab[t]]) for t in range(len(lab)))
                return total / len(lab) if normalise else total

            ref = np.array([[reference(p, l) for l in label_ids] for p in prompts[:3]])
            got = llm._score_batch(model, prompts[:3], label_ids, tok.pad_token_id, "cpu")
            assert np.allclose(ref, got, atol=1e-5), (
                "the batched score is not the length-normalised sequence "
                "log-likelihood of the label: max |diff| {:.3e}".format(
                    np.abs(ref - got).max()))
            # and the check has to be ABLE to tell -- on a label set that were
            # all one token, normalised and unnormalised would coincide and this
            # would pass on a build that had dropped the division entirely
            unnorm = np.array([[reference(p, l, normalise=False) for l in label_ids]
                               for p in prompts[:3]])
            assert not np.allclose(unnorm, got, atol=1e-5), (
                "normalised and unnormalised scores are indistinguishable on "
                "this label set, so this check cannot see the normalisation "
                "it exists to guard")
            ok("the score IS the length-normalised sequence log-likelihood "
               "(|diff| {:.0e} against a from-definition reference; the "
               "unnormalised form differs by {:.1f})".format(
                   np.abs(ref - got).max(), np.abs(unnorm - got).max()))
        except AssertionError as e:
            bad("score value: {}".format(e))

        # --- the adapters sit where the protocol says they sit -------------
        # TWO claims, and they are not the same claim.
        # (i) ⚠️ ARITY IS NOT IDENTITY. `num_hidden_layers x len(target_modules)`
        #     predicts the same count for ANY equinumerous target list, so the
        #     smoke's criterion 2 passed a build whose adapters were on the MLP
        #     (mutated, review 6). #92 happened to change the count; the next
        #     one need not. The suffix comparison is the #92 check: does what
        #     PEFT matched equal what the protocol declared?
        # (ii) the MLP assertion is a different statement -- that this arm
        #     adapts ATTENTION ONLY, which is prose in llm_protocol and nothing
        #     else enforced. It fires if the targets are ever widened to the
        #     MLP, deliberately, so that widening routes through the log
        #     instead of arriving as a silent cost increase.
        try:
            pc = llm._param_counts(model)
            targets = sorted(set(llm.FIXED["lora_target_modules"]))
            assert pc["adapted_suffixes"] == targets, (
                "the adapters sit on {} while llm_protocol declares {} -- a "
                "suffix is matching something other than it reads as (#92)".format(
                    pc["adapted_suffixes"], targets))
            assert not pc["adapted_in_mlp"], (
                "an MLP module carries an adapter: {}".format(pc["adapted_in_mlp"]))
            assert pc["adapted_modules"] == len(targets) * 2, (
                "2 layers x {} targets should be {} adapted modules, got {}".format(
                    len(targets), len(targets) * 2, pc["adapted_modules"]))
            ok("the adapters sit on exactly the declared modules {} and on no "
               "MLP projection".format(targets))
        except AssertionError as e:
            bad("adapter identity: {}".format(e))

        # --- right padding must not change a score ------------------------
        # The batching is only safe because a causal model never attends to a
        # later position. If that ever stops holding -- a bidirectional model, a
        # tokeniser that left-pads by default -- every score in a ragged batch
        # is silently wrong, and nothing downstream would look odd.
        try:
            alone = llm._score_batch(model, [prompts[0]], label_ids,
                                     tok.pad_token_id, "cpu")
            longest = max(range(len(prompts)), key=lambda i: len(prompts[i]))
            together = llm._score_batch(model, [prompts[0], prompts[longest]],
                                        label_ids, tok.pad_token_id, "cpu")
            assert np.allclose(alone[0], together[0], atol=1e-4), (
                "padding changed the score: alone {} vs batched {}".format(
                    alone[0], together[0]))
            ok("a score is unchanged by the padding of its batch neighbours")
        except AssertionError as e:
            bad("padding: {}".format(e))

        # --- the loss is masked to the verbaliser tokens ------------------
        try:
            gold = [label_ids[0]] * len(prompts)
            batch = next(llm._training_batches(prompts, gold, tok.pad_token_id,
                                               4, seed=0, shuffle=False))
            lab, mask, inp = batch["labels"], batch["attention_mask"], batch["input_ids"]
            supervised = (lab != -100)
            assert supervised.sum().item() == 4 * len(label_ids[0]), (
                "supervised positions {} != 4 prompts x {} label tokens".format(
                    supervised.sum().item(), len(label_ids[0])))
            assert bool((lab[supervised] == inp[supervised]).all()), \
                "a supervised position does not carry its own token id"
            assert bool((mask[supervised] == 1).all()), \
                "a supervised position is masked out of attention"
            # and every prompt position is unsupervised, which is the half that
            # would silently train the model to reproduce the instruction
            for k in range(4):
                assert bool((lab[k, :len(prompts[k])] == -100).all()), \
                    "prompt tokens of row {} are in the loss".format(k)
            ok("the loss covers the verbaliser tokens and nothing else "
               "(llm_protocol.objective)")
        except AssertionError as e:
            bad("loss masking: {}".format(e))

        # --- truncation is recorded, and it cuts the TEXT -----------------
        try:
            long_text = "wort " * 4000
            _, stats = llm._encode_prompts(tok, [long_text], SUBTASK, classes,
                                           llm.MAX_SEQ_LEN,
                                           max(len(x) for x in label_ids))
            assert stats["text_truncated"] == 1 and stats["text_truncated_share"] == 1.0
            assert stats["prompt_tokens_max"] <= llm.MAX_SEQ_LEN, (
                "a truncated prompt is still {} tokens, over max_seq_len {}".format(
                    stats["prompt_tokens_max"], llm.MAX_SEQ_LEN))
            # ⚠️ ASSERTED ON WHAT `_encode_prompts` RETURNED. Until review 6
            # this decoded nothing and inspected `build_prompt("x")` instead --
            # a different object, which is why a mutant that truncated the
            # ASSEMBLED prompt from the right, cutting "Antwort:" off every
            # over-long item, passed this check.
            ids_long, _ = llm._encode_prompts(tok, [long_text], SUBTASK, classes,
                                              llm.MAX_SEQ_LEN,
                                              max(len(x) for x in label_ids))
            decoded = tok.decode(ids_long[0])
            for c in classes:
                assert llm.verbalisers_for(SUBTASK)[str(c)] in decoded, (
                    "truncation ate the label block: {!r} is gone from the "
                    "ENCODED prompt".format(str(c)))
                assert str(c) in decoded, (
                    "truncation ate the organiser label {!r}".format(str(c)))
            assert llm._INSTRUCTION[SUBTASK] in decoded, \
                "truncation ate the instruction"
            assert decoded.rstrip().endswith("Antwort:"), (
                "truncation ate the answer slot; the encoded prompt ends {!r}"
                .format(decoded[-40:]))
            ok("over-long text is cut to a budget and counted; the instruction, "
               "label block and answer slot survive IN THE ENCODED PROMPT")
        except AssertionError as e:
            bad("truncation: {}".format(e))

        # --- the best epoch is the one that predicts -----------------------
        # `load_state_dict(..., strict=False)` reports an absent key as nothing
        # at all, so a restore that quietly does no work is invisible: the model
        # that predicts is then the LAST epoch's. Review 6 neutralised the
        # snapshot and every suite stayed green, which is why `_load_trainable`
        # now verifies the load and why this pins it.
        try:
            real_f1, real_snap = llm.f1_score, llm._trainable_snapshot
            taken = []

            def spy(m):
                taken.append(real_snap(m))
                return taken[-1]

            declining = iter([0.9, 0.1, 0.1])       # best is epoch 1, then falls
            llm.f1_score = lambda *a, **k: next(declining)
            llm._trainable_snapshot = spy
            try:
                gold_one = [label_ids[0]] * len(prompts)
                y_one = np.zeros(len(prompts), dtype=int)
                m2, be2, _, _, info2 = llm._fit_one(
                    COMPONENT, {"lr": 5.0e-3}, prompts, gold_one, prompts, y_one,
                    label_ids, tok.pad_token_id, seed=0, device="cpu",
                    max_epochs=3, n_classes=len(classes), patience=99)
            finally:
                llm.f1_score, llm._trainable_snapshot = real_f1, real_snap
            assert be2 == 1 and info2["epochs_run"] == 3, (
                "the forced decline did not produce best=1 of 3: best {} of "
                "{}".format(be2, info2["epochs_run"]))
            live = {n: p.detach().cpu().clone()
                    for n, p in m2.named_parameters() if p.requires_grad}
            drift = max(float((live[k] - taken[0][k]).abs().max()) for k in live)
            assert drift == 0.0, (
                "the model returned is not the best epoch's: max |diff| "
                "{:.3e}".format(drift))
            for bad_state, why in ((None, "an empty snapshot"),
                                   ({}, "an empty snapshot"),
                                   ({"nope.weight": torch.zeros(1)},
                                    "a snapshot of foreign keys")):
                try:
                    llm._load_trainable(m2, bad_state)
                except ValueError:
                    continue
                raise AssertionError("{} was accepted silently".format(why))
            ok("the best epoch is restored into the model that predicts (epoch "
               "1 of 3 run), and a partial restore is refused rather than "
               "silently returning the last epoch")
        except AssertionError as e:
            bad("best-epoch restore: {}".format(e))
        except Exception as e:
            bad("best-epoch restore raised {}: {}".format(type(e).__name__, e))

        # --- one full cell through run_component --------------------------
        try:
            e2_runner.load_pool = lambda st, ed: df
            llm.GRIDS[COMPONENT] = GRID
            store = ComponentStore(root=tmp / "store")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = run_component(SUBTASK, EDITION, COMPONENT, n_splits=N_SPLITS,
                                    max_epochs=MAX_EPOCHS, device="cpu",
                                    bootstrap_n=20, store=store,
                                    out_dir=tmp / "out", ckpt_root=tmp / "ck")
            assert res["meta"]["family"] == "llm_finetune", res["meta"]["family"]
            assert res["package"] == "B6", (
                "the LLM arm is stamped {!r}; a B4 stamp would make the encoder "
                "arm's total unreadable".format(res["package"]))
            assert res["meta"]["config_id"] == compute_config_id(COMPONENT)
            assert len(res["repeats"]) == 1 and res["repeats"][0]["seed"] == 42, \
                "the LLM family runs ONE partition seed (D3)"
            assert [r["model_seed"] for r in res["seed_replicates"]] == [43, 44]
            rec = store.load("{}_{}/{}__seed42".format(SUBTASK, EDITION, COMPONENT))
            assert np.allclose(rec["proba"].sum(axis=1), 1.0, atol=1e-5)
            assert rec["classes"] == [str(c) for c in classes]
            assert rec["meta"]["config_id"] == compute_config_id(COMPONENT)
            ok("run_component drives the LLM family end to end: D3 shape, B6 stamp, "
               "store sidecar with the right identity")
        except AssertionError as e:
            bad("run_component: {}".format(e))
        except Exception as e:
            bad("run_component raised {}: {}".format(type(e).__name__, e))

        # --- the fold record carries what 3.3 has to state ----------------
        try:
            rec = res["repeats"][0]["folds_detail"][0]
            for field in ("selected_params", "selected_epoch", "temperature",
                          "stopping", "tokenisation", "verbalisers",
                          "adapted_modules", "probability_interface"):
                assert field in rec, "the fold record omits {}".format(field)
            # C2: `llm_protocol.truncation` claimed "the share of
            # prompts exceeding max_seq_len" -- zero by construction -- and the
            # figure that WAS recorded described the fit slice while every
            # reported probability comes from the validation slice. Both are
            # recorded now and each names its own slice.
            assert rec["tokenisation"]["slice"] == "fit", rec["tokenisation"]["slice"]
            assert rec["tokenisation_predict"]["slice"] == "validation", (
                "the fold record carries no truncation figure for the slice "
                "that produced its probabilities")
            assert rec["probability_interface"] == "verbaliser_sequence_score"
            assert rec["stopping"]["ceiling"] == MAX_EPOCHS
            assert rec["patience"] == llm.PATIENCE
            assert rec["adapted_modules"] > 0, (
                "PEFT adapted no module -- target_modules {} match nothing in this "
                "architecture, which is #92 in the other direction".format(
                    llm.FIXED["lora_target_modules"]))
            assert len(rec["selection_grid"]) == len(GRID)
            assert all(len(c["epoch_trace"]) <= MAX_EPOCHS for c in rec["selection_grid"])
            ok("the fold record carries the protocol facts 3.3 has to state "
               "({} modules adapted, {} grid cells traced)".format(
                   rec["adapted_modules"], len(rec["selection_grid"])))
        except AssertionError as e:
            bad("fold record: {}".format(e))
        except Exception as e:
            bad("fold record raised {}: {}".format(type(e).__name__, e))

        # --- the class ordering, which is a SINGLE point of failure --------
        # Review 6 assumed run_cv asserted this too and checked: it does not.
        # `predict_proba_fn`'s comparison is the only thing standing between a
        # permuted `classes_` and a pooled matrix whose columns are silently
        # mislabelled -- every probability valid, every class name wrong, and
        # macro-F1 quietly meaningless. (`load_fold_checkpoint` also checks it,
        # but only on a RESUME.) Removing it left every suite green.
        try:
            from src import llm_fewshot as fewshot

            class _Stub:
                classes_ = [str(c) for c in classes]

                def predict_proba(self, frame):
                    raise AssertionError(
                        "predict_proba was reached despite a class-ordering "
                        "mismatch")

            permuted = list(reversed([str(c) for c in classes]))
            assert permuted != [str(c) for c in classes], "the permutation is a no-op"
            for fn, where in ((llm.predict_proba_fn, "llm_components"),
                              (fewshot.predict_proba_fn, "llm_fewshot")):
                try:
                    fn(_Stub(), df.head(2), permuted)
                except ValueError:
                    continue
                raise AssertionError(
                    "{}.predict_proba_fn accepted a permuted class ordering "
                    "instead of refusing it".format(where))
            ok("both LLM components refuse a class ordering that differs from "
               "run_cv's, which is the only check standing between a permuted "
               "classes_ and a mislabelled probability matrix")
        except AssertionError as e:
            bad("class ordering: {}".format(e))
        except Exception as e:
            bad("class ordering raised {}: {}".format(type(e).__name__, e))

        # --- the POST-CAMPAIGN path, against a real fine-tuned LLM artefact --------
        # test_encoder_cell has made this argument since B4 and it holds one arm
        # over: a break in e2_summary is discovered AFTER the money is spent.
        # The generative arm had no equivalent, and its artefacts are a shape
        # e2_summary has never seen -- the few-shot record carries
        # `selected_params: None` and an EMPTY `selection_grid`, which is exactly
        # the kind of field a table builder indexes into without asking.
        try:
            from src import e2_summary, significance

            runs = e2_summary.collect_runs(tmp / "out")
            assert runs, "collect_runs found no artefact to read"
            summary = e2_summary.summary_table(runs)
            assert len(summary) >= 1 and "macro_f1" in summary.columns, summary.columns
            assert summary["component"].isin(["llm_llammlein"]).any(), (
                "the fine-tuned LLM row is missing from the summary table")
            per_class = e2_summary.per_class_table(runs)
            assert len(per_class) >= len(classes), (
                "per_class_table produced {} rows for {} classes".format(
                    len(per_class), len(classes)))
            e2_summary.baseline_comparison(summary)
            # and the significance layer, which reads the STORE rather than the
            # artefact and pairs on ids/folds/gold
            recs = [k for k in store.list_runs() if k.endswith("__seed42")]
            assert recs, "the store holds no seed-42 run to pair"
            significance.compare_store(store, n_bootstrap=25)
            ok("the post-campaign path reads a real fine-tuned LLM artefact: summary, "
               "per-class, baseline comparison, and the significance layer")
        except AssertionError as e:
            bad("post-campaign path: {}".format(e))
        except Exception as e:
            bad("post-campaign path raised {}: {}".format(type(e).__name__, e))

        # --- the scorer reads three scalars, not the whole vocabulary -----
        # Added 2026-09-03 after a real CUDA OOM killed the few-shot arm on an
        # A40. `_score_batch` used to upcast the ENTIRE logits tensor to fp32
        # and log_softmax it: n_seq x width x 152 064 for Qwen2.5-14B, which is
        # 15 GB for a 16-sequence batch of 2 048-token few-shot prompts, read
        # for three scalars per sequence. The fine-tuned arm survives it only
        # because MAX_SEQ_LEN is 512, which is why every existing check passed.
        # The replacement restricts to the read positions BEFORE normalising --
        # arithmetically the same operation, so the assertion is EXACT equality
        # and not a tolerance. A tolerance here would let a real change through.
        try:
            def whole_vocab_reference(model, prompt_ids, label_ids, pad_id, device):
                pairs = [(i, j) for i in range(len(prompt_ids))
                         for j in range(len(label_ids))]
                seqs = [prompt_ids[i] + label_ids[j] for i, j in pairs]
                width = max(len(x) for x in seqs)
                inp = torch.full((len(seqs), width), pad_id, dtype=torch.long)
                msk = torch.zeros((len(seqs), width), dtype=torch.long)
                for k, x in enumerate(seqs):
                    inp[k, :len(x)] = torch.tensor(x, dtype=torch.long)
                    msk[k, :len(x)] = 1
                with torch.no_grad():
                    lg = model(input_ids=inp.to(device),
                               attention_mask=msk.to(device)).logits.float()
                lp = torch.log_softmax(lg, dim=-1).cpu()
                out = np.zeros((len(prompt_ids), len(label_ids)), dtype=np.float64)
                for k, (i, j) in enumerate(pairs):
                    start, lab = len(prompt_ids[i]), label_ids[j]
                    out[i, j] = sum(float(lp[k, start + t - 1, lab[t]])
                                    for t in range(len(lab))) / len(lab)
                return out

            probe_model = llm._build_model(COMPONENT, {"lr": 1e-4}, seed=0,
                                           device="cpu")
            probe_model.eval()
            rng = np.random.default_rng(0)
            worst = 0.0
            for n_prompts, plen in ((4, 37), (3, 120), (2, 301)):
                prompts = [[int(x) for x in
                            rng.integers(5, len(tok) - 5, size=plen)]
                           for _ in range(n_prompts)]
                a = whole_vocab_reference(probe_model, prompts, label_ids,
                                          tok.pad_token_id, "cpu")
                b = llm._score_batch(probe_model, prompts, label_ids,
                                     tok.pad_token_id, "cpu")
                worst = max(worst, float(np.max(np.abs(a - b))))
            del probe_model
            assert max(len(x) for x in label_ids) > 1, (
                "every label is one token here, so the multi-token gather this "
                "check exists for is not exercised")
            assert worst == 0.0, (
                "the position-restricted scorer disagrees with the whole-vocab "
                "one by {:.3e}; it is supposed to be the same arithmetic, so "
                "any difference is a defect and not a rounding".format(worst))
            # And the SECOND half of the same defect: restricting the
            # positions killed the fp32 copies but not the tensor `lm_head`
            # itself returns, which OOM'd again on the real Qwen at 7.53 GB
            # with 7.19 GB free. The forward pass is now split to
            # a byte budget. Each sub-batch is independent, so the assertion is
            # again EXACT: a chunked score that merely rounds the same is a
            # chunked score that is wrong.
            probe_model = llm._build_model(COMPONENT, {"lr": 1e-4}, seed=0,
                                           device="cpu")
            probe_model.eval()
            prompts = [[int(x) for x in rng.integers(5, len(tok) - 5, size=n)]
                       for n in (40, 17, 63, 29, 51)]
            budget = llm.LOGIT_BUDGET_BYTES
            width = max(len(p) + len(l) for p in prompts for l in label_ids)
            per_seq = width * probe_model.config.vocab_size * 2
            sizes, results = [], []
            try:
                for want in (len(prompts) * len(label_ids), 1, 2, 3):
                    llm.LOGIT_BUDGET_BYTES = per_seq * want
                    sizes.append(llm._chunk_size(width, probe_model.config.vocab_size))
                    results.append(llm._score_batch(
                        probe_model, prompts, label_ids, tok.pad_token_id, "cpu"))
            finally:
                llm.LOGIT_BUDGET_BYTES = budget
            spread = max(float(np.max(np.abs(r - results[0]))) for r in results)
            del probe_model
            assert sorted(set(sizes)) != [sizes[0]], (
                "every budget produced the same chunk size {}, so the chunked "
                "path was never exercised".format(sizes)) if len(set(sizes)) == 1 else True
            assert min(sizes) == 1 and max(sizes) >= len(prompts) * len(label_ids), (
                "the sweep did not span from one sequence per pass to all of "
                "them: {}".format(sizes))
            assert spread == 0.0, (
                "splitting the forward pass changed the score by {:.3e}; each "
                "sub-batch is independent, so any difference is a defect and "
                "not a rounding".format(spread))
            ok("the scorer restricts to the read positions before normalising "
               "and is bit-identical to the whole-vocabulary form (label "
               "lengths {}); splitting the forward pass across chunk sizes {} "
               "changes nothing".format([len(x) for x in label_ids], sizes))
        except AssertionError as e:
            bad("scorer equivalence: {}".format(e))
        except Exception as e:
            bad("scorer equivalence raised {}: {}".format(type(e).__name__, e))

        # --- _chunk_size is the one formula, not duplicated per caller -------
        # Extracted 2026-09-05 alongside fewshot_chunking_probe.py, which calls
        # it to reconstruct real batches' chunk counts without touching a
        # model -- a probe holding a second copy of this arithmetic is exactly
        # the defect class #61/#64/#86 already caught once in this module.
        try:
            budget = llm.LOGIT_BUDGET_BYTES
            try:
                # per_seq_bytes = width * vocab * 2 = 2*3*2 = 12 throughout.
                assert llm._chunk_size(2, 3, budget=36) == 3
                assert llm._chunk_size(2, 3, budget=35) == 2, (
                    "floor division: 35 // 12 = 2, not round(35/12) = 3")
                assert llm._chunk_size(2, 3, budget=12) == 1
                assert llm._chunk_size(2, 3, budget=1) == 1, (
                    "a budget too small for even one sequence must still "
                    "return 1, or _score_batch divides by zero next")
                llm.LOGIT_BUDGET_BYTES = 24
                assert llm._chunk_size(2, 3) == 2, (
                    "budget=None must read the CURRENT module global, not a "
                    "value frozen at import time -- a default-argument bug "
                    "here would silently break the monkeypatch sweep above, "
                    "which mutates llm.LOGIT_BUDGET_BYTES between calls")
            finally:
                llm.LOGIT_BUDGET_BYTES = budget
            ok("_chunk_size matches width*vocab*2 floor-division and reads "
               "LOGIT_BUDGET_BYTES live rather than at import time")
        except AssertionError as e:
            bad("_chunk_size: {}".format(e))
        except Exception as e:
            bad("_chunk_size raised {}: {}".format(type(e).__name__, e))

    finally:
        llm.MODEL_IDS[COMPONENT] = original_id
        llm.GRIDS[COMPONENT] = original_grid
        e2_runner.load_pool = original_pool
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 50)
    print("  {} passed, {} failed".format(_PASS, _FAIL))
    print("=" * 50)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
