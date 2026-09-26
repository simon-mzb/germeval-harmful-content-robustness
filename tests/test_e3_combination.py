"""test_e3_combination.py -- the E3 combination layer, offline, before a judge call is paid for in time.

WHAT IT GUARDS, in the order the defects would cost:
  * the cascade's thresholds for fold k never see fold k: a mutation that fits
    on all items is detected on a constructed case where leakage pays;
  * the cascade routes as 3.2 says -- a confident stage decides, the last stage
    always decides, and the cheaper rule wins an equal score;
  * soft voting is the equal-weight mean, nothing else;
  * the judge never sees a component name, its candidate order is a pure
    function of (seed, item id), and an answer outside the label set is refused
    rather than coerced;
  * the primary set rule is the Dimension 2 mean rank per family and refuses a tie;
  * on the real store the primary set is one member per family and the members
    are the same items in the same folds.

Run: .venv/bin/python -m tests.test_e3_combination
"""
from __future__ import annotations

import numpy as np

from src import e3_combination as e3

_PASS, _FAIL = 0, 0


def ok(m):
    global _PASS
    _PASS += 1
    print("  \033[32mPASS\033[0m  " + m)


def bad(m):
    global _FAIL
    _FAIL += 1
    print("  \033[31mFAIL\033[0m  " + m)


def check(cond, msg):
    (ok if cond else bad)(msg)


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    p[p == 3] = 2      # a class never predicted
    ref = f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0)
    check(abs(e3.macro_f1_idx(y, p, 4) - ref) < 1e-12, "macro-F1 equals sklearn, absent class included")


def test_soft_vote_is_equal_weight_mean():
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    b = np.array([[0.3, 0.7], [0.4, 0.6]])
    check(np.allclose(e3.soft_vote([a, b]), (a + b) / 2), "soft voting is the plain mean")


def test_cascade_routing():
    p1, c1 = np.array([0, 0, 0]), np.array([0.95, 0.5, 0.5])
    p2, c2 = np.array([1, 1, 1]), np.array([0.5, 0.95, 0.5])
    p3 = np.array([2, 2, 2])
    pred, stage = e3.cascade_predict([(p1, c1), (p2, c2), (p3, np.ones(3))], (0.9, 0.9))
    check(pred.tolist() == [0, 1, 2] and stage.tolist() == [0, 1, 2],
          "a confident stage decides, an unconfident one escalates, the last always decides")
    pred, stage = e3.cascade_predict([(p1, c1), (p2, c2), (p3, np.ones(3))], (1.01, 1.01))
    check(stage.tolist() == [2, 2, 2], "threshold 1.01 means never accept")


def test_cascade_tie_prefers_cheaper_rule():
    y = np.array([0, 1, 0, 1])
    p = np.array([0, 1, 0, 1])      # every stage perfect: all threshold pairs tie on F1
    stages = [(p, np.full(4, 0.6)), (p, np.full(4, 0.6)), (p, np.ones(4))]
    th = e3.fit_cascade_thresholds(stages, y, 2)
    check(th["t1"] <= 0.6, "an equal score is won by the rule that escalates least (t1 {})".format(th["t1"]))


def test_cross_fit_never_uses_the_heldout_fold():
    # Fold 1's items: stage 1 is confidently WRONG there and right everywhere else.
    # Thresholds fitted on folds 0+2 accept stage 1 at high confidence, so fold 1
    # must come out wrong. A threshold fitted on ALL items would learn to reject
    # stage 1 at that confidence and fix fold 1 -- that is the leak this detects.
    n = 300
    fold = np.repeat([0, 1, 2], n // 3)
    y = np.tile([0, 1], n // 2)
    p1 = y.copy()
    c1 = np.full(n, 0.8)
    f1 = fold == 1
    p1[f1] = 1 - y[f1]
    c1[f1] = 0.99
    stages = [(p1, c1), (1 - y, np.full(n, 0.5)), (y.copy(), np.ones(n))]
    res = e3.cross_fit_cascade(stages, y, fold, 2)
    wrong_heldout = float((res["pred"][f1] != y[f1]).mean())
    check(wrong_heldout == 1.0, "fold 1 is scored under thresholds fitted without it (wrong share {})".format(wrong_heldout))
    leaky = e3.fit_cascade_thresholds(stages, y, 2)
    lp, _ = e3.cascade_predict(stages, (leaky["t1"], leaky["t2"]))
    check(float((lp[f1] != y[f1]).mean()) < 1.0,
          "mutation: thresholds fitted on all items would have repaired fold 1, so the check can fail")


def test_judge_prompt_hides_components_and_order_is_deterministic():
    classes = ["agitation", "criticism", "nothing", "subversive"]
    cands = [("criticism", 0.61), ("nothing", 0.55), ("criticism", 0.40)]
    prompt = e3.build_judge_prompt("Ein Beitrag.", "dbo", classes, cands)
    names = list(e3.FAMILY_OF) + ["svm", "encoder", "llammlein", "qwen", "tml", "gbert", "llm"]
    leaked = [n for n in names if n.lower() in prompt.lower()]
    check(not leaked, "the judge prompt names no component ({})".format(leaked or "none"))
    check("Konfidenz 0.61" in prompt and "Kritik (criticism)" in prompt, "candidates carry verbaliser and confidence")
    o1, o2 = e3.candidate_order(1234, 3), e3.candidate_order(1234, 3)
    check(o1 == o2 and sorted(o1) == [0, 1, 2], "candidate order is a pure function of (seed, id)")
    orders = {tuple(e3.candidate_order(i, 3)) for i in range(200)}
    check(len(orders) == 6, "the order is actually permuted across items ({} of 6 seen)".format(len(orders)))


def test_parse_judge_answer():
    classes = ["agitation", "criticism", "nothing", "subversive"]
    check(e3.parse_judge_answer("Kritik", "dbo", classes) == "criticism", "a bare verbaliser parses")
    check(e3.parse_judge_answer(" **Hetze.** ", "dbo", classes) == "agitation", "markdown and punctuation are stripped")
    check(e3.parse_judge_answer("Subversion (subversive)", "dbo", classes) == "subversive", "the label-block echo parses")
    check(e3.parse_judge_answer("Kritik, aber auch Hetze", "dbo", classes) is None, "a hedged answer is refused")
    check(e3.parse_judge_answer("", "dbo", classes) is None, "an empty answer is refused")
    check(e3.parse_judge_answer("Ja", "c2a", ["False", "True"]) == "True", "binary verbaliser maps to its label")


class _Rec:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.model = text, error, "fake"
        self.latency_s, self.created, self.prompt_tokens, self.completion_tokens = 0.01, "t", 1, 1
        self.logprobs = []


class _FakeClient:
    """Scripted GWDG: `script` maps an item id to the sequence of replies it gets."""
    model = "fake-judge"

    def __init__(self, script):
        self.script, self.calls = script, {}

    def label_call(self, prompt, labels, **kw):
        item = int(prompt.split("ITEM")[1].split()[0])
        n = self.calls.get(item, 0)
        self.calls[item] = n + 1
        seq = self.script[item]
        return seq[min(n, len(seq) - 1)]


def test_rate_limit_is_not_a_judge_decision():
    import tempfile
    from pathlib import Path
    e3.time.sleep = lambda s: None
    classes = ["False", "True"]
    rl = _Rec(error="RateLimitError: Error code: 429")
    client = _FakeClient({
        1: [rl, rl, rl, _Rec(text="Ja")],            # rate-limited three times, then answered
        2: [_Rec(text="vielleicht"), _Rec(text="vielleicht")],   # answered, but outside the label set twice
        3: [rl],                                      # never gets through
    })
    items = [{"id": i, "text": "ITEM{} text".format(i), "candidates": [("True", 0.9), ("False", 0.6)]}
             for i in (1, 2, 3)]
    old = e3.MAX_TRANSPORT_RETRIES
    e3.MAX_TRANSPORT_RETRIES = 5
    try:
        cache = e3.JudgeCache("fake", "c2a", root=Path(tempfile.mkdtemp()))
        rows = {r["id"]: r for r in e3.judge_items(client, "c2a", classes, items, cache, workers=1)}
    finally:
        e3.MAX_TRANSPORT_RETRIES = old
    check(rows[1]["label"] == "True" and rows[1]["n_transport_errors"] == 3,
          "a rate limit is retried until answered, and the answer counts ({})".format(rows[1]["label"]))
    check(rows[2]["label"] is None and client.calls[2] == 2 and not rows[2]["transport_exhausted"],
          "an answer outside the label set is retried once, then left for the fallback")
    check(rows[3]["transport_exhausted"], "a call that never gets through is marked exhausted")
    judged = {"c2a": {"items": items, "rows": rows}}
    try:
        e3.assert_judge_answered(judged)
        bad("the artefact write is refused while any item was never answered")
    except SystemExit:
        ok("the artefact write is refused while any item was never answered")
    judged_ok = {"c2a": {"items": items[:2], "rows": rows}}
    try:
        e3.assert_judge_answered(judged_ok)
        ok("a genuine outside-label fallback does not block the write")
    except SystemExit:
        bad("a genuine outside-label fallback does not block the write")


def test_call_pacer_respects_the_quota():
    now = [0.0]
    slept = []
    pacer = e3.CallPacer(min_interval_s=18.9, budget=3, clock=lambda: now[0],
                         sleep=lambda d: (slept.append(d), now.__setitem__(0, now[0] + d)))
    pacer.acquire()
    now[0] += 5.0
    pacer.acquire()
    check(len(slept) == 1 and abs(slept[0] - 13.9) < 1e-9,
          "a call inside the interval waits for the rest of it ({})".format(slept))
    now[0] += 60.0
    pacer.acquire()
    check(len(slept) == 1, "a call after the interval does not wait")
    try:
        pacer.acquire()
        bad("the daily budget stops the campaign")
    except e3.BudgetSpent:
        ok("the daily budget stops the campaign")
    check(3600.0 / e3.MIN_CALL_INTERVAL_S < 200, "the default pace stays under GWDG's 200 calls per hour")


def test_primary_set_rule():
    scores = {
        "a": {"tml_svm": .6, "tml_xgboost": .5, "tml_lightgbm": .4, "encoder_1b": .8,
              "llm_llammlein": .85, "llm_qwen_fewshot": .7},
        "b": {"tml_svm": .4, "tml_xgboost": .45, "tml_lightgbm": .3, "encoder_1b": .5,
              "llm_llammlein": .55, "llm_qwen_fewshot": .3},
        "c": {"tml_svm": .6, "tml_xgboost": .5, "tml_lightgbm": .55, "encoder_1b": .8,
              "llm_llammlein": .82, "llm_qwen_fewshot": .78},
    }
    got = e3.select_primary_set(scores)
    check(got == {"tml": "tml_svm", "encoder": "encoder_1b", "llm": "llm_llammlein"},
          "one member per family by mean rank, even where one subtask disagrees")
    tie = {"a": {"tml_svm": .6, "tml_xgboost": .5}, "b": {"tml_svm": .4, "tml_xgboost": .5}}
    try:
        e3.select_primary_set({k: dict(v, encoder_1b=.8, llm_llammlein=.9) for k, v in tie.items()})
        bad("a tie on mean rank is refused")
    except ValueError:
        ok("a tie on mean rank is refused")


def test_real_store_alignment():
    from src.component_store import ComponentStore
    store = ComponentStore()
    for st in e3.SUBTASKS:
        recs = e3.load_subtask(st, ["tml_svm", "encoder_1b", "llm_llammlein"], store)
        ids = [r["ids"] for r in recs.values()]
        check(all(np.array_equal(ids[0], i) for i in ids), "{}: the members score the same items".format(st))


# ---------------------------------------------------------------------------
# Stage 1 on 2026 (src/e3_combination_2026.py)
# ---------------------------------------------------------------------------

def _synthetic_recs(n, k, seed, classes=None):
    rng = np.random.default_rng(seed)
    classes = classes or [str(i) for i in range(k)]
    y = rng.integers(0, k, n)
    recs = {}
    for j, c in enumerate(["tml_svm", "encoder_1b", "llm_llammlein"]):
        logits = rng.normal(size=(n, k)) + (1.0 + j) * np.eye(k)[y]
        proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        recs[c] = {"proba": proba, "y_true": [classes[i] for i in y], "ids": np.arange(n),
                   "fold": np.full(n, -1), "classes": classes, "meta": {"data_rules_id": "x"}}
    return recs


def test_stage1_never_reads_a_2026_label():
    from src import e3_combination_2026 as s1
    members = list(s1.MEMBERS.values())
    fit = s1.fit_on_2025(_synthetic_recs(600, 3, 1), members)
    direct = e3.fit_cascade_thresholds(
        [(r["proba"].argmax(1), r["proba"].max(1)) for r in _synthetic_recs(600, 3, 1).values()],
        np.array([int(v) for v in _synthetic_recs(600, 3, 1)["tml_svm"]["y_true"]]), 3)
    check((fit["t1"], fit["t2"]) == (direct["t1"], direct["t2"]),
          "stage 1: the cascade thresholds are E3's own fit on all 2025 items")
    r26 = _synthetic_recs(400, 3, 2)
    a = s1.apply_to_2026(fit, r26, members)
    rng = np.random.default_rng(3)
    shuffled = {c: dict(r, y_true=list(rng.permutation(r["y_true"]))) for c, r in r26.items()}
    b = s1.apply_to_2026(fit, shuffled, members)
    check(all(np.array_equal(a[n]["pred"], b[n]["pred"]) for n in ("soft_voting", "cascade", "soft_voting_all_six")),
          "stage 1: 2026 predictions do not change when the 2026 labels are permuted")


def test_stage1_refuses_misaligned_2026_inputs():
    from src import e3_combination_2026 as s1
    members = list(s1.MEMBERS.values())
    fit = s1.fit_on_2025(_synthetic_recs(300, 2, 4), members)
    try:
        s1.apply_to_2026(fit, _synthetic_recs(200, 2, 5, classes=["1", "0"]), members)
        bad("stage 1: a 2026 class order that differs from 2025 is refused")
    except ValueError:
        ok("stage 1: a 2026 class order that differs from 2025 is refused")

    class _Store:
        def __init__(self, recs):
            self.recs = recs
        def load_many(self, keys):
            return {k: dict(self.recs[k.split("/")[1].split("__")[0]],
                            meta={"trained_on": "2025", "data_rules_id": "x"}) for k in keys}
    recs = _synthetic_recs(100, 2, 6)
    recs["encoder_1b"] = dict(recs["encoder_1b"], ids=np.arange(100) + 1)
    try:
        s1.load_2026("c2a", members, _Store(recs))
        bad("stage 1: 2026 members scoring different items are refused")
    except ValueError:
        ok("stage 1: 2026 members scoring different items are refused")


def test_stage1_leaves_the_e3_protocol_alone():
    from src import e3_combination_2026 as s1
    check(e3.protocol_digest() == "f3916558e400",
          "the E3 protocol digest is still f3916558e400 (judge day 3 checks it; change only after G3)")
    check(s1.STAGE1 is not e3.PROTOCOL and "stage1" not in e3.PROTOCOL,
          "stage 1 carries its own rule and adds nothing to e3.PROTOCOL")


def test_stage1_real_store_2026_alignment():
    import warnings
    from src.component_store import ComponentStore
    from src import e3_combination_2026 as s1
    store = ComponentStore()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for st in e3.SUBTASKS:
            recs26 = s1.load_2026(st, sorted(e3.FAMILY_OF), store)
            recs25 = e3.load_subtask(st, list(s1.MEMBERS.values()), store)
            c26 = [str(c) for c in next(iter(recs26.values()))["classes"]]
            c25 = [str(c) for c in next(iter(recs25.values()))["classes"]]
            check(c26 == c25, "{}: all six E4c entries score the same 2026 items in the 2025 class order".format(st))


# ---------------------------------------------------------------------------
# B12 ablations (src/b12_ablation.py)
# ---------------------------------------------------------------------------

def test_b12_two_stage_cascade_is_the_plain_two_stage_rule():
    from src import b12_ablation as b12
    rng = np.random.default_rng(11)
    n, k = 500, 3
    y = rng.integers(0, k, n)
    P = {}
    for j, c in enumerate(["tml_svm", "encoder_1b"]):
        logits = rng.normal(size=(n, k)) + (0.8 + j) * np.eye(k)[y]
        P[c] = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    stages = b12.cascade_stages(P, ["tml_svm", "encoder_1b"])
    th = e3.fit_cascade_thresholds(stages, y, k)
    p1, c1 = P["tml_svm"].argmax(1), P["tml_svm"].max(1)
    p2 = P["encoder_1b"].argmax(1)
    manual = max((round(e3.macro_f1_idx(y, np.where(c1 >= t, p1, p2), k), 12), -t) for t in e3.THRESHOLD_GRID)
    check(abs(th["fit_macro_f1"] - manual[0]) < 1e-12,
          "B12: the duplicated-stage fit finds the best one-threshold two-stage rule")
    pred, _ = e3.cascade_predict(stages, (th["t1"], th["t2"]))
    check(np.array_equal(pred, np.where(c1 >= th["t1"], p1, p2)),
          "B12: a two-member cascade routes exactly like the plain two-stage rule")


def test_b12_variants_and_2026_labels():
    from src import b12_ablation as b12
    from src import e3_combination_2026 as s1
    members = list(s1.MEMBERS.values())
    v = b12.ablation_variants(members)
    check(sorted(v) == sorted(["full"] + ["without_" + m for m in members])
          and all(len(ms) == 2 and m not in ms for m_, ms in v.items() if m_ != "full" for m in [m_[len("without_"):]]),
          "B12: one full variant and one leave-one-out variant per member")
    r25 = _synthetic_recs(600, 3, 21)
    r25 = {c: dict(r, fold=np.arange(600) % 5) for c, r in r25.items()}
    r26 = _synthetic_recs(300, 3, 22)
    P26 = {c: r["proba"] for c, r in r26.items()}
    a = b12.run_2026(r25, P26, ["0", "1", "2"], members)
    full_s1 = s1.apply_to_2026(s1.fit_on_2025(r25, members), r26, members)
    check(np.array_equal(a["full"]["cascade"]["pred"], full_s1["cascade"]["pred"])
          and np.array_equal(a["full"]["soft_voting"], full_s1["soft_voting"]["pred"]),
          "B12: the full 2026 variant reproduces stage 1 exactly")
    check(np.array_equal(a["without_encoder_1b"]["soft_voting"],
                         ((P26["tml_svm"] + P26["llm_llammlein"]) / 2).argmax(1)),
          "B12: soft voting without a member is the mean of the other two")
    b = b12.run_2025(r25, members)
    off = e3.offline_strategies(r25, dict(zip(e3.CASCADE_ORDER, members)))
    check(np.array_equal(b["full"]["cascade"]["pred"], off["cascade"]["pred"]),
          "B12: the full 2025 variant reproduces E3's cross-fitted cascade exactly")


# ---------------------------------------------------------------------------
# E5 irreducible items (src/e5_irreducible.py)
# ---------------------------------------------------------------------------

def test_e5_irreducible_quantities():
    from src import e5_irreducible as ir
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(31)
    conf = rng.random(400)
    correct = rng.random(400) < conf
    check(abs(ir.auroc_confidence(conf, correct) - roc_auc_score(correct, conf)) < 1e-12,
          "E5: confidence AUROC equals sklearn's roc_auc_score")
    tied = np.array([0.5, 0.5, 0.9, 0.2]); ok_ = np.array([True, False, True, False])
    check(abs(ir.auroc_confidence(tied, ok_) - roc_auc_score(ok_, tied)) < 1e-12, "E5: ties count half")
    # No temperature invariance is claimed for the artefact. ONE common temperature
    # preserves the ranking on two classes and not on three; and temperatures are fitted per
    # fold, so even on two classes the POOLED ranking can change. All three are pinned, so the
    # docstring cannot drift back to an invariance claim unnoticed.
    T = 2.7
    corr = rng.random(400) < 0.7
    def soft(z, t):
        return np.exp(z / t) / np.exp(z / t).sum(1, keepdims=True)
    z2 = rng.normal(size=(400, 2))
    check(abs(ir.auroc_confidence(soft(z2, 1).max(1), corr) - ir.auroc_confidence(soft(z2, T).max(1), corr)) < 1e-12,
          "E5: one common temperature, two classes: the ranking and the AUROC are unchanged")
    z3 = rng.normal(size=(400, 3))
    check(abs(ir.auroc_confidence(soft(z3, 1).max(1), corr) - ir.auroc_confidence(soft(z3, T).max(1), corr)) > 1e-6,
          "E5: one common temperature, three classes: the AUROC can move")
    fold = np.arange(400) % 2
    per_fold = np.where(fold[:, None] == 0, soft(z2, 1.0), soft(z2, 3.5))
    check(abs(ir.auroc_confidence(soft(z2, 1).max(1), corr) - ir.auroc_confidence(per_fold.max(1), corr)) > 1e-6,
          "E5: per-fold temperatures, two classes: the pooled AUROC can move too")
    check("no invariance to temperature scaling is claimed" in ir.__doc__.replace("\n   ", " ") or
          "no invariance" in " ".join(ir.__doc__.split()),
          "E5: the module does not claim temperature invariance")
    lo, hi = ir.wilson(81, 263)
    check(abs(lo - 0.2553) < 5e-4 and abs(hi - 0.3662) < 5e-4, "E5: Wilson interval matches a hand-computed value")
    # the binary bound: if every component is wrong, the equal-weight mean is wrong too
    n = 300
    y = rng.integers(0, 2, n)
    probs = [np.where(y[:, None] == np.arange(2), rng.uniform(0.0, 0.499, (n, 1)), 0) for _ in range(6)]
    probs = [np.column_stack([np.where(y == 0, q[:, 0], 1 - q[:, 1]), np.where(y == 1, q[:, 1], 1 - q[:, 0])])
             for q in probs]
    mean = np.mean(probs, axis=0)
    check(not (mean.argmax(1) == y).any(), "E5: on a binary task a mean over all-wrong members is wrong everywhere")
    # the fitted-cascade shares are read off the stage vector the way the artefact reads them
    stage = np.array([0, 1, 2, 2, 1, 0, 2, 1])
    check(abs(float((stage >= 1).mean()) - 0.75) < 1e-12 and abs(float((stage == 2).mean()) - 0.375) < 1e-12,
          "E5: share beyond the first stage and to the last stage are read off the stage vector")
    import json
    from pathlib import Path
    art = json.loads((Path(__file__).resolve().parents[1] / "results" / "e5_irreducible.json").read_text())
    for st, cell in art["cells"].items():
        cf = cell.get("cascade_fitted")
        check(cf is not None and len(cf["thresholds_per_fold"]) == 5
              and 0.0 <= cf["share_to_last_stage"] <= cf["share_beyond_first_stage"] <= 1.0,
              f"E5: {st} artefact carries the fitted cascade, five folds, shares ordered")


def main():
    import warnings
    warnings.simplefilter("ignore")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\n{} passed, {} failed".format(_PASS, _FAIL))
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
