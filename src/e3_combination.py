"""
e3_combination.py -- E3: the combination layer over the stored E2 components, and its
verdict G3 (does combination help?).

WHAT IT ANSWERS. SQ2: does combining the components improve on the best single
component, and which of the three strategies of 3.2 does it best? Everything
runs on the calibrated out-of-fold probabilities E2 already stored, so the
components are identical across strategies and a difference between strategies
is a difference of arbitration rule, not of a re-trained system (3.2).

THE PROTOCOL IS FIXED IN `PROTOCOL` BELOW AND WAS WRITTEN BEFORE ANY E3 RESULT.
Its digest travels in the artefact. The one thing measured beforehand is
`combination_headroom.json` (`src.combination_headroom`): an oracle ceiling and soft voting over
all six / top three / top two. That pre-flight is why the primary set below is
NOT top-two -- choosing the set that pre-flight happened to favour would be a
selection on the evaluation items. The set rule is structural instead.

1. **The component set** (primary): the best component of each FAMILY -- TML,
   encoder, fine-tuned or few-shot LLM -- by mean rank of pooled E2 macro-F1
   across the three subtasks, the Dimension 2 selection rule applied per family. One member per
   family is the pipeline 3.2 draws, and the cascade's three stages need exactly
   that. Counterfactual: all six members swamp the vote with four components
   20+ pp behind (headroom: -7.6 to -9.6 pp); top-two has no classical stage for
   the cascade and was visible before this design. All six are still reported
   for soft voting, as the secondary set that reconnects to the pre-flight.
2. **Soft voting**: equal-weight mean of the calibrated distributions, argmax.
3. **Confidence cascade**: TML -> encoder -> LLM. A stage decides when its top
   calibrated probability reaches its threshold. The two thresholds are
   CROSS-FITTED: for every fold k they are chosen on the out-of-fold items of
   the other folds by macro-F1 and applied to fold k only, so no item is ever
   scored under a threshold its own label helped to choose ("validation folds,
   never evaluation data", 3.2). Ties: least LLM escalation, then least encoder
   escalation -- the cheaper rule wins an equal score. Counterfactual: one
   threshold pair fitted on all items leaks; a fixed a-priori pair (e.g. 0.9) is
   an arbitrary number the thesis would have to defend.
4. **LLM-as-judge**: unanimous items keep their label. On the others, the GWDG
   model named in the call (`mistral-medium-3.5-128b`, chosen by `src.gwdg_smoke`) receives the
   component instruction, the label block, the text, and every member's label
   with its calibrated confidence -- WITHOUT component names and in an order
   drawn per item from (seed, item id) (3.2, zheng2023). It may answer any label
   of the set, not only a candidate. Temperature 0. A transport error (rate
   limit, timeout, 5xx) is retried with backoff until answered: it says nothing
   about the judge. An answer outside the label set is retried once; a second
   one falls back to the soft vote and is COUNTED, never hidden.
   ⚠️ Treating HTTP 429 as a judge failure would silently turn disagreement
   items into soft votes reported as the judge; this rule and
   `assert_judge_answered` exist to prevent that.
5. **G3**: every strategy against the best single component, paired on the
   same items (`significance.compare`: McNemar + paired bootstrap), Holm over
   the whole family of strategy-vs-best comparisons in this artefact (the
   largest family). "Combination helps" on a subtask iff a primary
   strategy is ABOVE the best single component and survives Holm on the
   bootstrap p-value. Otherwise the answer on that subtask is "no".

WHAT IT DOES NOT CLAIM. One seed per component: a corrected p says "this
combination beat that model on these items", not "this method beats that
method". The TML member ran on Darwin/arm64 and the neural members on
Linux/x86_64; a combination is not a baseline delta, so the same-machine rule
does not apply, but the artefact records `mixes_machines`. The served judge has
no version handle: the campaign is one run, the `/models` listing is stored, and a
fixed probe is re-called at the end so a silent swap would be visible.

Usage
-----
python -m src.e3_combination --help
python -m src.e3_combination --plan               # members, volumes; no network, no write
python -m src.e3_combination --offline            # voting + cascade to stdout; no network, no write
python -m src.e3_combination --judge-smoke 10     # 10 real judge calls on dbo, printed, cached
python -m src.e3_combination --go                 # the campaign; writes results/e3_combination.json
python -m src.e3_combination --go --judge-model gemma-4-31b-it --sensitivity
                                                  # a second judge on the same items -> its own artefact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "e3_combination.json"
PRED_OUT = RESULTS / "e3_combination_predictions.npz"
JUDGE_CACHE = RESULTS / "e3_judge"

SUBTASKS = ("c2a", "dbo", "vio")
EDITION = "2025"
SEED = 42
FAMILY_OF = {
    "tml_svm": "tml", "tml_xgboost": "tml", "tml_lightgbm": "tml",
    "encoder_1b": "encoder",
    "llm_llammlein": "llm", "llm_qwen_fewshot": "llm",
}
CASCADE_ORDER = ("tml", "encoder", "llm")
THRESHOLD_GRID = tuple([round(x, 2) for x in np.arange(0.0, 1.0001, 0.01)] + [1.01])
PRIMARY_JUDGE = "mistral-medium-3.5-128b"
N_DRIFT_PROBE = 20
MAX_TRANSPORT_RETRIES = 40      # ~1 h of backoff per item before the campaign stops

# ⚠️ THE GWDG QUOTA, MEASURED FROM THE RESPONSE HEADERS, NOT FROM THE DOCS:
# 30 requests/minute, 200/hour, 1000/day, 3000/month per account. A short burst
# never touches the hour bucket, so a smoke test cannot see it. The whole primary
# judge is ~2260 calls, i.e. three days and three quarters of a month, so the
# campaign is PACED rather than retried: one call at a time, never faster than
# the hour bucket refills, and it stops cleanly at a daily budget and resumes
# from the cache the next day. Each day's batch opens with a drift probe on
# already-answered items, because a multi-day campaign is exactly what the
# one-campaign rule guards against.
MIN_CALL_INTERVAL_S = 3600.0 / 190     # just under the 200/h bucket
DAILY_CALL_BUDGET = 900                # of 1000/day, leaving room for other GWDG use
N_DAY_PROBE = 10

PROTOCOL: dict[str, Any] = {
    "package": "B7/E3",
    "input": "E2 component store, edition 2025, seed 42, condition none, calibrated OOF probabilities",
    "primary_set_rule": "best component per family (tml / encoder / llm) by mean rank of pooled E2 "
                        "macro-F1 across c2a, dbo, vio; ties by lower inference cost",
    "secondary_set": "all six E2 components, soft voting only",
    "soft_voting": "equal-weight mean of calibrated distributions, argmax; ties to the first class "
                   "in the stored class order",
    "cascade": {
        "order": list(CASCADE_ORDER),
        "confidence": "max calibrated probability of the stage",
        "accept_if": "confidence >= threshold",
        "threshold_grid": "0.00..1.00 step 0.01, plus 1.01 = never accept",
        "fitting": "cross-fitted: thresholds for fold k chosen on the other folds' OOF items by "
                   "pooled macro-F1, applied to fold k only",
        "tie_break": "lowest LLM escalation share, then lowest encoder escalation share, then lowest thresholds",
    },
    "judge": {
        "invoked_on": "items whose primary-set argmax labels are not unanimous",
        "backend": "gwdg",
        "model_primary": PRIMARY_JUDGE,
        "shown": "component instruction, label block, text, every member's label with calibrated "
                 "confidence (2 decimals); no component names; order permuted per item by "
                 "numpy default_rng([42, item_id])",
        "answer_space": "any label of the subtask's label set",
        "decoding": "temperature 0, max_tokens 8, top_logprobs 5",
        "failure": "transport errors (rate limit, timeout, 5xx) are retried with exponential backoff "
                   "until the call is answered -- infrastructure, not a judge decision; an answer "
                   "outside the label set is retried once, then the item falls back to the soft vote "
                   "over the primary set and is counted",
        "drift_probe": "the first {} items per subtask (by id) that the judge ANSWERED, re-called after "
                       "the campaign; only answered pairs are compared".format(N_DRIFT_PROBE),
    },
    "g3": {
        "comparison": "each strategy vs the best single component, significance.compare on identical items",
        "family": "every strategy-vs-best comparison in the artefact, Holm on the bootstrap p-value",
        "helps_if": "primary strategy macro-F1 above best single AND survives Holm (bootstrap)",
        "wording": "#58(b): where the test is negative the text says indistinguishable",
    },
}


def protocol_digest() -> str:
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def e2_key(subtask: str, component: str) -> str:
    return "{}_{}/{}__seed{}".format(subtask, EDITION, component, SEED)


def load_subtask(subtask: str, components: Sequence[str], store=None) -> dict[str, dict]:
    """The members' stored runs, aligned by id and refused if they are not the same items."""
    from src.component_store import ComponentStore
    store = store or ComponentStore()
    recs = store.load_many([e2_key(subtask, c) for c in components])
    out: dict[str, dict] = {}
    base = None
    for c in components:
        r = recs[e2_key(subtask, c)]
        if r["meta"].get("uncalibrated"):
            raise ValueError("{} / {} is flagged uncalibrated; E3 consumes calibrated "
                             "probabilities only".format(subtask, c))
        order = np.argsort(np.asarray(r["ids"]), kind="stable")
        r = dict(r, proba=np.asarray(r["proba"])[order], ids=np.asarray(r["ids"])[order],
                 fold=np.asarray(r["fold"])[order],
                 y_true=[str(v) for v in np.asarray(r["y_true"])[order]])
        if base is None:
            base = r
        else:
            for f in ("ids", "fold"):
                if not np.array_equal(base[f], r[f]):
                    raise ValueError("{}: {} differs between {} and {}".format(subtask, f, components[0], c))
            if base["y_true"] != r["y_true"] or list(base["classes"]) != list(r["classes"]):
                raise ValueError("{}: gold labels or class order differ for {}".format(subtask, c))
            if base["meta"].get("data_rules_id") != r["meta"].get("data_rules_id"):
                raise ValueError("{}: data-rule generations differ for {}".format(subtask, c))
        out[c] = r
    return out


def select_primary_set(scores: dict[str, dict[str, float]]) -> dict[str, str]:
    """family -> component, by mean rank across subtasks of pooled macro-F1, per family.

    `scores[subtask][component]` = pooled macro-F1. Refuses a tie: the cost tie-break
    needs a cost figure this function does not have, and a silent choice is worse.
    """
    chosen: dict[str, str] = {}
    for fam in CASCADE_ORDER:
        members = sorted(c for c in FAMILY_OF if FAMILY_OF[c] == fam)
        ranks = {c: [] for c in members}
        for st, row in scores.items():
            present = [c for c in members if c in row]
            ordered = sorted(present, key=lambda c: -row[c])
            for i, c in enumerate(ordered):
                ranks[c].append(i + 1)
        mean = {c: float(np.mean(v)) for c, v in ranks.items() if v}
        best = min(mean.values())
        winners = [c for c, m in mean.items() if m == best]
        if len(winners) != 1:
            raise ValueError("family {}: tie on mean rank between {}; the cost tie-break "
                             "must be decided by hand".format(fam, winners))
        chosen[fam] = winners[0]
    return chosen


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_f1_idx(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    conf = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(conf).astype(float)
    denom = conf.sum(0) + conf.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def per_class_f1_idx(y: np.ndarray, pred: np.ndarray, k: int) -> list[float]:
    conf = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(conf).astype(float)
    denom = conf.sum(0) + conf.sum(1)
    return [float(v) for v in np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)]


# ---------------------------------------------------------------------------
# Strategies (offline)
# ---------------------------------------------------------------------------

def soft_vote(probas: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probas), axis=0)


def cascade_predict(stages: Sequence[tuple[np.ndarray, np.ndarray]], thresholds: Sequence[float]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """stages = [(pred, conf), ...] in cascade order; the last stage always decides.

    Returns (prediction, deciding stage index).
    """
    n = len(stages[0][0])
    pred = stages[-1][0].copy()
    stage = np.full(n, len(stages) - 1, dtype=np.int8)
    undecided = np.ones(n, dtype=bool)
    for i, (p, c) in enumerate(stages[:-1]):
        take = undecided & (c >= thresholds[i])
        pred[take] = p[take]
        stage[take] = i
        undecided &= ~take
    return pred, stage


def fit_cascade_thresholds(stages, y: np.ndarray, k: int, grid=THRESHOLD_GRID) -> dict[str, Any]:
    """Best (t1, t2) on these items; ties to least LLM, then least encoder escalation."""
    (p1, c1), (p2, c2), (p3, _) = stages
    best = None
    for t1 in grid:
        a1 = c1 >= t1
        for t2 in grid:
            a2 = (~a1) & (c2 >= t2)
            pred = np.where(a1, p1, np.where(a2, p2, p3))
            f1 = round(macro_f1_idx(y, pred, k), 12)
            llm_share = float((~a1 & ~a2).mean())
            enc_share = float((~a1).mean())
            key = (-f1, llm_share, enc_share, t1, t2)
            if best is None or key < best[0]:
                best = (key, t1, t2, f1)
    return {"t1": best[1], "t2": best[2], "fit_macro_f1": best[3]}


def cross_fit_cascade(stages, y: np.ndarray, fold: np.ndarray, k: int, grid=THRESHOLD_GRID
                      ) -> dict[str, Any]:
    n = len(y)
    pred = np.empty(n, dtype=np.int64)
    stage = np.empty(n, dtype=np.int8)
    per_fold = []
    for f in sorted(np.unique(fold).tolist()):
        tr, te = fold != f, fold == f
        th = fit_cascade_thresholds([(p[tr], c[tr]) for p, c in stages], y[tr], k, grid)
        p, s = cascade_predict([(p[te], c[te]) for p, c in stages], (th["t1"], th["t2"]))
        pred[te], stage[te] = p, s
        per_fold.append(dict(th, fold=int(f), n_heldout=int(te.sum())))
    return {"pred": pred, "stage": stage, "thresholds": per_fold}


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------

def candidate_order(item_id: int, n: int, seed: int = SEED) -> list[int]:
    return np.random.default_rng([seed, int(item_id)]).permutation(n).tolist()


def build_judge_prompt(text: str, subtask: str, classes: Sequence[str],
                       candidates: Sequence[tuple[str, float]]) -> str:
    """Component instruction and label block (llm_components), then anonymous candidates.

    `candidates` must already be in the per-item random order; nothing here knows
    which component produced which line, by construction.
    """
    from src.llm_components import _INSTRUCTION, verbalisers_for
    verb = verbalisers_for(subtask)
    block = "\n".join("- {} ({})".format(verb[str(c)], c) for c in classes)
    cand = "\n".join("- {} ({}), Konfidenz {:.2f}".format(verb[str(lab)], lab, conf)
                     for lab, conf in candidates)
    return (
        "{}\n\nMoegliche Antworten:\n{}\n\nBeitrag: {}\n\n"
        "Automatische Klassifikatoren haben diesen Beitrag bereits eingeordnet "
        "(in zufaelliger Reihenfolge, mit kalibrierter Konfidenz). "
        "Sie stimmen nicht ueberein und koennen sich irren:\n{}\n\nAntwort:"
    ).format(_INSTRUCTION[subtask.lower()], block, text, cand)


def judge_system(subtask: str, classes: Sequence[str]) -> str:
    from src.llm_components import verbalisers_for
    verb = verbalisers_for(subtask)
    return ("Du bist ein strenger Klassifikator. Antworte mit genau einer der folgenden "
            "Antworten und sonst nichts: " + ", ".join(verb[str(c)] for c in classes) + ".")


_STRIP = re.compile(r"^[\s\"'`*.:,;!-]+|[\s\"'`*.:,;!-]+$")


def parse_judge_answer(text: str | None, subtask: str, classes: Sequence[str]) -> str | None:
    """The class label the answer names, or None. Verbaliser or organiser label, whole answer only."""
    if not text:
        return None
    from src.llm_components import verbalisers_for
    verb = verbalisers_for(subtask)
    ans = _STRIP.sub("", text).casefold()
    ans = re.sub(r"\s*\(.*\)$", "", ans)
    for c in classes:
        if ans in (verb[str(c)].casefold(), str(c).casefold()):
            return str(c)
    return None


def judge_first_token_distribution(logprobs: list[dict], subtask: str, classes: Sequence[str]
                                   ) -> list[float] | None:
    """Label distribution from the first token's top alternatives, where they are unambiguous.

    An alternative counts for a label when it is a non-empty prefix of exactly
    one verbaliser. Mass that maps to no label is dropped and the rest
    renormalised -- a renormalisation with the bias badhe2026 describes, which is
    why this distribution is secondary and the label is primary.
    """
    if not logprobs or not logprobs[0].get("top"):
        return None
    from src.llm_components import verbalisers_for
    verb = verbalisers_for(subtask)
    vs = [verb[str(c)].casefold() for c in classes]
    mass = np.zeros(len(classes))
    for alt in logprobs[0]["top"]:
        tok = alt["token"].strip().casefold()
        if not tok:
            continue
        hits = [i for i, v in enumerate(vs) if v.startswith(tok)]
        if len(hits) == 1:
            mass[hits[0]] += float(np.exp(alt["logprob"]))
    if mass.sum() <= 0:
        return None
    return (mass / mass.sum()).tolist()


class JudgeCache:
    """Append-only JSONL per (model, subtask): one campaign, resumable, never rewritten."""

    def __init__(self, model: str, subtask: str, root: Path = JUDGE_CACHE):
        self.path = root / model / "{}.jsonl".format(subtask)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.rows: dict[int, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("kind", "judge") == "judge":
                        self.rows[int(r["id"])] = r

    def put(self, row: dict) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row.get("kind", "judge") == "judge":
                self.rows[int(row["id"])] = row


class CallPacer:
    """Serialises every GWDG call of the process to the quota, and counts them.

    `budget` is the number of calls this process may still make; when it is spent
    `acquire` raises BudgetSpent and the campaign stops with its cache intact.
    """

    def __init__(self, min_interval_s: float = MIN_CALL_INTERVAL_S, budget: int = DAILY_CALL_BUDGET,
                 clock=time.monotonic, sleep=time.sleep):
        self.min_interval_s, self.budget = min_interval_s, budget
        self.clock, self.sleep = clock, sleep
        self._lock = threading.Lock()
        self._last = None
        self.calls = 0

    def acquire(self) -> None:
        with self._lock:
            if self.calls >= self.budget:
                raise BudgetSpent("daily call budget of {} spent".format(self.budget))
            now = self.clock()
            if self._last is not None and now - self._last < self.min_interval_s:
                self.sleep(self.min_interval_s - (now - self._last))
            self._last = self.clock()
            self.calls += 1


class BudgetSpent(RuntimeError):
    pass


PACER: CallPacer | None = None


def _call_once(client, prompt: str, system: str, labels: Sequence[str]):
    if PACER is not None:
        PACER.acquire()
    return client.label_call(prompt, labels, system=system, logprobs=True, top_logprobs=5, max_tokens=8)


def judge_items(client, subtask: str, classes: Sequence[str], items: list[dict],
                cache: JudgeCache, *, workers: int = 4, kind: str = "judge",
                log: Callable[[str], None] = print) -> list[dict]:
    """items: {id, text, candidates:[(label, conf)]} in final order. Returns cache rows."""
    from src.llm_components import verbalisers_for
    verb = verbalisers_for(subtask)
    labels = [verb[str(c)] for c in classes]
    system = judge_system(subtask, classes)

    def work(it):
        rng = np.random.default_rng([SEED, int(it["id"]), 7])
        prompt = build_judge_prompt(it["text"], subtask, classes, it["candidates"])
        sha = hashlib.sha256((system + "\n" + prompt).encode()).hexdigest()[:16]
        if kind == "judge":
            prev = cache.rows.get(int(it["id"]))
            # A row is final when it carries a label, or when the judge ANSWERED
            # and the answer was outside the label set twice (the counted
            # fallback). Only rows that never got through are asked again --
            # including every pre-backoff row, which lacks n_transport_errors.
            if prev and prev.get("prompt_sha") == sha and (
                    prev.get("label") is not None
                    or ("n_transport_errors" in prev and not prev.get("transport_exhausted"))):
                return prev
        attempts = []
        label, rec = None, None
        answers, transport = 0, 0
        while answers < 2:
            rec = _call_once(client, prompt, system, labels)
            attempts.append({"text": rec.text, "error": rec.error, "latency_s": round(rec.latency_s, 3)})
            if rec.error:
                transport += 1
                if transport > MAX_TRANSPORT_RETRIES:
                    break
                wait = 150.0 if "429" in str(rec.error) else min(120.0, 2.0 * 2 ** min(transport, 6))
                time.sleep(wait * (0.5 + rng.random()))
                continue
            answers += 1
            label = parse_judge_answer(rec.text, subtask, classes)
            if label is not None:
                break
        row = {
            "kind": kind, "id": int(it["id"]), "prompt_sha": sha, "model_requested": client.model,
            "model_served": rec.model, "created": rec.created, "label": label,
            "candidates": [[str(a), round(float(b), 4)] for a, b in it["candidates"]],
            "attempts": attempts, "n_transport_errors": transport,
            "transport_exhausted": transport > MAX_TRANSPORT_RETRIES,
            "prompt_tokens": rec.prompt_tokens,
            "completion_tokens": rec.completion_tokens,
            "first_token_top": (rec.logprobs[0]["top"] if rec.logprobs else None),
            "distribution": judge_first_token_distribution(rec.logprobs, subtask, classes),
        }
        cache.put(row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(work, items)):
            rows.append(row)
            if (i + 1) % 250 == 0:
                log("  {} {}: {}/{} judged".format(subtask, kind, i + 1, len(items)))
    return rows


# ---------------------------------------------------------------------------
# One subtask
# ---------------------------------------------------------------------------

def texts_for(subtask: str, ids: np.ndarray) -> dict[int, str]:
    from src.e2_runner import load_pool
    from src.encoder_components import _texts
    df = load_pool(subtask, EDITION)
    txt = dict(zip(df["id"].astype(int).tolist(), _texts(df)))
    missing = [int(i) for i in ids if int(i) not in txt]
    if missing:
        raise ValueError("{}: {} stored ids are not in the live pool (first {})".format(
            subtask, len(missing), missing[:3]))
    return txt


def offline_strategies(recs: dict[str, dict], primary: dict[str, str]) -> dict[str, Any]:
    any_rec = next(iter(recs.values()))
    classes = [str(c) for c in any_rec["classes"]]
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[v] for v in any_rec["y_true"]])
    fold = np.asarray(any_rec["fold"])
    members = [primary[f] for f in CASCADE_ORDER]
    P = {c: recs[c]["proba"] for c in recs}

    out: dict[str, Any] = {"classes": classes, "y": y, "fold": fold, "members": members}
    out["single"] = {c: {"pred": P[c].argmax(1), "macro_f1": macro_f1_idx(y, P[c].argmax(1), k)}
                     for c in P}
    sv = soft_vote([P[c] for c in members])
    out["soft_voting"] = {"proba": sv, "pred": sv.argmax(1)}
    sv_all = soft_vote([P[c] for c in sorted(P)])
    out["soft_voting_all"] = {"proba": sv_all, "pred": sv_all.argmax(1), "members": sorted(P)}
    stages = [(P[c].argmax(1), P[c].max(1)) for c in members]
    out["cascade"] = cross_fit_cascade(stages, y, fold, k)
    preds = np.stack([P[c].argmax(1) for c in members])
    out["disagree"] = (preds != preds[0]).any(0)
    return out


def assemble_judge(off: dict, recs: dict, rows_by_id: dict[int, dict]) -> dict[str, Any]:
    ids = np.asarray(next(iter(recs.values()))["ids"])
    classes = off["classes"]
    idx = {c: i for i, c in enumerate(classes)}
    members = off["members"]
    pred = recs[members[0]]["proba"].argmax(1).copy()
    dis = off["disagree"]
    n_fallback, n_outside = 0, 0
    for j in np.flatnonzero(dis):
        row = rows_by_id.get(int(ids[j]))
        lab = row.get("label") if row else None
        if lab is None:
            pred[j] = off["soft_voting"]["pred"][j]
            n_fallback += 1
            continue
        pred[j] = idx[lab]
        if lab not in {str(c) for c, _ in row["candidates"]}:
            n_outside += 1
    return {"pred": pred, "n_invoked": int(dis.sum()), "n_fallback": n_fallback,
            "n_answer_outside_candidates": n_outside}


# ---------------------------------------------------------------------------
# G3
# ---------------------------------------------------------------------------

def day_probe_for(client, per, args) -> dict | None:
    """Re-call N_DAY_PROBE already-answered items at the start of a batch; log agreement.

    A multi-day campaign cannot rule out a silent model swap; this makes one
    visible. Each batch's result is appended to the cache as kind="day_probe" and
    the artefact reports all of them.
    """
    for st in SUBTASKS:
        cache = JudgeCache(args.judge_model, st)
        answered = sorted((r for r in cache.rows.values() if r.get("label")), key=lambda r: r["id"])
        if len(answered) < N_DAY_PROBE:
            continue
        recs, off = per[st]
        ids = np.asarray(recs[off["members"][0]]["ids"])
        txt = texts_for(st, ids)
        probe = [{"id": r["id"], "text": txt[r["id"]], "candidates": [tuple(c) for c in r["candidates"]]}
                 for r in answered[:N_DAY_PROBE]]
        again = judge_items(client, st, off["classes"], probe, cache, workers=1, kind="day_probe")
        same = sum(1 for r0, r1 in zip(answered, again) if r1["label"] == r0["label"])
        n_ans = sum(1 for r1 in again if r1["label"] is not None)
        print("  day probe {}: {}/{} answered, {} identical to the first answer".format(st, n_ans, len(probe), same))
        return {"subtask": st, "n": len(probe), "answered": n_ans, "same_label": same}
    return None


def assert_judge_answered(judged: dict, max_transport_share: float = 0.0) -> None:
    """Refuse to report a judge that did not answer.

    A transport failure is not a judge decision, and before this existed an
    exhausted rate limit became a soft vote and was reported as the judge. Any
    item whose calls never got through stops the write; the campaign is re-run
    (the cache resumes) instead of publishing a mixture.
    """
    for st, J in judged.items():
        rows = [J["rows"].get(it["id"]) for it in J["items"]]
        missing = sum(1 for r in rows if r is None)
        exhausted = sum(1 for r in rows if r is not None and r.get("transport_exhausted"))
        if missing or exhausted > max_transport_share * len(rows):
            raise SystemExit("REFUSING to write: {} -- {} item(s) without a row, {} whose calls "
                             "never got through. Re-run --go; the cache resumes.".format(
                                 st, missing, exhausted))


def as_record(template: dict, pred_idx: np.ndarray, k: int) -> dict:
    """A store-shaped record whose argmax is `pred_idx`, for significance.compare."""
    proba = np.eye(k)[pred_idx]
    return {"proba": proba, "y_true": template["y_true"], "ids": template["ids"],
            "classes": template["classes"], "fold": template["fold"], "meta": template["meta"]}


def _cache_rows(model: str, subtask: str, kind: str) -> list[dict]:
    path = JUDGE_CACHE / model / "{}.jsonl".format(subtask)
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("kind", "judge") == kind]


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except Exception:
        return None


def _usage() -> str:
    return __doc__


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--judge-smoke", type=int, metavar="N")
    mode.add_argument("--go", action="store_true")
    ap.add_argument("--judge-model", default=PRIMARY_JUDGE)
    ap.add_argument("--sensitivity", action="store_true",
                    help="a second judge: writes e3_combination_sensitivity_<model>.json")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel calls; the pacer serialises them anyway (GWDG quota)")
    ap.add_argument("--daily-budget", type=int, default=DAILY_CALL_BUDGET,
                    help="calls this invocation may make before stopping cleanly")
    ap.add_argument("--no-pacing", action="store_true", help="tests only")
    ap.add_argument("--skip-day-probe", action="store_true",
                    help="a restart on the SAME day: the day's probe already ran")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if not (args.plan or args.offline or args.judge_smoke or args.go):
        print(_usage())
        return 0
    try:
        return run(args)
    except BudgetSpent as exc:
        print("STOPPED CLEANLY: {} ({} calls this run). The cache holds every answer; "
              "re-run --go after the daily quota resets.".format(exc, PACER.calls if PACER else 0))
        return 3


def run(args) -> int:
    from src.component_store import ComponentStore
    store = ComponentStore()
    scores = {}
    for st in SUBTASKS:
        scores[st] = {}
        for c in FAMILY_OF:
            meta = json.loads(store._json(e2_key(st, c)).read_text(encoding="utf-8"))
            scores[st][c] = float(meta["pooled_macro_f1"])
    primary = select_primary_set(scores)
    print("E3 protocol digest {}; primary set {}".format(protocol_digest(), primary))

    per = {}
    for st in SUBTASKS:
        recs = load_subtask(st, sorted(FAMILY_OF), store)
        off = offline_strategies(recs, primary)
        per[st] = (recs, off)
        k = len(off["classes"])
        print("  {}: n={} disagreement={} ({:.1%})".format(
            st, len(off["y"]), int(off["disagree"].sum()), off["disagree"].mean()))
        if args.offline:
            y = off["y"]
            best = max(off["single"], key=lambda c: off["single"][c]["macro_f1"])
            print("    best single {} {:.4f} | soft vote {:.4f} | all-six vote {:.4f} | cascade {:.4f} "
                  "(to encoder {:.1%}, to LLM {:.1%})".format(
                      best, off["single"][best]["macro_f1"],
                      macro_f1_idx(y, off["soft_voting"]["pred"], k),
                      macro_f1_idx(y, off["soft_voting_all"]["pred"], k),
                      macro_f1_idx(y, off["cascade"]["pred"], k),
                      float((off["cascade"]["stage"] >= 1).mean()),
                      float((off["cascade"]["stage"] == 2).mean())))
    if args.plan or args.offline:
        return 0

    out_path = OUT if not args.sensitivity else RESULTS / "e3_combination_sensitivity_{}.json".format(args.judge_model)
    if args.go and out_path.exists() and not args.force:
        print("SKIP: {} exists; re-measure with --force only if you mean to replace it.".format(out_path))
        return 0
    if args.go and not args.sensitivity and args.judge_model != PRIMARY_JUDGE:
        print("REFUSING: the primary judge is fixed by D6a ({}); a different model runs "
              "only with --sensitivity.".format(PRIMARY_JUDGE))
        return 2

    global PACER
    if not args.no_pacing:
        PACER = CallPacer(budget=args.daily_budget)
    from src.gwdg_client import GWDGClient
    client = GWDGClient(args.judge_model, timeout=180.0, max_retries=0)
    models_listing = client.list_models()
    if args.judge_model not in models_listing:
        print("REFUSING: {} is not served right now".format(args.judge_model))
        return 2

    day_probe = day_probe_for(client, per, args) if (args.go and not args.skip_day_probe) else None
    judged = {}
    try:
      for st in SUBTASKS:
          if args.judge_smoke and st != "dbo":
              continue
          recs, off = per[st]
          members = off["members"]
          ids = np.asarray(recs[members[0]]["ids"])
          txt = texts_for(st, ids)
          items = []
          for j in np.flatnonzero(off["disagree"]):
              cands = [(off["classes"][int(recs[c]["proba"][j].argmax())], float(recs[c]["proba"][j].max()))
                       for c in members]
              order = candidate_order(int(ids[j]), len(cands))
              items.append({"id": int(ids[j]), "text": txt[int(ids[j])],
                            "candidates": [cands[o] for o in order]})
          if args.judge_smoke:
              items = items[:args.judge_smoke]
          cache = JudgeCache(args.judge_model, st)
          rows = judge_items(client, st, off["classes"], items, cache, workers=args.workers)
          judged[st] = {"items": items, "rows": {r["id"]: r for r in rows}, "cache": cache}
          if args.judge_smoke:
              y = off["y"]
              idx = {c: i for i, c in enumerate(off["classes"])}
              for it, r in zip(items, rows):
                  j = int(np.flatnonzero(ids == it["id"])[0])
                  print("    id {} gold {} judge {} cands {} text {!r}".format(
                      it["id"], off["classes"][y[j]], r["label"], r["candidates"], it["text"][:70]))
              return 0
    except BudgetSpent as exc:
        print("STOPPED CLEANLY: {} ({} calls this run). The cache holds every answer; "
              "re-run --go after the daily quota resets.".format(exc, PACER.calls))
        return 3

    # drift probe: the first N judged items per subtask, called again now
    drift = {}
    for st, J in judged.items():
        answered = [it for it in J["items"] if J["rows"][it["id"]].get("label") is not None]
        probe = sorted(answered, key=lambda it: it["id"])[:N_DRIFT_PROBE]
        again = judge_items(client, st, per[st][1]["classes"], probe, J["cache"],
                            workers=args.workers, kind="drift_probe")
        pairs = [(J["rows"][it["id"]]["label"], r["label"]) for it, r in zip(probe, again)
                 if r["label"] is not None]
        drift[st] = {"n_probed": len(probe), "n_answered_again": len(pairs),
                     "same_label": sum(1 for a, b in pairs if a == b)}

    assert_judge_answered(judged)
    return write_artefact(args, out_path, primary, per, judged, drift, models_listing)


def write_artefact(args, out_path, primary, per, judged, drift, models_listing) -> int:
    from src import significance as sig
    cells, comparisons = {}, {}
    pred_arrays = {}
    for st in SUBTASKS:
        recs, off = per[st]
        y, k, classes = off["y"], len(off["classes"]), off["classes"]
        best = max(off["single"], key=lambda c: off["single"][c]["macro_f1"])
        J = assemble_judge(off, recs, judged[st]["rows"])
        strategies = {
            "soft_voting": off["soft_voting"]["pred"],
            "cascade": off["cascade"]["pred"],
            "llm_as_judge": J["pred"],
            "soft_voting_all_six": off["soft_voting_all"]["pred"],
        }
        from src.calibration import calibration_report
        cell = {
            "n_items": int(len(y)), "classes": classes, "members": off["members"],
            "mixes_machines": len({recs[c]["meta"].get("machine") for c in off["members"]}) > 1,
            "member_machines": {c: recs[c]["meta"].get("machine") for c in off["members"]},
            "per_component_macro_f1": {c: off["single"][c]["macro_f1"] for c in off["single"]},
            "best_single": best,
            "best_single_macro_f1": off["single"][best]["macro_f1"],
            "strategies": {},
        }
        for name, pred in strategies.items():
            cell["strategies"][name] = {
                "macro_f1": macro_f1_idx(y, pred, k),
                "per_class_f1": dict(zip(classes, per_class_f1_idx(y, pred, k))),
                "delta_vs_best_pp": round(100 * (macro_f1_idx(y, pred, k) - off["single"][best]["macro_f1"]), 2),
            }
            pred_arrays["{}__{}".format(st, name)] = pred.astype(np.int8)
        cell["strategies"]["soft_voting"]["calibration"] = calibration_report(
            off["soft_voting"]["proba"], [classes[i] for i in y], classes)
        stage = off["cascade"]["stage"]
        cell["strategies"]["cascade"].update({
            "thresholds_per_fold": off["cascade"]["thresholds"],
            "share_decided_by": {"tml": float((stage == 0).mean()), "encoder": float((stage == 1).mean()),
                                 "llm": float((stage == 2).mean())},
            "escalation_rate": {"to_encoder": float((stage >= 1).mean()), "to_llm": float((stage == 2).mean())},
            "inference_calls_per_1000_items": {"tml": 1000.0, "encoder": round(1000 * float((stage >= 1).mean()), 1),
                                               "llm": round(1000 * float((stage == 2).mean()), 1)},
        })
        cell["strategies"]["llm_as_judge"].update({
            "judge_model": args.judge_model,
            "invoked": J["n_invoked"], "invocation_rate": J["n_invoked"] / len(y),
            "fallback_to_soft_vote": J["n_fallback"],
            "answer_outside_candidates": J["n_answer_outside_candidates"],
            "inference_calls_per_1000_items": {"tml": 1000.0, "encoder": 1000.0, "llm": 1000.0,
                                               "judge": round(1000 * J["n_invoked"] / len(y), 1)},
            "drift_probe": drift.get(st),
        })
        dis = off["disagree"]
        cell["disagreement_subset"] = {
            "n": int(dis.sum()),
            "judge_accuracy": float((J["pred"][dis] == y[dis]).mean()) if dis.any() else None,
            "soft_vote_accuracy": float((off["soft_voting"]["pred"][dis] == y[dis]).mean()) if dis.any() else None,
            "best_single_accuracy": float((off["single"][best]["pred"][dis] == y[dis]).mean()) if dis.any() else None,
            "any_member_correct": float((np.stack([recs[c]["proba"].argmax(1) for c in off["members"]])[:, dis]
                                         == y[dis]).any(0).mean()) if dis.any() else None,
        }
        base = recs[best]
        comparisons[st] = []
        for name, pred in strategies.items():
            r = sig.compare(base, as_record(base, pred, k), key_a=best, key_b=name)
            r["strategy"], r["primary"] = name, name != "soft_voting_all_six"
            comparisons[st].append(r)
        cells[st] = cell
        pred_arrays["{}__ids".format(st)] = np.asarray(base["ids"])
        pred_arrays["{}__y".format(st)] = y.astype(np.int8)

    mult = sig.apply_multiplicity(comparisons)
    mult["family"] = "every strategy-vs-best-single comparison in this artefact, every subtask"
    g3 = {}
    for st in SUBTASKS:
        helps = []
        for r in comparisons[st]:
            name = r["strategy"]
            cells[st]["strategies"][name]["significance"] = r
            d = cells[st]["strategies"][name]["delta_vs_best_pp"]
            if r["primary"] and d > 0 and r["multiplicity"]["survives_holm_bootstrap"]:
                helps.append(name)
        g3[st] = {"combination_helps": bool(helps), "strategies_that_help": helps}
    artefact = {
        "experiment": "B7/E3 combination layer" + (" -- judge sensitivity" if args.sensitivity else ""),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "protocol": PROTOCOL, "protocol_digest": protocol_digest(),
        "primary_set": primary,
        "judge": {"model": args.judge_model, "served_models_at_start": models_listing,
                  "version_handle": "none exposed by the API (THESIS_LOG #44); one campaign, drift probe below"},
        "cells": cells,
        "multiplicity": mult,
        "g3": g3,
        "day_probes": {st: [dict(id=r["id"], label=r["label"], created=r["created"])
                            for r in _cache_rows(args.judge_model, st, "day_probe")]
                       for st in SUBTASKS},
        "g3_decision": ("yes on {}".format([s for s in SUBTASKS if g3[s]["combination_helps"]])
                        if any(g3[s]["combination_helps"] for s in SUBTASKS) else
                        "no -- no primary strategy is distinguishably above the best single component"),
        "claims_not_made": [
            "one seed per component: a corrected p is about these trained models on these items, not the method",
            "the TML member ran on Darwin/arm64, the neural members on Linux/x86_64",
        ],
    }
    out_path.write_text(json.dumps(artefact, indent=1, default=float) + "\n", encoding="utf-8")
    if not args.sensitivity:
        np.savez_compressed(PRED_OUT, **pred_arrays)
    print("wrote {}; G3: {}".format(out_path, artefact["g3_decision"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
