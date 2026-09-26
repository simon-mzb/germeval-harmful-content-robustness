"""
gwdg_smoke.py -- the acceptance test for GWDG ChatAI / SAIA, which chose the judge model.

The protocol states three criteria and says which one to check first:

  1. one judge-style call returns output restricted to the label set and
     parseable without free-text recovery;
  2. token log-probabilities (or a usable confidence proxy) are available --
     **checked first, because it is the one that can force a design change**:
     without a confidence signal the judge returns a label only and the
     calibration claim in 3.2 narrows to the components;
  3. measured requests per minute and the latency of a single call.

The verdict goes to `results/gwdg_smoke.json`, together with what must
accompany any GWDG number: the model id, the served version string where
the API exposes one, and the date of the calls.

Credentials are not included. Copy `.env.example` to `.env`, fill in the token
and the base URL, then run this. Nothing here prints or stores the key.

Usage
-----
python -m src.gwdg_smoke --list-models
python -m src.gwdg_smoke --model <served-model-id>
python -m src.gwdg_smoke --model <id> --n-throughput 20
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.gwdg_client import GWDGClient, MissingCredentials

RESULTS = Path(__file__).parent.parent / "results" / "gwdg_smoke.json"

# A judge is asked to arbitrate between components on one item. These prompts
# mirror that shape rather than a generic chat turn, because what matters is
# whether *this* usage yields a parseable label.
JUDGE_LABELS = ["TRUE", "FALSE"]
JUDGE_PROMPTS = [
    "Text: \"Wir muessen endlich auf die Strasse gehen und das stoppen!\"\n"
    "Frage: Enthaelt der Text einen Aufruf zum Handeln?",
    "Text: \"Das Wetter in Regensburg ist heute wirklich angenehm.\"\n"
    "Frage: Enthaelt der Text einen Aufruf zum Handeln?",
    "Text: \"Unterschreibt die Petition, bevor es zu spaet ist.\"\n"
    "Frage: Enthaelt der Text einen Aufruf zum Handeln?",
]


def criterion_2_logprobs(client: GWDGClient) -> dict[str, Any]:
    """Checked first: does the endpoint return a usable confidence signal?"""
    rec = client.label_call(JUDGE_PROMPTS[0], JUDGE_LABELS, logprobs=True)
    has = bool(rec.logprobs)
    top = rec.logprobs[0]["top"] if has and rec.logprobs[0].get("top") else []
    return {
        "criterion": "2 - token log-probabilities available",
        "passed": has,
        "n_tokens_with_logprobs": len(rec.logprobs),
        "top_alternatives_on_first_token": len(top),
        "first_token": rec.logprobs[0]["token"] if has else None,
        "error": rec.error,
        "consequence_if_failed": (
            "the judge returns a label only; narrow the calibration claim in "
            "3.2 to the components and record the deviation in 3.3"
        ),
    }


def criterion_1_parseable(client: GWDGClient) -> dict[str, Any]:
    """Is the output inside the label set without free-text recovery?"""
    calls, clean = [], 0
    for prompt in JUDGE_PROMPTS:
        rec = client.label_call(prompt, JUDGE_LABELS, logprobs=False)
        text = (rec.text or "").strip().strip(".").upper()
        ok = text in JUDGE_LABELS
        clean += int(ok)
        calls.append({"raw_output": rec.text, "parseable": ok,
                      "latency_s": rec.latency_s, "error": rec.error})
    return {
        "criterion": "1 - output restricted to the label set",
        "passed": clean == len(JUDGE_PROMPTS),
        "n_clean": clean,
        "n_calls": len(JUDGE_PROMPTS),
        "calls": calls,
    }


def criterion_3_limits(client: GWDGClient, n: int) -> dict[str, Any]:
    probe = client.probe_throughput(JUDGE_PROMPTS[0], JUDGE_LABELS, n_calls=n)
    probe["criterion"] = "3 - measured throughput and latency"
    probe["passed"] = probe["n_succeeded"] > 0
    return probe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D6a acceptance test for GWDG ChatAI.")
    ap.add_argument("--model", default=None, help="served model id (required to run the test)")
    ap.add_argument("--list-models", action="store_true",
                    help="list what the endpoint serves and exit")
    ap.add_argument("--n-throughput", type=int, default=12,
                    help="calls in the bounded throughput burst (default 12)")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)

    try:
        client = GWDGClient(args.model or "unused-for-listing")
    except MissingCredentials as exc:
        print(exc)
        return 2

    if args.list_models:
        for m in client.list_models():
            print(" ", m)
        print("\nPick a judge from a different model family than the LLM "
              "component, then rerun with --model <id>.")
        return 0

    if not args.model:
        ap.error("--model is required; run --list-models first")

    print("D6a smoke test against model {!r}".format(args.model))
    print("  criterion 2 (log-probabilities) first: it is the one that can "
          "force a design change\n")

    c2 = criterion_2_logprobs(client)
    print("  [2] log-probabilities: {}".format("PASS" if c2["passed"] else "FAIL"))
    c1 = criterion_1_parseable(client)
    print("  [1] parseable labels:  {} ({}/{})".format(
        "PASS" if c1["passed"] else "FAIL", c1["n_clean"], c1["n_calls"]))
    c3 = criterion_3_limits(client, args.n_throughput)
    rpm = c3.get("achieved_requests_per_minute")
    print("  [3] throughput:        {:.1f} req/min, mean latency {:.2f}s".format(
        rpm or 0.0, c3.get("latency_mean_s") or 0.0))

    payload = {
        "gate": "D6a",
        "model_requested": args.model,
        "model_reported": c1["calls"][0].get("raw_output") and args.model,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": "GWDG ChatAI / SAIA (OpenAI-compatible)",
        "criterion_2_logprobs": c2,
        "criterion_1_parseable": c1,
        "criterion_3_limits": c3,
        "all_passed": bool(c2["passed"] and c1["passed"] and c3["passed"]),
        "decision": "TODO - go / adjust / drop the judge (D6a, human judgement)",
        "decision_note": "TODO",
        "note": ("Credentials are read from .env and are neither printed nor "
                 "stored here. Per-day quota is not probed by exhaustion; take "
                 "it from the service documentation."),
    }

    out = Path(args.out)
    if out.exists():
        prior = json.loads(out.read_text(encoding="utf-8"))
        for key in ("decision", "decision_note"):
            value = prior.get(key)
            if isinstance(value, str) and not value.strip().startswith("TODO"):
                payload[key] = value
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n  wrote {}".format(out))
    print("  verdict: {}".format("all criteria met" if payload["all_passed"]
                                 else "at least one criterion failed - see the JSON"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
