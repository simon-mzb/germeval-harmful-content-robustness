"""
gwdg_client.py -- thin client for the GWDG ChatAI / SAIA OpenAI-compatible API.

The protocol allocates this service to **judge inference only** (the LLM judge
of the combination layer, E3). The multilingual few-shot component stays local,
because 3.3 commits to Qwen 2.5-14B and serving a different model would change
the model's identity rather than its location.

Credentials are read from `.env` (see `.env.example`) and never printed, never
logged and never written into a result file. The file is gitignored.

Every call records what must travel with a GWDG number:
the model id, the served version string where the API exposes one, and the
timestamp. An external service can update a served model mid-study, which is
why each experiment must run as a single campaign.

Usage
-----
from src.gwdg_client import GWDGClient
client = GWDGClient()
print(client.list_models())
r = client.label_call("Ist dieser Text ein Aufruf zur Handlung?", ["TRUE", "FALSE"])
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_ENV_PATH = Path(__file__).parent.parent / ".env"


def load_env(path: Path = _ENV_PATH) -> dict[str, str]:
    """
    Read KEY=VALUE pairs from .env into os.environ without overwriting.

    A deliberately minimal parser rather than a dependency: the file has two
    keys. Values are not echoed anywhere.
    """
    if not path.exists():
        return {}
    loaded = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value:
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


class MissingCredentials(RuntimeError):
    """Raised when .env is absent or incomplete, with instructions and no secrets."""


@dataclass
class CallRecord:
    """What must travel with every GWDG number."""

    model: str
    created: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    system_fingerprint: str | None = None
    text: str | None = None
    logprobs: list[dict[str, Any]] = field(default_factory=list)
    http_status: int | None = None
    error: str | None = None


class GWDGClient:
    """
    Minimal wrapper over the OpenAI-compatible endpoint.

    Parameters
    ----------
    model : str
        Served model id. Deliberately has no default: the protocol requires the model to
        be named explicitly in every experiment, and a silent default is how a
        study ends up mixing two models.
    """

    def __init__(self, model: str, *, api_key: str | None = None,
                 base_url: str | None = None, timeout: float = 60.0,
                 max_retries: int | None = None):
        load_env()
        self.model = model
        key = api_key or os.environ.get("OPENAI_API_KEY")
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not key or not url:
            raise MissingCredentials(
                "GWDG credentials not found. Copy .env.example to "
                ".env and fill in OPENAI_API_KEY and "
                "OPENAI_BASE_URL. The file is gitignored; do not paste the key "
                "anywhere else."
            )
        from openai import OpenAI

        # max_retries=None keeps the SDK default (2), which RETRIES SILENTLY on a
        # timeout and on HTTP 429 with the server's retry-after. For a quota-
        # bound campaign that hides exactly what must be logged,
        # so e3_combination passes 0 and does its own, recorded backoff.
        extra = {} if max_retries is None else {"max_retries": max_retries}
        self._client = OpenAI(api_key=key, base_url=url, timeout=timeout, **extra)

    # -- introspection -------------------------------------------------------

    def list_models(self) -> list[str]:
        """Model ids the endpoint serves. Answers which judge families exist."""
        return sorted(m.id for m in self._client.models.list().data)

    # -- calls ---------------------------------------------------------------

    def label_call(
        self,
        prompt: str,
        labels: Sequence[str],
        *,
        system: str | None = None,
        logprobs: bool = True,
        top_logprobs: int = 5,
        max_tokens: int = 8,
    ) -> CallRecord:
        """
        One judge-style call whose output should be a bare label.

        `logprobs` is requested because D6a criterion 2 turns on whether the
        endpoint returns them: without a usable confidence signal the judge can
        only return a label, and the calibration claim in 3.2 has to be
        narrowed to the components. The request degrades gracefully, so a
        server that rejects the parameter still yields the text.
        """
        instruction = (
            system
            or "You are a strict classifier. Answer with exactly one of the "
               "following labels and nothing else: " + ", ".join(labels) + "."
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": instruction},
                         {"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs

        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if logprobs:
                return self.label_call(prompt, labels, system=system,
                                       logprobs=False, max_tokens=max_tokens)
            return CallRecord(model=self.model, latency_s=time.perf_counter() - start,
                              created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              error="{}: {}".format(type(exc).__name__, exc)[:300])
        return self._to_record(resp, time.perf_counter() - start)

    def _to_record(self, resp: Any, latency: float) -> CallRecord:
        choice = resp.choices[0]
        lp: list[dict[str, Any]] = []
        raw = getattr(choice, "logprobs", None)
        for tok in (getattr(raw, "content", None) or []):
            lp.append({
                "token": tok.token,
                "logprob": tok.logprob,
                "top": [{"token": t.token, "logprob": t.logprob}
                        for t in (getattr(tok, "top_logprobs", None) or [])],
            })
        usage = getattr(resp, "usage", None)
        return CallRecord(
            model=getattr(resp, "model", self.model),
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            latency_s=latency,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            system_fingerprint=getattr(resp, "system_fingerprint", None),
            text=(choice.message.content or "").strip(),
            logprobs=lp,
        )

    # -- rate limits ---------------------------------------------------------

    def probe_throughput(self, prompt: str, labels: Sequence[str],
                         n_calls: int = 12) -> dict[str, Any]:
        """
        A **bounded** throughput probe: n short calls back to back.

        D6a asks for requests per minute and per day before throttling. The
        per-day figure is deliberately *not* found by exhaustion -- hammering a
        shared academic service until it refuses is both rude and a good way to
        lose the account. What is measured is achieved throughput over a small
        burst plus any 429 encountered; the documented quota is read from the
        response headers or the service documentation and recorded separately.
        """
        latencies, errors, throttled = [], 0, False
        start = time.perf_counter()
        for _ in range(n_calls):
            rec = self.label_call(prompt, labels, logprobs=False, max_tokens=4)
            if rec.error:
                errors += 1
                if "429" in rec.error or "rate" in rec.error.lower():
                    throttled = True
                    break
            else:
                latencies.append(rec.latency_s)
        elapsed = time.perf_counter() - start
        done = len(latencies)
        return {
            "n_requested": n_calls,
            "n_succeeded": done,
            "n_errors": errors,
            "throttled_within_burst": throttled,
            "elapsed_s": elapsed,
            "achieved_requests_per_minute": (done / elapsed * 60) if elapsed > 0 else None,
            "latency_mean_s": (sum(latencies) / done) if done else None,
            "latency_min_s": min(latencies) if latencies else None,
            "latency_max_s": max(latencies) if latencies else None,
            "note": ("sequential burst, not a saturation test; the per-day quota "
                     "is taken from headers or service documentation, not probed"),
        }
