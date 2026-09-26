"""
baselines.py -- organizer baseline reimplementations for E1 reproduction.

Reproduces the three GermEval 2025 organizer baselines as closely as possible
in a unified Python interface, for integration into our shared evaluation
harness (harness.py).

Baselines
---------
C2A  GradientBoosting + SentenceBERT embeddings (distiluse-base-multilingual-
     cased-v2) + TextBlob-DE polarity + random undersampling.
     Target: macro-F1 = 59.13 (organizer reported).
     Source: baseline/c2a/c2a_baseline.ipynb

DBO  TF-IDF (unigrams+bigrams, max 5000 features) + LinearSVC with balanced
     class weights.
     Target: macro-F1 = 47.44 (organizer reported).
     Source: baseline/dbo/dbo_baseline.ipynb

VIO  [Option C — not reproduced identically]
     The organizer baseline is Qwen2.5-32B few-shot via Ollama in R, which
     requires a 32B model not available in our Python stack. Organizer-reported
     macro-F1 = 68.97 is used as the reference figure and attributed to the
     original implementation (felser2025). An approximate Python reimplementation
     via a GWDG-API Qwen model is planned for E1b (post-E1 smoke test) and will
     be reported separately with the deviation documented.

Notes
-----
- All randomness uses random_state=42 to match organizer notebooks.
- Lemmatisation in C2A requires the spaCy German pipeline:
    python -m spacy download de_core_news_md
- C2A also requires sentence-transformers and textblob-de.
- Run `uv sync` in experiments/ before executing (sandbox has no network).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import compute_class_weight, resample

from src.preprocessing import clean_c2a, clean_dbo


# ---------------------------------------------------------------------------
# DBO baseline (pure sklearn — no heavy deps)
# ---------------------------------------------------------------------------

class DBOBaseline:
    """
    DBO organizer baseline: TF-IDF (1-2gram, max 5000) + LinearSVC.

    Closely follows baseline/dbo/dbo_baseline.ipynb.
    Class imbalance handled via balanced class weights passed to SVC.

    Parameters
    ----------
    max_features : int
        TF-IDF vocabulary size (default 5000, matching organizer).
    random_state : int
        SVC random seed (default 42).
    """

    # Label order used by the organizer notebook
    LABEL_ORDER = ["agitation", "criticism", "nothing", "subversive"]

    def __init__(self, max_features: int = 5000, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer_: TfidfVectorizer | None = None
        self.model_: LinearSVC | None = None
        self.label_order_: list[str] = self.LABEL_ORDER

    def fit(self, train_df: pd.DataFrame) -> "DBOBaseline":
        """Fit TF-IDF vectoriser + LinearSVC on cleaned training data."""
        texts = clean_dbo(train_df["description"])
        labels = train_df["label"]

        self.vectorizer_ = TfidfVectorizer(ngram_range=(1, 2), max_features=self.max_features)
        X = self.vectorizer_.fit_transform(texts)

        # Balanced class weights (organizer: compute_class_weight 'balanced')
        classes = np.array(self.label_order_)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=labels.values,
        )
        class_weight_dict = {c: w for c, w in zip(classes, class_weights)}

        self.model_ = LinearSVC(
            class_weight=class_weight_dict,
            random_state=self.random_state,
            max_iter=2000,
        )
        self.model_.fit(X, labels)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict labels for df['description']."""
        assert self.vectorizer_ is not None and self.model_ is not None, \
            "Call fit() first."
        texts = clean_dbo(df["description"])
        X = self.vectorizer_.transform(texts)
        return self.model_.predict(X)

    # Convenience wrappers for harness.run_cv
    def as_train_fn(self):
        def train_fn(train_df):
            m = DBOBaseline(
                max_features=self.max_features,
                random_state=self.random_state,
            )
            return m.fit(train_df)
        return train_fn

    @staticmethod
    def predict_fn(model: "DBOBaseline", val_df: pd.DataFrame) -> np.ndarray:
        return model.predict(val_df)


# ---------------------------------------------------------------------------
# C2A baseline (requires sentence-transformers, spaCy, textblob-de)
# ---------------------------------------------------------------------------

class C2ABaseline:
    """
    C2A organizer baseline: GradientBoosting + SentenceBERT + TextBlob polarity.

    Closely follows baseline/c2a/c2a_baseline.ipynb.
    Undersamples the majority class to match minority size (random_state=42).

    Heavy dependencies (loaded lazily so the module can be imported without them):
        sentence_transformers  -- SentenceTransformer('distiluse-base-multilingual-cased-v2')
        spacy                  -- de_core_news_md (lemmatisation)
        textblob_de            -- TextBlobDE (polarity)

    Parameters
    ----------
    sbert_model : str
        SentenceBERT model name (default matches organizer).
    random_state : int
        Seed for undersampling + GradientBoosting (default 42).
    """

    SBERT_MODEL = "distiluse-base-multilingual-cased-v2"

    def __init__(self, sbert_model: str = SBERT_MODEL, random_state: int = 42):
        self.sbert_model = sbert_model
        self.random_state = random_state
        self._nlp = None       # loaded lazily
        self._sbert = None     # loaded lazily
        self.model_: GradientBoostingClassifier | None = None

    # -- lazy loaders --------------------------------------------------------

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("de_core_news_md")
        return self._nlp

    def _get_sbert(self):
        if self._sbert is None:
            # REFUSE RATHER THAN SEGFAULT. `harness.env_pin()` imports
            # lightgbm and torch; loading SentenceBERT afterwards pulls a SECOND
            # OpenMP runtime into the same process and the process dies with
            # SIGSEGV -- no traceback, no artefact, exit 139. A refusal that
            # names the cause is worth more than a docstring nobody reads at
            # the moment they need it.
            from src import harness as _h

            if getattr(_h, "_ENV_PIN_CALLED", False):
                raise RuntimeError(
                    "SentenceBERT cannot be loaded after harness.env_pin() in the "
                    "same process: env_pin imports torch and lightgbm, and a "
                    "second OpenMP runtime segfaults the interpreter. "
                    "Move the env_pin() call AFTER every model run -- note that a "
                    "dict literal evaluates its values top to bottom, which is "
                    "how proxy_2026 broke this.")
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self.sbert_model)
        return self._sbert

    # -- feature extraction --------------------------------------------------

    def _lemmatize(self, texts: list[str]) -> list[str]:
        nlp = self._get_nlp()
        result = []
        for doc in nlp.pipe(texts, batch_size=50):
            tokens = [t.lemma_.lower() for t in doc if not t.is_punct]
            result.append(" ".join(tokens))
        return result

    def _polarity(self, texts: list[str]) -> np.ndarray:
        from textblob_de import TextBlobDE
        return np.array([TextBlobDE(t).sentiment.polarity for t in texts])

    def _embeddings(self, texts: list[str]) -> np.ndarray:
        sbert = self._get_sbert()
        return sbert.encode(texts, show_progress_bar=False)

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return feature matrix: [polarity (1) | embeddings (512)]."""
        cleaned = clean_c2a(df["description"]).tolist()
        lemmatized = self._lemmatize(cleaned)
        polarity = self._polarity(lemmatized).reshape(-1, 1)
        embeddings = self._embeddings(lemmatized)
        return np.hstack([polarity, embeddings])

    # -- fit/predict ---------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "C2ABaseline":
        """
        Undersample majority class, extract features, fit GradientBoosting.
        Undersampling matches organizer: resample majority to len(minority).
        """
        minority = train_df[train_df["label"] == True]
        majority = train_df[train_df["label"] == False]
        majority_down = resample(
            majority,
            replace=False,
            n_samples=len(minority),
            random_state=self.random_state,
        )
        balanced = pd.concat([minority, majority_down]).reset_index(drop=True)

        X = self._extract_features(balanced)
        y = balanced["label"].values

        self.model_ = GradientBoostingClassifier(random_state=self.random_state)
        self.model_.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict labels for df['description']."""
        assert self.model_ is not None, "Call fit() first."
        X = self._extract_features(df)
        return self.model_.predict(X)

    # Convenience wrappers for harness.run_cv
    def as_train_fn(self):
        def train_fn(train_df):
            m = C2ABaseline(
                sbert_model=self.sbert_model,
                random_state=self.random_state,
            )
            return m.fit(train_df)
        return train_fn

    @staticmethod
    def predict_fn(model: "C2ABaseline", val_df: pd.DataFrame) -> np.ndarray:
        return model.predict(val_df)


# ---------------------------------------------------------------------------
# VIO — Option C stub (not reproduced)
# ---------------------------------------------------------------------------

class VIOBaselineStub:
    """
    Placeholder for the VIO organizer baseline (Option C — not reproduced).

    The organizer baseline (felser2025) uses Qwen2.5-32B via Ollama in R,
    achieving macro-F1 = 68.97 on the 2025 test set. This model and runtime
    are not available in our Python stack; the organizer-reported figure is
    used as the reference and attributed accordingly in §3.3 and §5.

    A separate approximate Python reimplementation via GWDG ChatAI API
    (Qwen3 family) is planned as E1b for comparison, and will be reported
    with the model difference and deviation explicitly documented.

    This stub allows the E1 notebook to reference all three subtasks
    without runtime errors.
    """

    REPORTED_MACRO_F1 = 68.97
    SOURCE = "felser2025"
    NOTE = (
        "VIO baseline not reproduced: organizer used Qwen2.5-32B/Ollama/R. "
        "Reference figure 68.97 from felser2025. "
        "Approximate Python reimplementation (GWDG Qwen3 API) planned as E1b."
    )

    def fit(self, train_df: pd.DataFrame) -> "VIOBaselineStub":
        raise NotImplementedError(self.NOTE)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(self.NOTE)


# ---------------------------------------------------------------------------
# VIO -- E1b: the organisers' Qwen2.5-32B few-shot baseline, served locally
# ---------------------------------------------------------------------------
#
# Ollama serves `qwen2.5:32b` on one rented A40 GPU. Without a reproduced VIO
# baseline the E2 delta on vio does not exist at all (`baseline_comparison`
# withholds it).
#
# FIDELITY RULE: the request is built field for field from
# `data/codabench/GermEval2025/baseline/vio/vio_baseline.Rmd`, and the prompts
# are READ from that file rather than copied here -- two copies of a prompt
# drift. What the organisers' R code
# actually sent, including what it probably did not intend:
#   * `prompt = str_escape(paste0(user_prompt, text))` -- stringr's regex
#     escaping, applied to prose, so every `.`, `(`, `?` reaches the model with a
#     backslash in front of it;
#   * `modelfile = str_escape(system_prompt)` -- `modelfile` is not a field of
#     Ollama's /api/generate, so the system prompt very likely never reached the
#     model; `probe_request_semantics` measures this instead of assuming;
#   * `seed = "123"` at the top level, where Ollama does not read a seed.
# All three are reproduced as sent. ONE deliberate deviation: `options.seed =
# 123 + trial index`, which makes the run repeatable without changing the
# sampling distribution, and still lets a retry resample the way the unseeded
# original did.

VIO_RMD = (Path(__file__).resolve().parents[1] / "data" / "codabench" / "GermEval2025"
           / "baseline" / "vio" / "vio_baseline.Rmd")
VIO_OLLAMA_MODEL = "qwen2.5:32b"
VIO_MAX_TRIALS = 10
VIO_BASE_SEED = 123


def _r_unescape(x: str) -> str:
    import re
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), x)


def load_vio_prompts(path: Path = VIO_RMD) -> tuple[str, str]:
    """(system_prompt_vio, user_prompt_vio) as the R interpreter would hold them."""
    import re
    s = Path(path).read_text(encoding="utf-8")
    m1 = re.search(r'system_prompt_vio <- "((?:[^"\\]|\\.)*)"', s)
    m2 = re.search(r"user_prompt_vio <- '((?:[^'\\]|\\.)*)'", s)
    if not (m1 and m2):
        raise ValueError("could not find both VIO prompt literals in {}".format(path))
    return _r_unescape(m1.group(1)), _r_unescape(m2.group(1))


def r_str_escape(s: str) -> str:
    """stringr::str_escape: a backslash before each of . ^ $ \\ | * + ? { } [ ] ( )."""
    import re
    return re.sub(r"([.^$\\|*+?{}\[\]()])", r"\\\1", s)


def vio_request_body(text: str, system_prompt: str, user_prompt: str, *,
                     trial: int = 1, seed_option: bool = True) -> dict:
    body = {
        "model": VIO_OLLAMA_MODEL,
        "prompt": r_str_escape(user_prompt + text),
        "format": "json",
        "seed": "123",
        "stream": False,
        "modelfile": r_str_escape(system_prompt),
    }
    if seed_option:
        body["options"] = {"seed": VIO_BASE_SEED + trial - 1}
    return body


def parse_vio_response(response_text: str | None) -> str | None:
    """The organisers' acceptance rule, or None where their loop would retry.

    R: `fromJSON(response)` (a parse error is caught and retried), then
    `str_extract_all(values, or1(c("true","false")))` -- case-sensitive, over
    the parsed VALUES (a JSON boolean becomes "TRUE" and matches nothing) --
    then `if (result %in% possible)`, which errors, and so retries, unless
    exactly one match came back.
    """
    import json
    import re
    try:
        parsed = json.loads(response_text or "")
    except (TypeError, ValueError):
        return None
    leaves: list[str] = []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, bool):
            leaves.append("TRUE" if v else "FALSE")
        elif v is not None:
            leaves.append(str(v))

    walk(parsed)
    hits = [h for leaf in leaves for h in re.findall(r"true|false", leaf)]
    return hits[0] if len(hits) == 1 else None


class VIOBaselineOllama:
    """The VIO organiser baseline against a local Ollama server (E1b).

    Few-shot, so `fit` trains nothing -- exactly as the notebook, which imports
    only the test data. `predict` classifies item by item, sequentially, with
    the notebook's retry-up-to-ten; an item still unparsed after ten trials is
    set to `false`, which is what the organisers did with theirs.
    """

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434", *, timeout: float = 600.0,
                 seed_option: bool = True, log_path: Path | str | None = None,
                 progress_every: int = 100):
        self.url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.seed_option = seed_option
        self.log_path = Path(log_path) if log_path else None
        self.progress_every = progress_every
        self.system_prompt, self.user_prompt = load_vio_prompts()
        self.info_: dict = {}

    def _request(self, path: str, body: dict | None = None) -> dict:
        import json
        import urllib.request
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def serving_info(self) -> dict:
        tags = self._request("/api/tags")
        entry = next((m for m in tags.get("models", []) if m.get("name") == VIO_OLLAMA_MODEL), None)
        show = self._request("/api/show", {"model": VIO_OLLAMA_MODEL}) if entry else {}
        return {
            "ollama_version": self._request("/api/version").get("version"),
            "model": VIO_OLLAMA_MODEL,
            "served": entry is not None,
            "digest": (entry or {}).get("digest"),
            "size_bytes": (entry or {}).get("size"),
            "details": show.get("details"),
            "default_parameters": show.get("parameters"),
        }

    def probe_request_semantics(self, text: str = "Das ist ein Test.") -> dict:
        """Does the organisers' `modelfile` field reach the model? Measured, not assumed."""
        body = vio_request_body(text, self.system_prompt, self.user_prompt, seed_option=True)
        with_system = dict(body, system=self.system_prompt)
        b = self._request("/api/generate", with_system)
        a = self._request("/api/generate", body)
        a2 = self._request("/api/generate", body)
        return {
            "prompt_eval_count_as_sent": a.get("prompt_eval_count"),
            "prompt_eval_count_with_system_field": b.get("prompt_eval_count"),
            "system_prompt_reaches_model_as_sent": (
                None if a.get("prompt_eval_count") is None or b.get("prompt_eval_count") is None
                else a.get("prompt_eval_count") >= b.get("prompt_eval_count")),
            "same_seed_same_response": a.get("response") == a2.get("response"),
            "note": "the system field adds the system prompt's tokens; if the body as sent "
                    "evaluates FEWER tokens, its modelfile field was ignored",
        }

    def fit(self, train_df: pd.DataFrame) -> "VIOBaselineOllama":
        return self

    def classify(self, text: str) -> tuple[bool, dict]:
        trials = []
        for n in range(1, VIO_MAX_TRIALS + 1):
            body = vio_request_body(text, self.system_prompt, self.user_prompt,
                                    trial=n, seed_option=self.seed_option)
            label = None
            try:
                out = self._request("/api/generate", body)
                resp = out.get("response", "")
                label = parse_vio_response(resp)
                trials.append({"response": resp[:200], "prompt_eval_count": out.get("prompt_eval_count"),
                               "eval_count": out.get("eval_count")})
            except Exception as exc:  # noqa: BLE001 -- the notebook's tryCatch
                trials.append({"error": "{}: {}".format(type(exc).__name__, exc)[:200]})
            if label is not None:
                return label == "true", {"n_trials": n, "final": label, "fallback": False, "trials": trials}
        return False, {"n_trials": VIO_MAX_TRIALS, "final": "false", "fallback": True, "trials": trials}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        import json
        preds, n_fallback, n_retried, n_trials = [], 0, 0, 0
        fh = self.log_path.open("a", encoding="utf-8") if self.log_path else None
        try:
            for i, (item_id, text) in enumerate(zip(df["id"], df["description"])):
                pred, rec = self.classify(str(text))
                preds.append(pred)
                n_fallback += rec["fallback"]
                n_retried += rec["n_trials"] > 1
                n_trials += rec["n_trials"]
                if fh:
                    fh.write(json.dumps(dict(rec, id=int(item_id)), ensure_ascii=False) + "\n")
                    fh.flush()
                if self.progress_every and (i + 1) % self.progress_every == 0:
                    print("  vio ollama: {}/{} classified ({} retried, {} fell back)".format(
                        i + 1, len(df), n_retried, n_fallback), flush=True)
        finally:
            if fh:
                fh.close()
        self.info_ = {"n_items": len(preds), "n_fallback_to_false": int(n_fallback),
                      "n_items_retried": int(n_retried), "n_requests": int(n_trials),
                      "max_trials": VIO_MAX_TRIALS, "seed_option": self.seed_option}
        return np.asarray(preds, dtype=bool)
