"""
preprocessing.py -- shared text cleaning for all pipeline components.

Used by baselines (E1) and all E2+ components. Keeps preprocessing
consistent across the entire experiment so differences in results
reflect the models, not the text transformations.

Usage
-----
from src.preprocessing import clean_text, clean_series
"""

from __future__ import annotations

import re

import pandas as pd


# ---------------------------------------------------------------------------
# Core cleaning
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_NUM_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_hashtags: bool = True,
    replace_numbers: bool = False,
    num_token: str = " NUM ",
) -> str:
    """
    Clean a single text string.

    Parameters
    ----------
    text : str
        Raw tweet / social-media text.
    lowercase : bool
        Convert to lowercase (default True).
    remove_urls : bool
        Strip http/https/www URLs (default True).
    remove_mentions : bool
        Strip @mentions (default True).
    remove_hashtags : bool
        Strip #hashtags (default True).
    replace_numbers : bool
        Replace digit sequences with *num_token* (default False).
        The DBO organizer baseline uses this; C2A does not.
    num_token : str
        Replacement token when replace_numbers=True.

    Returns
    -------
    str
        Cleaned text, whitespace-normalised.

    Notes
    -----
    No stemming or lemmatisation here — those are component-specific
    (e.g. the C2A baseline uses spaCy lemmatisation on top of this).
    """
    if not isinstance(text, str):
        text = str(text)

    if lowercase:
        text = text.lower()
    if remove_urls:
        text = _URL_RE.sub("", text)
    if remove_mentions:
        text = _MENTION_RE.sub("", text)
    if remove_hashtags:
        text = _HASHTAG_RE.sub("", text)
    if replace_numbers:
        text = _NUM_RE.sub(num_token, text)

    text = _WS_RE.sub(" ", text).strip()
    return text


def clean_series(
    series: pd.Series,
    **kwargs,
) -> pd.Series:
    """
    Apply clean_text to a pandas Series.

    All keyword arguments are forwarded to clean_text.
    """
    return series.apply(lambda t: clean_text(t, **kwargs))


# ---------------------------------------------------------------------------
# Preset profiles (match organizer baseline configs exactly)
# ---------------------------------------------------------------------------

def clean_c2a(series: pd.Series) -> pd.Series:
    """
    C2A organizer baseline preprocessing profile.

    Removes URLs, mentions, hashtags; does NOT replace numbers or lowercase
    (the organizer notebook lowercases implicitly via lemmatisation only).
    Lemmatisation is handled separately in the C2A baseline class.
    """
    return clean_series(
        series,
        lowercase=False,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=True,
        replace_numbers=False,
    )


def clean_dbo(series: pd.Series) -> pd.Series:
    """
    DBO organizer baseline preprocessing profile.

    Lowercases, removes URLs/mentions/hashtags, replaces numbers with NUM —
    exactly matching the organizer notebook's text_clean function.
    """
    return clean_series(
        series,
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=True,
        replace_numbers=True,
        num_token=" NUM ",
    )


def clean_default(series: pd.Series) -> pd.Series:
    """
    Default profile for our own E2+ components.

    Lowercases and strips noise tokens; no number replacement.
    Suitable for TF-IDF and encoder tokenisers.
    """
    return clean_series(
        series,
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=True,
        replace_numbers=False,
    )


def clean_encoder(series: pd.Series) -> pd.Series:
    """
    Profile for the fine-tuned encoder components (E2).

    Identical to clean_default except that it does **not** lowercase, because
    German capitalises nouns and the ModernGBERT tokeniser is cased --
    e2_matrix.yaml records this as `encoder_1b.lowercase: false` and 3.2 argues
    it. Everything else (URL, mention and hashtag stripping, no number
    replacement) is deliberately the same as the classical components: the
    module contract is that differences between components reflect the models
    rather than the text transformations, so exactly one flag may differ and
    that flag is the one the protocol names.
    """
    return clean_series(
        series,
        lowercase=False,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=True,
        replace_numbers=False,
    )
