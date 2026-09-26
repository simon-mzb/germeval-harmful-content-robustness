"""
data_loading.py -- canonical GermEval 2025/2026 data loader.

All notebooks and experiments import from here. Never read CSVs directly.

Usage
-----
from src.data_loading import load_split, load_subtask, SUBTASKS, EDITIONS, SPLITS

df = load_split("c2a", "2025", "train")
dfs = load_subtask("dbo", "2025")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.data_manifest import ensure_verified

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBTASKS = ("c2a", "dbo", "vio")
SUBTASKS_2026 = ("c2a", "dbo", "vio", "def")
EDITIONS = ("2025", "2026")
SPLITS = ("train", "test", "trial")

_DATA_ROOT = Path(__file__).parent.parent / "data" / "codabench"

_EDITION_DIR = {
    "2025": _DATA_ROOT / "GermEval2025" / "data",
    "2026": _DATA_ROOT / "GermEval2026" / "data",
}

_FILENAME_PATTERNS = {
    ("2025", "c2a", "train"):  "c2a/c2a_train.csv",
    ("2025", "c2a", "test"):   "c2a/c2a_test.csv",
    ("2025", "c2a", "trial"):  "c2a/c2a_trial.csv",
    ("2025", "dbo", "train"):  "dbo/dbo_train.csv",
    ("2025", "dbo", "test"):   "dbo/dbo_test.csv",
    ("2025", "dbo", "trial"):  "dbo/dbo_trial.csv",
    ("2025", "vio", "train"):  "vio/vio_train.csv",
    ("2025", "vio", "test"):   "vio/vio_test.csv",
    ("2025", "vio", "trial"):  "vio/vio_trial.csv",
    ("2026", "c2a", "train"):  "c2a/c2a_train_26.csv",
    ("2026", "c2a", "test"):   "c2a/c2a_test_26.csv",
    ("2026", "c2a", "trial"):  "c2a/c2a_trial.csv",
    ("2026", "dbo", "train"):  "dbo/dbo_train_26.csv",
    ("2026", "dbo", "test"):   "dbo/dbo_test_26.csv",
    ("2026", "dbo", "trial"):  "dbo/dbo_trial.csv",
    ("2026", "vio", "train"):  "vio/vio_train_26.csv",
    ("2026", "vio", "test"):   "vio/vio_test_26.csv",
    ("2026", "vio", "trial"):  "vio/vio_trial.csv",
    ("2026", "def", "train"):  "def/def_train.csv",
    ("2026", "def", "test"):   "def/def_test.csv",
    ("2026", "def", "trial"):  "def/def_trial.csv",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_columns(df, subtask):
    df.columns = [c.lower() for c in df.columns]
    label_col = subtask.lower()
    if label_col in df.columns:
        df = df.rename(columns={label_col: "label"})
    return df


def _fix_known_typos(df, subtask):
    if subtask == "vio" and "label" in df.columns:
        df["label"] = df["label"].str.replace(r"^prospensity$", "propensity", regex=True)
    return df


def _cast_labels(df, subtask):
    """
    Cast label column to consistent dtype.
    Auto-detected: if all non-null values are in {TRUE, FALSE}, cast to bool.
    Multi-class (dbo, vio-2026): stay as string.
    """
    if "label" not in df.columns:
        return df
    non_null = df["label"].dropna()
    if len(non_null) == 0:
        return df
    unique_vals = set(non_null.unique())
    if unique_vals <= {"TRUE", "FALSE"}:
        df = df.copy()
        df["label"] = df["label"].map({"TRUE": True, "FALSE": False})
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_split(subtask, edition, split, *, fix_typos=True, cast_labels=True):
    """
    Load a single split as a tidy DataFrame.
    Columns: id (Int64), description (str), label (if labelled).

    Verifies the raw data against results/data_manifest.json on the first call
    of a process (see src/data_manifest.py); the raw CSVs are gitignored, so
    this is the only record that they have not changed under our feet.
    """
    ensure_verified()
    key = (edition, subtask.lower(), split)
    if key not in _FILENAME_PATTERNS:
        raise ValueError(
            "No data file for subtask={!r}, edition={!r}, split={!r}.".format(
                subtask, edition, split
            )
        )
    path = _EDITION_DIR[edition] / _FILENAME_PATTERNS[key]
    if not path.exists():
        raise FileNotFoundError("Expected data file not found: {}".format(path))

    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    df = _normalize_columns(df, subtask)

    if fix_typos:
        df = _fix_known_typos(df, subtask)
    if cast_labels:
        df = _cast_labels(df, subtask)

    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

    df.attrs.update({"subtask": subtask, "edition": edition, "split": split})
    return df


def load_subtask(subtask, edition, splits=SPLITS, **kwargs):
    """Load all (or specified) splits for a subtask/edition."""
    result = {}
    for split in splits:
        try:
            result[split] = load_split(subtask, edition, split, **kwargs)
        except (ValueError, FileNotFoundError):
            pass
    return result


def load_all_train(edition):
    """Load all train splits for an edition."""
    subtasks = SUBTASKS_2026 if edition == "2026" else SUBTASKS
    return {st: load_split(st, edition, "train") for st in subtasks}
