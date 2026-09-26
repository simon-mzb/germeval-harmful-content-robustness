"""
data_manifest.py -- provenance guard for the GermEval CSVs.

The raw data under ``data/codabench/`` is deliberately gitignored (17.8 MB of
bulk), so git records nothing about it. Every number in the thesis depends on
those files, which means a silent change -- a re-download, a partial sync, an
editor rewriting line endings -- would move results with no trace. This module
closes that gap: it records a sha256 per file once, and re-checks on every run.

The check is cheap (~17 ms for all 21 files), so it runs automatically on the
first ``load_split`` call of a process rather than waiting to be remembered.

Usage
-----
    python -m src.data_manifest --write     # create/refresh the manifest
    python -m src.data_manifest --verify    # explicit check, prints a report

    from src.data_manifest import ensure_verified, verify_manifest
    ensure_verified()          # no-op after the first call in a process

Escape hatch: set ``GERMEVAL_SKIP_MANIFEST=1`` to disable the automatic check.
Refreshing the manifest is a deliberate act -- if ``--write`` changes a hash,
the data changed, and any result produced before that point is stale.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

_EXPERIMENTS_ROOT = Path(__file__).parent.parent
_DATA_ROOT = _EXPERIMENTS_ROOT / "data" / "codabench"
DEFAULT_MANIFEST_PATH = _EXPERIMENTS_ROOT / "results" / "data_manifest.json"

_CHUNK = 1048576


class DataManifestError(RuntimeError):
    """Raised when a tracked data file no longer matches its recorded hash."""


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _iter_csv_paths(data_root=_DATA_ROOT):
    return sorted(data_root.rglob("*.csv"))


def _file_record(path):
    """sha256 + size + line count, read once in a single streaming pass."""
    digest = hashlib.sha256()
    size = 0
    newlines = 0
    tail = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            newlines += chunk.count(b"\n")
            tail = chunk[-1:]
    # A file not ending in a newline still has content on its final line.
    n_lines = newlines + (1 if size > 0 and tail != b"\n" else 0)
    return {"sha256": digest.hexdigest(), "size_bytes": size, "n_lines": n_lines}


def build_manifest(data_root=_DATA_ROOT):
    paths = _iter_csv_paths(data_root)
    if not paths:
        raise FileNotFoundError("No CSV files found under {}".format(data_root))
    files = {}
    for path in paths:
        files[path.relative_to(data_root).as_posix()] = _file_record(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root": data_root.relative_to(_EXPERIMENTS_ROOT).as_posix(),
        "n_files": len(files),
        "note": (
            "Provenance guard for gitignored raw data. Regenerate only when the "
            "data deliberately changes; a changed hash invalidates every result "
            "produced before the change. n_lines counts physical file lines, "
            "NOT records: descriptions contain embedded newlines, so n_lines "
            "exceeds the row count (e.g. dbo_train.csv: 9618 lines, 7454 rows)."
        ),
        "files": files,
    }


def write_manifest(manifest_path=DEFAULT_MANIFEST_PATH, data_root=_DATA_ROOT):
    manifest = build_manifest(data_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------------------
# Verifying
# ---------------------------------------------------------------------------

def verify_manifest(manifest_path=DEFAULT_MANIFEST_PATH, data_root=_DATA_ROOT):
    """
    Compare the data tree against the manifest.

    Returns a report dict with ``changed`` (hash mismatch -- the dangerous
    case), ``missing`` (recorded but absent) and ``untracked`` (present but not
    recorded). Does not raise; ``ensure_verified`` decides on severity.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            "No data manifest at {}. Create it with: "
            "python -m src.data_manifest --write".format(manifest_path)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("files", {})

    on_disk = {
        p.relative_to(data_root).as_posix(): p for p in _iter_csv_paths(data_root)
    }

    changed, missing = [], []
    for rel, want in recorded.items():
        path = on_disk.get(rel)
        if path is None:
            missing.append(rel)
            continue
        got = _file_record(path)
        if got["sha256"] != want["sha256"]:
            changed.append(
                {
                    "file": rel,
                    "expected_sha256": want["sha256"],
                    "actual_sha256": got["sha256"],
                    "expected_n_lines": want.get("n_lines"),
                    "actual_n_lines": got["n_lines"],
                }
            )

    untracked = sorted(set(on_disk) - set(recorded))
    return {
        "ok": not changed and not missing,
        "manifest_created_utc": manifest.get("created_utc"),
        "n_recorded": len(recorded),
        "changed": changed,
        "missing": missing,
        "untracked": untracked,
    }


_verified = False


def ensure_verified(manifest_path=DEFAULT_MANIFEST_PATH, data_root=_DATA_ROOT):
    """
    Verify once per process. Raises on a changed file, warns on missing or
    untracked ones.

    Severity is deliberately asymmetric. A changed hash is silent corruption
    and must stop the run. An absent file is already loud -- the loader raises
    FileNotFoundError with a clear message -- so a warning suffices.
    """
    global _verified
    if _verified or os.environ.get("GERMEVAL_SKIP_MANIFEST") == "1":
        return None

    report = verify_manifest(manifest_path, data_root)

    if report["changed"]:
        lines = [
            "Data files changed since the manifest was written "
            "({}).".format(report["manifest_created_utc"]),
            "Every result produced before this point is stale.",
            "",
        ]
        for entry in report["changed"]:
            lines.append(
                "  {}: {} lines -> {} lines".format(
                    entry["file"], entry["expected_n_lines"], entry["actual_n_lines"]
                )
            )
        lines.append("")
        lines.append(
            "If the change is deliberate, refresh with "
            "'python -m src.data_manifest --write', then regenerate all results."
        )
        raise DataManifestError("\n".join(lines))

    if report["missing"]:
        warnings.warn(
            "Data files recorded in the manifest are absent: {}".format(
                ", ".join(report["missing"])
            ),
            stacklevel=2,
        )
    if report["untracked"]:
        warnings.warn(
            "Data files present but not in the manifest: {}. Refresh with "
            "'python -m src.data_manifest --write' if intended.".format(
                ", ".join(report["untracked"])
            ),
            stacklevel=2,
        )

    _verified = True
    return report


def _main(argv):
    if "--write" in argv:
        manifest = write_manifest()
        print("Wrote manifest for {} files -> {}".format(
            manifest["n_files"], DEFAULT_MANIFEST_PATH
        ))
        return 0
    if "--verify" in argv:
        report = verify_manifest()
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
