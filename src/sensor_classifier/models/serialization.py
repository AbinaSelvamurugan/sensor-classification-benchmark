"""Persist pipeline and load only with explicit trust + artifact digest verification."""
from __future__ import annotations

from pathlib import Path

import joblib

from sensor_classifier.utils.reproducibility import sha256


def save_model(path: Path, model) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    digest = sha256(path)
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def load_model(path: Path, trust: bool = False):
    if not trust:
        raise PermissionError("joblib/pickle can run code. Pass --trust-model only for locally trusted artifacts")
    digest_path = path.with_suffix(".sha256")
    if not digest_path.is_file() or not path.is_file():
        raise FileNotFoundError("Model or SHA256 record missing")
    if sha256(path) != digest_path.read_text(encoding="utf-8").strip():
        raise ValueError("Model SHA256 mismatch — refusing to deserialize")
    return joblib.load(path)
