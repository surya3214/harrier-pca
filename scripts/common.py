"""Shared helpers for the air-gapped Harrier → mMiniLM PCA distillation pipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

PROMPTS = {
    "sts": "sts_query",
    "retrieval_query": "web_search_query",
    "document": "",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else repo_root() / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_bundle_dir(cfg: dict[str, Any], override: str | None = None) -> Path:
    raw = override or cfg.get("bundle_dir", "offline_bundle")
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def bundle_paths(bundle_dir: Path) -> dict[str, Path]:
    paths = {
        "bundle": bundle_dir,
        "models": bundle_dir / "models",
        "teacher": bundle_dir / "models" / "harrier-oss-v1-0.6b",
        "student": bundle_dir / "models" / "mMiniLMv2-L12-H384",
        "datasets": bundle_dir / "datasets",
        "corpus": bundle_dir / "datasets" / "corpus",
        "stsb": bundle_dir / "datasets" / "stsb",
        "nanobeir": bundle_dir / "datasets" / "NanoBEIR-en",
        "artifacts": bundle_dir / "artifacts",
        "pca": bundle_dir / "artifacts" / "pca_384.npz",
        "train_mse": bundle_dir / "artifacts" / "train_mse",
        "eval_mse": bundle_dir / "artifacts" / "eval_mse",
        "outputs": bundle_dir / "outputs",
        "student_final": bundle_dir / "outputs" / "student-final",
        "logs": bundle_dir / "logs",
        "manifest": bundle_dir / "MANIFEST.json",
    }
    return paths


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path | None = None, name: str = "harrier-pca") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "fsspec", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger(name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_offline_env() -> None:
    """Force Hub/datasets/transformers to stay local (GPU air-gap)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def l2_normalize(vectors):
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms
