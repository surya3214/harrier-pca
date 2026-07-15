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
        "teacher_emb": bundle_dir / "artifacts" / "teacher_emb_1024.npy",
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


def limit_blas_threads(n: int = 1) -> None:
    """Avoid PyTorch + OpenMP/MKL deadlocks (e.g. sklearn PCA hang after GPU encode)."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, str(n))


def l2_normalize(vectors):
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    out = vectors / norms
    if not np.isfinite(out).all():
        bad = int((~np.isfinite(out)).sum())
        raise ValueError(f"Non-finite values after L2 normalize ({bad} elements)")
    return out


def assert_finite_embeddings(emb, name: str = "embeddings") -> None:
    import numpy as np

    arr = np.asarray(emb)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan or n_inf:
        raise ValueError(f"{name} has non-finite values (nan={n_nan}, inf={n_inf})")


def label_stats(labels) -> dict:
    import numpy as np

    arr = np.asarray(labels, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1)
    return {
        "shape": list(arr.shape),
        "finite": bool(np.isfinite(arr).all()),
        "nan": int(np.isnan(arr).sum()),
        "inf": int(np.isinf(arr).sum()),
        "norm_mean": float(norms.mean()) if len(norms) else 0.0,
        "norm_min": float(norms.min()) if len(norms) else 0.0,
        "norm_max": float(norms.max()) if len(norms) else 0.0,
        "abs_max": float(np.abs(arr).max()) if arr.size else 0.0,
    }


def encode_with_prompts(model, texts: list[str], prompt_names: list[str], batch_size: int):
    """Encode texts grouped by prompt_name (Harrier requires task prompts)."""
    from collections import defaultdict

    import numpy as np

    logger = logging.getLogger("encode")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(prompt_names):
        groups[name or ""].append(i)

    dim = model.get_embedding_dimension()
    out = np.zeros((len(texts), dim), dtype=np.float32)

    for prompt_name, indices in groups.items():
        batch_texts = [texts[i] for i in indices]
        kwargs = {
            "batch_size": batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": True,
        }
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
            logger.info("Encoding %s texts with prompt_name=%r", f"{len(batch_texts):,}", prompt_name)
        else:
            logger.info("Encoding %s texts with no prompt (documents)", f"{len(batch_texts):,}")
        emb = model.encode(batch_texts, **kwargs)
        out[indices] = emb.astype(np.float32, copy=False)
    return out


def save_mse_datasets(
    texts: list[str],
    reduced,
    eval_mse_size: int,
    train_path,
    eval_path,
) -> tuple[int, int]:
    """Split reduced embeddings into train/eval HF datasets and save_to_disk."""
    import shutil

    import numpy as np
    from datasets import Dataset

    assert_finite_embeddings(reduced, "pca_reduced_labels")
    # Drop empty texts (can destabilize Normalize grads)
    keep = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    if len(keep) < len(texts):
        texts = [texts[i] for i in keep]
        reduced = np.asarray(reduced, dtype=np.float32)[keep]
    if len(texts) <= eval_mse_size + 1:
        raise ValueError(f"Too few non-empty texts after filtering: {len(texts)}")

    train_end = len(texts) - eval_mse_size
    train_ds = Dataset.from_dict(
        {
            "sentence": texts[:train_end],
            "label": np.asarray(reduced[:train_end], dtype=np.float32).tolist(),
        }
    )
    eval_ds = Dataset.from_dict(
        {
            "sentence": texts[train_end:],
            "label": np.asarray(reduced[train_end:], dtype=np.float32).tolist(),
        }
    )
    for path, ds in ((train_path, train_ds), (eval_path, eval_ds)):
        if path.exists():
            shutil.rmtree(path)
        ds.save_to_disk(str(path))
    return len(train_ds), len(eval_ds)
