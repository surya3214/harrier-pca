#!/usr/bin/env python3
"""GPU OFFLINE: PCA via torch.pca_lowrank (avoids sklearn/OpenMP hang after CUDA encode).

Same bundle contract as 02_fit_pca_and_encode.py:
  - reads datasets/corpus (text, prompt_name)
  - writes artifacts/pca_384.npz, train_mse/, eval_mse/

Difference vs 02:
  1. Encode full corpus once → save teacher_emb_1024.npy checkpoint
  2. Free the teacher from GPU
  3. Fit PCA with torch.pca_lowrank on CUDA (no sklearn)
  4. Transform + L2-normalize → MSE label datasets

Usage:
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  python scripts/02b_fit_pca_and_encode_torch.py --bundle-dir offline_bundle

  # Resume PCA from a previous encode checkpoint (skip teacher encode):
  python scripts/02b_fit_pca_and_encode_torch.py --bundle-dir offline_bundle --skip-encode

  SMOKE_TEST=1 python scripts/02b_fit_pca_and_encode_torch.py
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

# Cap BLAS threads BEFORE numpy/torch import to avoid OpenMP deadlocks.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import limit_blas_threads  # noqa: E402

limit_blas_threads(1)

from common import (  # noqa: E402
    assert_finite_embeddings,
    bundle_paths,
    encode_with_prompts,
    ensure_dirs,
    export_pca_teacher_st,
    l2_normalize,
    load_config,
    resolve_bundle_dir,
    save_mse_datasets,
    set_offline_env,
    setup_logging,
    write_json,
)

logger = logging.getLogger("pca-torch")


def fit_pca_torch(emb, n_components: int, device: str, niter: int = 4):
    """sklearn-compatible PCA via torch.pca_lowrank.

    Returns (components_(n_comp, dim), mean_(dim,), explained_variance_ratio_(n_comp,)).
    """
    import torch

    x = torch.as_tensor(emb, dtype=torch.float32, device=device)
    n_samples, n_features = x.shape
    if n_components > min(n_samples, n_features):
        raise ValueError(
            f"n_components={n_components} > min(n_samples, n_features)={min(n_samples, n_features)}"
        )

    t0 = time.time()
    logger.info(
        "torch.pca_lowrank on %s: shape=%s q=%s niter=%s ...",
        device,
        tuple(x.shape),
        n_components,
        niter,
    )
    # flush so the hang is never "silent"
    for h in logging.root.handlers:
        h.flush()

    mean = x.mean(dim=0)
    # pca_lowrank centers internally when center=True; V is (n_features, q)
    _u, s, v = torch.pca_lowrank(x, q=n_components, center=True, niter=niter)
    components = v.T.contiguous()  # (q, n_features) == sklearn PCA.components_

    # Explained variance from singular values: var_i = s_i^2 / (n-1)
    # Total variance ≈ sum of all feature variances (exact for full SVD; approx for low-rank).
    x_centered = x - mean
    total_var = (x_centered.pow(2).sum() / max(n_samples - 1, 1)).clamp_min(1e-12)
    ev = (s.pow(2) / max(n_samples - 1, 1))
    evr = (ev / total_var).clamp(min=0)

    elapsed = time.time() - t0
    explained = float(evr.sum().item())
    logger.info(
        "PCA done in %.1fs | cumulative explained variance (top %s): %.4f",
        elapsed,
        n_components,
        explained,
    )
    return (
        components.detach().cpu().numpy(),
        mean.detach().cpu().numpy(),
        evr.detach().cpu().numpy(),
        explained,
    )


def transform_pca(emb, components, mean):
    import numpy as np

    return (emb - mean) @ components.T


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit PCA with torch.pca_lowrank and write MSE labels (GPU-safe)"
    )
    p.add_argument("--config", default=None)
    p.add_argument("--bundle-dir", default=None)
    p.add_argument(
        "--skip-encode",
        action="store_true",
        help="Reuse artifacts/teacher_emb_1024.npy (skip Harrier encode)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="PCA device: cuda|cpu (default: cuda if available else cpu)",
    )
    p.add_argument("--pca-niter", type=int, default=4, help="torch.pca_lowrank power iterations")
    p.add_argument(
        "--skip-export-teacher",
        action="store_true",
        help="Do not save outputs/teacher-pca-384 SentenceTransformer",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_offline_env()
    limit_blas_threads(1)

    cfg = load_config(args.config)
    bundle_dir = resolve_bundle_dir(cfg, args.bundle_dir)
    paths = bundle_paths(bundle_dir)
    ensure_dirs(paths["artifacts"], paths["logs"])
    setup_logging(paths["logs"] / "02b_fit_pca_and_encode_torch.log", name="pca-torch")

    smoke = os.environ.get("SMOKE_TEST") == "1"
    pca_dim = int(cfg["pca_dim"])
    pca_fit_size = 64 if smoke else int(cfg["pca_fit_size"])
    eval_mse_size = 16 if smoke else int(cfg["eval_mse_size"])
    batch_size = 8 if smoke else int(cfg["teacher_encode_batch_size"])
    max_seq_length = cfg.get("max_seq_length")  # e.g. 512 to match student

    import numpy as np
    import torch
    from datasets import load_from_disk

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("PCA device=%s | smoke=%s", device, smoke)

    if not paths["corpus"].exists():
        raise SystemExit(f"Corpus missing: {paths['corpus']} (run 01_download_bundle.py first)")

    logger.info("Loading corpus from %s", paths["corpus"])
    corpus = load_from_disk(str(paths["corpus"]))
    n = len(corpus)
    if smoke:
        n = min(n, pca_fit_size + eval_mse_size + 32)
        corpus = corpus.select(range(n))
        logger.info("SMOKE_TEST=1: using %s corpus rows", n)

    if n < pca_fit_size + eval_mse_size:
        raise SystemExit(
            f"Corpus too small ({n}) for pca_fit_size={pca_fit_size} + eval_mse_size={eval_mse_size}"
        )

    texts = list(corpus["text"])
    prompt_names = list(corpus["prompt_name"])

    # --- Encode (or load checkpoint) ---
    if args.skip_encode:
        if not paths["teacher_emb"].exists():
            raise SystemExit(f"--skip-encode set but missing {paths['teacher_emb']}")
        logger.info("Loading teacher embeddings from %s", paths["teacher_emb"])
        full_emb = np.load(paths["teacher_emb"])
        if full_emb.shape[0] != len(texts):
            raise SystemExit(
                f"teacher_emb rows {full_emb.shape[0]} != corpus rows {len(texts)}. "
                "Re-encode without --skip-encode."
            )
    else:
        if not paths["teacher"].exists():
            raise SystemExit(f"Teacher model missing: {paths['teacher']}")

        from sentence_transformers import SentenceTransformer

        logger.info("Loading teacher from %s", paths["teacher"])
        teacher = SentenceTransformer(str(paths["teacher"]), model_kwargs={"dtype": "auto"})
        if max_seq_length is not None:
            teacher.max_seq_length = int(max_seq_length)
            logger.info("Set teacher.max_seq_length=%s", max_seq_length)

        teacher_dim = teacher.get_embedding_dimension()
        logger.info("Teacher embedding dim=%s → PCA dim=%s", teacher_dim, pca_dim)
        if teacher_dim < pca_dim:
            raise SystemExit(f"Teacher dim {teacher_dim} < pca_dim {pca_dim}")

        logger.info("Encoding FULL corpus once (%s texts) ...", f"{len(texts):,}")
        t0 = time.time()
        full_emb = encode_with_prompts(teacher, texts, prompt_names, batch_size)
        logger.info("Encode finished in %.1fs | shape=%s", time.time() - t0, full_emb.shape)
        assert_finite_embeddings(full_emb, "teacher_embeddings")

        logger.info("Saving encode checkpoint → %s", paths["teacher_emb"])
        np.save(paths["teacher_emb"], full_emb)

        # Free GPU before PCA (also avoids any encoder/BLAS interaction)
        logger.info("Releasing teacher from memory before PCA ...")
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Teacher released")

    teacher_dim = int(full_emb.shape[1])
    if teacher_dim < pca_dim:
        raise SystemExit(f"Embedding dim {teacher_dim} < pca_dim {pca_dim}")

    # --- PCA (torch, no sklearn) ---
    fit_emb = full_emb[:pca_fit_size]
    logger.info("Fitting PCA on first %s rows (torch.pca_lowrank) ...", f"{pca_fit_size:,}")
    for h in logging.root.handlers:
        h.flush()

    components, mean, evr, explained = fit_pca_torch(
        fit_emb, n_components=pca_dim, device=device, niter=int(args.pca_niter)
    )

    np.savez(
        paths["pca"],
        components_=components.astype(np.float32),
        mean_=mean.astype(np.float32),
        explained_variance_ratio_=evr.astype(np.float32),
        teacher_dim=np.array([teacher_dim], dtype=np.int32),
        pca_dim=np.array([pca_dim], dtype=np.int32),
        method=np.array(["torch.pca_lowrank"]),
    )
    logger.info("Saved PCA → %s", paths["pca"])

    logger.info("Transforming full corpus + L2-normalize ...")
    reduced = transform_pca(full_emb, components, mean).astype(np.float32)
    assert_finite_embeddings(reduced, "pca_transformed")
    reduced = l2_normalize(reduced).astype(np.float32)
    assert_finite_embeddings(reduced, "pca_l2_normalized")

    train_n, eval_n = save_mse_datasets(
        texts, reduced, eval_mse_size, paths["train_mse"], paths["eval_mse"]
    )
    logger.info("Saved train_mse=%s eval_mse=%s", f"{train_n:,}", f"{eval_n:,}")

    write_json(
        paths["artifacts"] / "pca_encode_stats.json",
        {
            "method": "torch.pca_lowrank",
            "teacher_dim": teacher_dim,
            "pca_dim": pca_dim,
            "pca_fit_size": pca_fit_size,
            "explained_variance_sum": explained,
            "train_rows": train_n,
            "eval_mse_rows": eval_n,
            "smoke_test": smoke,
            "device": device,
            "max_seq_length": max_seq_length,
            "teacher_emb_path": str(paths["teacher_emb"]),
            "teacher_pca_st": str(paths["teacher_pca"]),
        },
    )

    if not args.skip_export_teacher:
        logger.info("Exporting PCA teacher SentenceTransformer → %s", paths["teacher_pca"])
        out = export_pca_teacher_st(
            teacher_path=paths["teacher"],
            pca_npz_path=paths["pca"],
            output_path=paths["teacher_pca"],
            max_seq_length=int(max_seq_length) if max_seq_length is not None else None,
        )
        logger.info("Saved PCA teacher ST model to %s (dim=%s)", out, pca_dim)

    logger.info("Done. Next: scripts/03_train_student.py")


if __name__ == "__main__":
    main()
