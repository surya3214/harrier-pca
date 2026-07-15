#!/usr/bin/env python3
"""GPU OFFLINE: fit PCA (1024→384) with sklearn and write MSE distillation labels.

NOTE: After CUDA encode, sklearn PCA can hang due to OpenMP/MKL + PyTorch thread
deadlocks. Prefer scripts/02b_fit_pca_and_encode_torch.py on H100 infra.

This script still sets BLAS thread caps and logs around pca.fit for diagnostics.

Usage:
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  # Prefer:
  python scripts/02b_fit_pca_and_encode_torch.py --bundle-dir offline_bundle
  # Fallback:
  python scripts/02_fit_pca_and_encode.py --bundle-dir offline_bundle
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import limit_blas_threads  # noqa: E402

limit_blas_threads(1)

from common import (  # noqa: E402
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

logger = logging.getLogger("pca-encode")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit PCA on Harrier and encode MSE labels (sklearn)")
    p.add_argument("--config", default=None)
    p.add_argument("--bundle-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_offline_env()
    limit_blas_threads(1)
    cfg = load_config(args.config)
    bundle_dir = resolve_bundle_dir(cfg, args.bundle_dir)
    paths = bundle_paths(bundle_dir)
    ensure_dirs(paths["artifacts"], paths["logs"])
    setup_logging(paths["logs"] / "02_fit_pca_and_encode.log", name="pca-encode")

    smoke = os.environ.get("SMOKE_TEST") == "1"
    pca_dim = int(cfg["pca_dim"])
    pca_fit_size = 64 if smoke else int(cfg["pca_fit_size"])
    eval_mse_size = 16 if smoke else int(cfg["eval_mse_size"])
    batch_size = 8 if smoke else int(cfg["teacher_encode_batch_size"])
    max_seq_length = cfg.get("max_seq_length")

    if not paths["teacher"].exists():
        raise SystemExit(f"Teacher model missing: {paths['teacher']} (run 01_download_bundle.py first)")
    if not paths["corpus"].exists():
        raise SystemExit(f"Corpus missing: {paths['corpus']} (run 01_download_bundle.py first)")

    import numpy as np
    import torch
    from datasets import load_from_disk
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

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

    logger.info("Loading teacher from %s", paths["teacher"])
    teacher = SentenceTransformer(str(paths["teacher"]), model_kwargs={"dtype": "auto"})
    if max_seq_length is not None:
        teacher.max_seq_length = int(max_seq_length)
        logger.info("Set teacher.max_seq_length=%s", max_seq_length)

    teacher_dim = teacher.get_embedding_dimension()
    logger.info("Teacher embedding dim=%s → PCA dim=%s", teacher_dim, pca_dim)
    if teacher_dim < pca_dim:
        raise SystemExit(f"Teacher dim {teacher_dim} < pca_dim {pca_dim}")

    # Encode full corpus once (same as 02b), then free teacher before sklearn PCA
    logger.info("Encoding FULL corpus once (%s texts) ...", f"{len(texts):,}")
    t0 = time.time()
    full_emb = encode_with_prompts(teacher, texts, prompt_names, batch_size)
    logger.info("Encode finished in %.1fs | shape=%s", time.time() - t0, full_emb.shape)
    np.save(paths["teacher_emb"], full_emb)

    logger.info("Releasing teacher before sklearn PCA (avoids OpenMP hang) ...")
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fit_emb = full_emb[:pca_fit_size]
    logger.info(
        "Fitting sklearn PCA(%s) on %s rows (svd_solver=randomized) ...",
        pca_dim,
        f"{len(fit_emb):,}",
    )
    for h in logging.root.handlers:
        h.flush()

    t0 = time.time()
    pca = PCA(
        n_components=pca_dim,
        svd_solver="randomized",
        random_state=int(cfg.get("seed", 12)),
    )
    pca.fit(fit_emb)
    explained = float(pca.explained_variance_ratio_.sum())
    logger.info(
        "PCA done in %.1fs | cumulative explained variance (top %s): %.4f",
        time.time() - t0,
        pca_dim,
        explained,
    )

    np.savez(
        paths["pca"],
        components_=pca.components_.astype(np.float32),
        mean_=pca.mean_.astype(np.float32),
        explained_variance_ratio_=pca.explained_variance_ratio_.astype(np.float32),
        teacher_dim=np.array([teacher_dim], dtype=np.int32),
        pca_dim=np.array([pca_dim], dtype=np.int32),
        method=np.array(["sklearn.PCA.randomized"]),
    )
    logger.info("Saved PCA to %s", paths["pca"])

    logger.info("Transforming full corpus + L2-normalize ...")
    reduced = pca.transform(full_emb).astype(np.float32)
    reduced = l2_normalize(reduced).astype(np.float32)

    train_n, eval_n = save_mse_datasets(
        texts, reduced, eval_mse_size, paths["train_mse"], paths["eval_mse"]
    )
    logger.info("Saved train_mse=%s eval_mse=%s", f"{train_n:,}", f"{eval_n:,}")

    write_json(
        paths["artifacts"] / "pca_encode_stats.json",
        {
            "method": "sklearn.PCA.randomized",
            "teacher_dim": teacher_dim,
            "pca_dim": pca_dim,
            "pca_fit_size": pca_fit_size,
            "explained_variance_sum": explained,
            "train_rows": train_n,
            "eval_mse_rows": eval_n,
            "smoke_test": smoke,
            "max_seq_length": max_seq_length,
            "teacher_pca_st": str(paths["teacher_pca"]),
        },
    )
    logger.info("Exporting PCA teacher SentenceTransformer → %s", paths["teacher_pca"])
    export_pca_teacher_st(
        teacher_path=paths["teacher"],
        pca_npz_path=paths["pca"],
        output_path=paths["teacher_pca"],
        max_seq_length=int(max_seq_length) if max_seq_length is not None else None,
    )
    logger.info("Done. Next: scripts/03_train_student.py")


if __name__ == "__main__":
    main()
