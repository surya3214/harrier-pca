#!/usr/bin/env python3
"""GPU OFFLINE: fit PCA (1024→384) on Harrier embeddings and write MSE distillation labels.

Requires the offline_bundle from 01_download_bundle.py. No internet.

Usage:
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  python scripts/02_fit_pca_and_encode.py --bundle-dir /path/to/offline_bundle

  SMOKE_TEST=1 python scripts/02_fit_pca_and_encode.py   # tiny slice
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    bundle_paths,
    ensure_dirs,
    l2_normalize,
    load_config,
    resolve_bundle_dir,
    set_offline_env,
    setup_logging,
    write_json,
)

logger = logging.getLogger("pca-encode")


def encode_with_prompts(model, texts: list[str], prompt_names: list[str], batch_size: int):
    """Encode texts grouped by prompt_name (Harrier requires task prompts)."""
    from collections import defaultdict

    import numpy as np

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit PCA on Harrier and encode MSE labels")
    p.add_argument("--config", default=None)
    p.add_argument("--bundle-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_offline_env()
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

    if not paths["teacher"].exists():
        raise SystemExit(f"Teacher model missing: {paths['teacher']} (run 01_download_bundle.py first)")
    if not paths["corpus"].exists():
        raise SystemExit(f"Corpus missing: {paths['corpus']} (run 01_download_bundle.py first)")

    import numpy as np
    from datasets import Dataset, load_from_disk
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
    teacher_dim = teacher.get_embedding_dimension()
    logger.info("Teacher embedding dim=%s → PCA dim=%s", teacher_dim, pca_dim)
    if teacher_dim < pca_dim:
        raise SystemExit(f"Teacher dim {teacher_dim} < pca_dim {pca_dim}")

    # --- Fit PCA ---
    fit_texts = texts[:pca_fit_size]
    fit_prompts = prompt_names[:pca_fit_size]
    logger.info("Encoding %s texts for PCA fit", f"{len(fit_texts):,}")
    fit_emb = encode_with_prompts(teacher, fit_texts, fit_prompts, batch_size)

    logger.info("Fitting PCA(%s)", pca_dim)
    pca = PCA(n_components=pca_dim, random_state=int(cfg.get("seed", 12)))
    pca.fit(fit_emb)
    explained = float(pca.explained_variance_ratio_.sum())
    logger.info("PCA cumulative explained variance (top %s): %.4f", pca_dim, explained)

    np.savez(
        paths["pca"],
        components_=pca.components_.astype(np.float32),
        mean_=pca.mean_.astype(np.float32),
        explained_variance_ratio_=pca.explained_variance_ratio_.astype(np.float32),
        teacher_dim=np.array([teacher_dim], dtype=np.int32),
        pca_dim=np.array([pca_dim], dtype=np.int32),
    )
    logger.info("Saved PCA to %s", paths["pca"])

    # --- Encode full corpus, transform, L2-normalize ---
    logger.info("Encoding full corpus (%s) with teacher", f"{len(texts):,}")
    full_emb = encode_with_prompts(teacher, texts, prompt_names, batch_size)
    reduced = pca.transform(full_emb).astype(np.float32)
    reduced = l2_normalize(reduced).astype(np.float32)

    # Hold out last eval_mse_size rows for MSE eval; rest for train
    train_end = len(texts) - eval_mse_size
    train_sentences = texts[:train_end]
    train_labels = reduced[:train_end]
    eval_sentences = texts[train_end:]
    eval_labels = reduced[train_end:]

    train_ds = Dataset.from_dict(
        {"sentence": train_sentences, "label": train_labels.tolist()}
    )
    eval_ds = Dataset.from_dict(
        {"sentence": eval_sentences, "label": eval_labels.tolist()}
    )

    for path, ds in ((paths["train_mse"], train_ds), (paths["eval_mse"], eval_ds)):
        if path.exists():
            import shutil

            shutil.rmtree(path)
        ds.save_to_disk(str(path))
        logger.info("Saved %s (%s rows)", path, f"{len(ds):,}")

    write_json(
        paths["artifacts"] / "pca_encode_stats.json",
        {
            "teacher_dim": teacher_dim,
            "pca_dim": pca_dim,
            "pca_fit_size": pca_fit_size,
            "explained_variance_sum": explained,
            "train_rows": len(train_ds),
            "eval_mse_rows": len(eval_ds),
            "smoke_test": smoke,
        },
    )
    logger.info("Done. Next: scripts/03_train_student.py")


if __name__ == "__main__":
    main()
