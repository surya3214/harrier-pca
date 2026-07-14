#!/usr/bin/env python3
"""ONLINE: upload trained student + PCA artifacts to the Hugging Face Hub.

Copy bundle/outputs and bundle/artifacts back from the GPU host first.

Usage:
  python scripts/04_upload_results.py
  python scripts/04_upload_results.py --bundle-dir /path/to/offline_bundle --private
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    bundle_paths,
    load_config,
    resolve_bundle_dir,
    setup_logging,
)

logger = logging.getLogger("upload")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload distilled student + PCA artifacts")
    p.add_argument("--config", default=None)
    p.add_argument("--bundle-dir", default=None)
    p.add_argument("--hub-model-id", default=None, help="Override Hub model repo id")
    p.add_argument("--hub-pca-dataset-id", default=None, help="Override Hub dataset repo for PCA npz")
    p.add_argument("--private", action="store_true", help="Create private repos")
    p.add_argument("--skip-model", action="store_true")
    p.add_argument("--skip-pca", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    bundle_dir = resolve_bundle_dir(cfg, args.bundle_dir)
    paths = bundle_paths(bundle_dir)
    setup_logging(paths["logs"] / "04_upload_results.log", name="upload")

    model_id = args.hub_model_id or cfg.get("hub_model_id")
    pca_id = args.hub_pca_dataset_id or cfg.get("hub_pca_dataset_id")

    if not args.skip_model:
        if not paths["student_final"].exists():
            raise SystemExit(f"Missing student folder: {paths['student_final']}")
        logger.info("Uploading student model to %s", model_id)
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(paths["student_final"]))
        url = model.push_to_hub(model_id, private=args.private)
        logger.info("Model pushed: %s", url)

    if not args.skip_pca:
        if not paths["pca"].exists():
            raise SystemExit(f"Missing PCA file: {paths['pca']}")
        logger.info("Uploading PCA artifacts to dataset repo %s", pca_id)
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(pca_id, repo_type="dataset", private=args.private, exist_ok=True)
        # Upload pca npz + optional stats
        api.upload_file(
            path_or_fileobj=str(paths["pca"]),
            path_in_repo="pca_384.npz",
            repo_id=pca_id,
            repo_type="dataset",
            commit_message="Add Harrier PCA 1024→384 components",
        )
        stats = paths["artifacts"] / "pca_encode_stats.json"
        if stats.exists():
            api.upload_file(
                path_or_fileobj=str(stats),
                path_in_repo="pca_encode_stats.json",
                repo_id=pca_id,
                repo_type="dataset",
                commit_message="Add PCA encode stats",
            )
        summary = paths["outputs"] / "train_summary.json"
        if summary.exists():
            api.upload_file(
                path_or_fileobj=str(summary),
                path_in_repo="train_summary.json",
                repo_id=pca_id,
                repo_type="dataset",
                commit_message="Add train summary",
            )
        logger.info("PCA artifacts at https://huggingface.co/datasets/%s", pca_id)

    logger.info("Upload complete.")


if __name__ == "__main__":
    main()
