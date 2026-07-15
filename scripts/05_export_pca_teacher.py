#!/usr/bin/env python3
"""Export Harrier + fitted PCA as a loadable SentenceTransformer (384-d).

Use when 02b already wrote pca_384.npz and you want outputs/teacher-pca-384/
without re-encoding the corpus.

Usage:
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  python scripts/05_export_pca_teacher.py --bundle-dir offline_bundle
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    bundle_paths,
    export_pca_teacher_st,
    load_config,
    resolve_bundle_dir,
    set_offline_env,
    setup_logging,
)

logger = logging.getLogger("export-pca-teacher")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Harrier+PCA as SentenceTransformer")
    p.add_argument("--config", default=None)
    p.add_argument("--bundle-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_offline_env()
    cfg = load_config(args.config)
    bundle_dir = resolve_bundle_dir(cfg, args.bundle_dir)
    paths = bundle_paths(bundle_dir)
    setup_logging(paths["logs"] / "05_export_pca_teacher.log", name="export-pca-teacher")

    max_seq_length = cfg.get("max_seq_length")
    out = export_pca_teacher_st(
        teacher_path=paths["teacher"],
        pca_npz_path=paths["pca"],
        output_path=paths["teacher_pca"],
        max_seq_length=int(max_seq_length) if max_seq_length is not None else None,
    )
    logger.info(
        "Saved PCA teacher to %s — load with SentenceTransformer(%r)",
        out,
        out,
    )


if __name__ == "__main__":
    main()
