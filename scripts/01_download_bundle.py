#!/usr/bin/env python3
"""ONLINE (internet, no GPU required): download models + capped corpora into offline_bundle/.

Run on the networked machine, then copy the whole bundle to the air-gapped H100 host.

Usage:
  python scripts/01_download_bundle.py
  python scripts/01_download_bundle.py --bundle-dir /path/to/offline_bundle --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROMPTS,
    bundle_paths,
    ensure_dirs,
    load_config,
    resolve_bundle_dir,
    setup_logging,
    write_json,
)

logger = logging.getLogger("download")


def download_models(paths: dict, cfg: dict) -> None:
    from huggingface_hub import snapshot_download

    ensure_dirs(paths["models"])
    logger.info("Downloading teacher: %s", cfg["teacher_repo"])
    snapshot_download(
        repo_id=cfg["teacher_repo"],
        local_dir=str(paths["teacher"]),
    )
    logger.info("Downloading student: %s", cfg["student_repo"])
    snapshot_download(
        repo_id=cfg["student_repo"],
        local_dir=str(paths["student"]),
    )


def _add_text(seen: set[str], rows: list[dict], text: str, prompt_name: str, source: str, cap: int) -> bool:
    """Add unique non-empty text. Returns False if source cap reached."""
    if not isinstance(text, str):
        return True
    text = text.strip()
    if not text or text in seen:
        return True
    if len(rows) >= cap:
        return False
    seen.add(text)
    rows.append({"text": text, "prompt_name": prompt_name, "source": source})
    return True


def collect_all_nli(cap: int) -> list[dict]:
    from datasets import load_dataset

    logger.info("Collecting all-nli pair (cap=%s)", cap)
    ds = load_dataset("sentence-transformers/all-nli", "pair", split="train", streaming=True)
    seen: set[str] = set()
    rows: list[dict] = []
    for row in ds:
        for key in ("anchor", "positive", "premise", "hypothesis"):
            if key in row and not _add_text(seen, rows, row[key], PROMPTS["sts"], "all-nli", cap):
                return rows
        if len(rows) >= cap:
            break
    return rows


def collect_gooaq(cap: int) -> list[dict]:
    from datasets import load_dataset

    logger.info("Collecting gooaq (cap=%s)", cap)
    ds = load_dataset("sentence-transformers/gooaq", split="train", streaming=True)
    seen: set[str] = set()
    rows: list[dict] = []
    for row in ds:
        q = row.get("question") or row.get("query")
        a = row.get("answer")
        if q and not _add_text(seen, rows, q, PROMPTS["retrieval_query"], "gooaq", cap):
            return rows
        if a and not _add_text(seen, rows, a, PROMPTS["document"], "gooaq", cap):
            return rows
        if len(rows) >= cap:
            break
    return rows


def collect_natural_questions(cap: int) -> list[dict]:
    from datasets import load_dataset

    logger.info("Collecting natural-questions (cap=%s)", cap)
    ds = load_dataset("sentence-transformers/natural-questions", split="train", streaming=True)
    seen: set[str] = set()
    rows: list[dict] = []
    for row in ds:
        q = row.get("query") or row.get("question")
        a = row.get("answer")
        if q and not _add_text(seen, rows, q, PROMPTS["retrieval_query"], "natural-questions", cap):
            return rows
        if a and not _add_text(seen, rows, a, PROMPTS["document"], "natural-questions", cap):
            return rows
        if len(rows) >= cap:
            break
    return rows


def collect_msmarco(cap: int) -> list[dict]:
    """Cap MS MARCO via queries + streamed corpus (avoids multi-GB triplets dump)."""
    from datasets import load_dataset

    logger.info("Collecting msmarco queries+corpus (cap=%s)", cap)
    seen: set[str] = set()
    rows: list[dict] = []

    half = max(1, cap // 2)
    queries = load_dataset("sentence-transformers/msmarco", "queries", split="train", streaming=True)
    for row in queries:
        if not _add_text(seen, rows, row["query"], PROMPTS["retrieval_query"], "msmarco", half):
            break

    passage_cap = cap  # fill remaining budget with passages
    corpus = load_dataset("sentence-transformers/msmarco", "corpus", split="train", streaming=True)
    for row in corpus:
        passage = row.get("passage") or row.get("text")
        if passage and not _add_text(seen, rows, passage, PROMPTS["document"], "msmarco", passage_cap):
            break
        if len(rows) >= cap:
            break
    return rows


def build_corpus(paths: dict, cfg: dict) -> dict:
    from datasets import Dataset

    caps = cfg["caps"]
    parts = [
        collect_all_nli(int(caps["all_nli"])),
        collect_gooaq(int(caps["gooaq"])),
        collect_natural_questions(int(caps["natural_questions"])),
        collect_msmarco(int(caps["msmarco"])),
    ]

    # Global dedup keeping first occurrence (preserves preferred prompt_name)
    seen: set[str] = set()
    merged: list[dict] = []
    per_source: dict[str, int] = {}
    for part in parts:
        for row in part:
            if row["text"] in seen:
                continue
            seen.add(row["text"])
            merged.append(row)
            per_source[row["source"]] = per_source.get(row["source"], 0) + 1

    logger.info("Corpus after global dedup: %s rows (%s)", f"{len(merged):,}", per_source)
    ds = Dataset.from_list(merged)
    out = paths["corpus"]
    if out.exists():
        import shutil

        shutil.rmtree(out)
    ds.save_to_disk(str(out))
    return {"num_rows": len(merged), "per_source": per_source}


def download_stsb(paths: dict) -> dict:
    from datasets import load_dataset

    logger.info("Downloading STS-B")
    ds = load_dataset("sentence-transformers/stsb")
    out = paths["stsb"]
    if out.exists():
        import shutil

        shutil.rmtree(out)
    ds.save_to_disk(str(out))
    return {split: len(ds[split]) for split in ds}


def download_nanobeir(paths: dict) -> None:
    from huggingface_hub import snapshot_download

    logger.info("Downloading NanoBEIR-en for offline retrieval eval")
    ensure_dirs(paths["datasets"])
    snapshot_download(
        repo_id="sentence-transformers/NanoBEIR-en",
        repo_type="dataset",
        local_dir=str(paths["nanobeir"]),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download models + datasets into an offline bundle")
    p.add_argument("--config", default=None, help="Path to configs/default.yaml")
    p.add_argument("--bundle-dir", default=None, help="Output bundle directory")
    p.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    p.add_argument("--skip-corpus", action="store_true", help="Skip training corpus build")
    p.add_argument("--skip-eval", action="store_true", help="Skip STS-B / NanoBEIR downloads")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    bundle_dir = resolve_bundle_dir(cfg, args.bundle_dir)
    paths = bundle_paths(bundle_dir)
    ensure_dirs(paths["bundle"], paths["logs"], paths["artifacts"], paths["outputs"])
    setup_logging(paths["logs"] / "01_download_bundle.log", name="download")

    logger.info("Bundle directory: %s", bundle_dir)
    stats: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher_repo": cfg["teacher_repo"],
        "student_repo": cfg["student_repo"],
        "pca_dim": cfg["pca_dim"],
        "caps": cfg["caps"],
    }

    if not args.skip_models:
        download_models(paths, cfg)
        stats["models"] = {
            "teacher": str(paths["teacher"]),
            "student": str(paths["student"]),
        }

    if not args.skip_corpus:
        stats["corpus"] = build_corpus(paths, cfg)

    if not args.skip_eval:
        stats["stsb"] = download_stsb(paths)
        download_nanobeir(paths)
        stats["nanobeir"] = str(paths["nanobeir"])

    write_json(paths["manifest"], stats)
    logger.info("Wrote manifest: %s", paths["manifest"])
    logger.info("Done. Copy %s to the air-gapped GPU host.", bundle_dir)


if __name__ == "__main__":
    main()
