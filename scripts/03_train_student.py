#!/usr/bin/env python3
"""GPU OFFLINE: MSE-distill mMiniLM against PCA-reduced Harrier embeddings.

Requires artifacts from 02_fit_pca_and_encode.py. No internet / no Hub push.

Usage:
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  python scripts/03_train_student.py --bundle-dir /path/to/offline_bundle

  SMOKE_TEST=1 python scripts/03_train_student.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    bundle_paths,
    ensure_dirs,
    label_stats,
    load_config,
    resolve_bundle_dir,
    set_offline_env,
    setup_logging,
    write_json,
)

logger = logging.getLogger("train")


def autocast_ctx():
    import torch

    if not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def build_evaluator(paths: dict, cfg: dict, smoke: bool):
    from datasets import load_from_disk
    from sentence_transformers.sentence_transformer.evaluation import (
        EmbeddingSimilarityEvaluator,
        NanoBEIREvaluator,
        SequentialEvaluator,
    )
    from sentence_transformers.util.similarity import SimilarityFunction

    stsb = load_from_disk(str(paths["stsb"]))
    val = stsb["validation"] if "validation" in stsb else stsb["train"]
    if smoke:
        val = val.select(range(min(32, len(val))))

    sts_eval = EmbeddingSimilarityEvaluator(
        sentences1=list(val["sentence1"]),
        sentences2=list(val["sentence2"]),
        scores=list(val["score"]),
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )

    nanobeir_names = cfg.get("nanobeir_datasets", ["msmarco", "nfcorpus", "nq"])
    if smoke:
        nanobeir_names = nanobeir_names[:1]

    nano_kwargs = {
        "dataset_names": nanobeir_names,
        "batch_size": 16 if smoke else 128,
        "show_progress_bar": False,
    }
    if paths["nanobeir"].exists():
        nano_kwargs["dataset_id"] = str(paths["nanobeir"])
    else:
        logger.warning(
            "NanoBEIR local path missing (%s); evaluator will try Hub id (fails offline)",
            paths["nanobeir"],
        )

    nano_eval = NanoBEIREvaluator(**nano_kwargs)
    return SequentialEvaluator([sts_eval, nano_eval]), sts_eval, nano_eval


def pick_metric(results: dict, sts_eval, nano_eval) -> tuple[str, float]:
    """Prefer NanoBEIR mean nDCG; fall back to STS Spearman."""
    candidates = []
    for key, value in results.items():
        if not isinstance(value, (int, float)):
            continue
        lk = key.lower()
        if "nanobeir" in lk and "ndcg@10" in lk and "mean" in lk:
            candidates.append((key, float(value), 2))
        elif sts_eval.primary_metric and key == sts_eval.primary_metric:
            candidates.append((key, float(value), 1))
        elif "sts-dev" in lk and "spearman" in lk and "cosine" in lk:
            candidates.append((key, float(value), 1))
    if not candidates:
        # last resort: first numeric
        for key, value in results.items():
            if isinstance(value, (int, float)):
                return key, float(value)
        raise RuntimeError(f"No numeric metrics in evaluator results: {results}")
    candidates.sort(key=lambda x: -x[2])
    return candidates[0][0], candidates[0][1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MSE distill student from PCA Harrier labels")
    p.add_argument("--config", default=None)
    p.add_argument("--bundle-dir", default=None)
    p.add_argument(
        "--eval-only",
        type=str,
        default=None,
        help="Skip training; load this saved model and run evaluators only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_offline_env()
    cfg = load_config(args.config)
    bundle_dir = resolve_bundle_dir(cfg, args.bundle_dir)
    paths = bundle_paths(bundle_dir)
    ensure_dirs(paths["outputs"], paths["logs"])
    setup_logging(paths["logs"] / "03_train_student.log", name="train")

    smoke = os.environ.get("SMOKE_TEST") == "1"
    if smoke:
        logger.info("SMOKE_TEST=1: tiny data, max_steps=1, no long training")

    import torch
    from datasets import load_from_disk
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerModelCardData,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import MSELoss
    from sentence_transformers.sentence_transformer.modules import Normalize

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    evaluator, sts_eval, nano_eval = build_evaluator(paths, cfg, smoke)

    if args.eval_only:
        logger.info("Eval-only mode: loading %s", args.eval_only)
        model = SentenceTransformer(args.eval_only)
        with autocast_ctx():
            results = evaluator(model)
        metric_name, score = pick_metric(results, sts_eval, nano_eval)
        logger.info("Eval-only %s=%.4f", metric_name, score)
        return

    for required in ("student", "train_mse", "eval_mse", "stsb"):
        if not paths[required].exists():
            raise SystemExit(f"Missing {paths[required]} — run prior pipeline steps first")

    logger.info("Loading student from %s", paths["student"])
    student = SentenceTransformer(
        str(paths["student"]),
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="mMiniLMv2-L12-H384 distilled from harrier-oss-v1-0.6b via PCA-384",
        ),
    )
    max_seq_length = cfg.get("max_seq_length")
    if max_seq_length is not None:
        student.max_seq_length = int(max_seq_length)
        logger.info("Set student.max_seq_length=%s", max_seq_length)
    if not any(isinstance(m, Normalize) for m in student):
        student.append(Normalize())
        logger.info("Appended Normalize() to student")

    train_dataset = load_from_disk(str(paths["train_mse"]))
    eval_dataset = load_from_disk(str(paths["eval_mse"]))
    if smoke:
        train_dataset = train_dataset.select(range(min(64, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(16, len(eval_dataset))))

    # Drop empty sentences (Normalize + MSE can NaN on degenerate inputs)
    def _nonempty(example):
        t = example["sentence"]
        return isinstance(t, str) and bool(t.strip())

    before = len(train_dataset)
    train_dataset = train_dataset.filter(_nonempty)
    eval_dataset = eval_dataset.filter(_nonempty)
    if len(train_dataset) < before:
        logger.info("Filtered empty train sentences: %s → %s", before, len(train_dataset))

    label0 = train_dataset[0]["label"]
    label_dim = len(label0) if isinstance(label0, list) else int(label0.shape[-1])
    student_dim = student.get_embedding_dimension()
    if label_dim != student_dim:
        raise SystemExit(
            f"Label dim {label_dim} != student dim {student_dim}. "
            "Re-run 02b_fit_pca_and_encode_torch.py with matching pca_dim."
        )

    # Preflight: NaN labels are a common cause of step-2 grad_norm=nan / loss logged as 0
    import numpy as np

    sample_n = min(2048, len(train_dataset))
    sample_labels = np.asarray(train_dataset.select(range(sample_n))["label"], dtype=np.float32)
    stats = label_stats(sample_labels)
    logger.info("Label preflight (n=%s): %s", sample_n, stats)
    # Expected healthy first-step MSE ≈ 2/dim ≈ 0.0052 for near-orthogonal unit vectors
    logger.info(
        "Healthy first-step MSE for unit 384-d targets is ~%.6f (2/dim); "
        "logged loss=0 with grad_norm=nan usually means real NaN (Trainer filters NaN logs).",
        2.0 / student_dim,
    )
    if not stats["finite"]:
        raise SystemExit(
            f"Non-finite labels in train_mse (nan={stats['nan']}, inf={stats['inf']}). "
            "Re-run 02b and check teacher embeddings."
        )
    if stats["norm_min"] < 0.5 or stats["norm_max"] > 1.5:
        logger.warning(
            "Label L2 norms not near 1.0 (min=%.4f max=%.4f). "
            "Student uses Normalize(); mismatched scales can destabilize MSE.",
            stats["norm_min"],
            stats["norm_max"],
        )

    loss = MSELoss(model=student)

    logger.info("Student performance before distillation:")
    with autocast_ctx():
        baseline_results = evaluator(student)
    baseline_metric, baseline_eval = pick_metric(baseline_results, sts_eval, nano_eval)
    # Prefer NanoBEIR mean nDCG@10 for checkpoint selection when present
    best_metric_name = None
    for key in baseline_results:
        lk = key.lower()
        if "nanobeir" in lk and "ndcg@10" in lk and "mean" in lk:
            best_metric_name = key
            break
    if best_metric_name is None:
        best_metric_name = getattr(evaluator, "primary_metric", None) or baseline_metric
    metric_key = f"eval_{best_metric_name}"
    logger.info("Baseline %s=%.4f (metric_for_best_model=%s)", baseline_metric, baseline_eval, metric_key)

    output_dir = paths["outputs"] / "checkpoints"
    ensure_dirs(output_dir)

    train_bs = 8 if smoke else int(cfg["train_batch_size"])
    use_bf16 = bool(cfg.get("bf16", False)) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = bool(cfg.get("fp16", False)) and torch.cuda.is_available() and not use_bf16
    if use_bf16 or use_fp16:
        logger.warning(
            "Mixed precision enabled (bf16=%s fp16=%s). If you see grad_norm=nan / loss=0, "
            "set bf16: false and fp16: false in configs/default.yaml",
            use_bf16,
            use_fp16,
        )
    else:
        logger.info("Training in fp32 (recommended for MSE+Normalize on mMiniLM)")

    report_to = cfg.get("report_to", "tensorboard")
    if isinstance(report_to, str):
        report_to_list = ["none"] if report_to in (None, "", "none") else [report_to]
    else:
        report_to_list = list(report_to) if report_to else ["none"]
    if smoke:
        report_to_list = ["none"]

    tb_dir = paths["tensorboard"] / "mminilm-harrier-pca-mse"
    ensure_dirs(tb_dir)
    if "tensorboard" in report_to_list:
        logger.info(
            "TensorBoard logging → %s  |  view: tensorboard --logdir %s --port 6006",
            tb_dir,
            paths["tensorboard"],
        )

    args_train = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1 if smoke else float(cfg["num_train_epochs"]),
        max_steps=1 if smoke else -1,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=train_bs,
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=0.01,
        warmup_steps=float(cfg["warmup_steps"]),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        bf16=use_bf16,
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=1.0 if smoke else 0.1,
        save_strategy="steps",
        save_steps=1.0 if smoke else 0.1,
        save_total_limit=2,
        logging_steps=1 if smoke else 0.01,
        logging_first_step=True,
        # Default True hides NaN loss as 0 — keep False so instability is visible
        logging_nan_inf_filter=False,
        load_best_model_at_end=not smoke,
        metric_for_best_model=metric_key,
        greater_is_better=True,
        report_to=report_to_list,
        logging_dir=str(tb_dir),
        run_name="mminilm-harrier-pca-mse",
        seed=int(cfg.get("seed", 12)),
        remove_unused_columns=False,
    )

    trainer = SentenceTransformerTrainer(
        model=student,
        args=args_train,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    logger.info("Student performance after distillation:")
    with autocast_ctx():
        final_results = evaluator(student)
    final_metric, score = pick_metric(final_results, sts_eval, nano_eval)
    # Align delta on the same baseline key when possible
    if final_metric in baseline_results and isinstance(baseline_results[final_metric], (int, float)):
        baseline_eval = float(baseline_results[final_metric])
        baseline_metric = final_metric
    delta = score - baseline_eval
    verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
    logger.info(
        "VERDICT: %s | score=%.4f | baseline=%.4f | delta=%+.4f | metric=%s",
        verdict,
        score,
        baseline_eval,
        delta,
        final_metric,
    )

    final_dir = paths["student_final"]
    if final_dir.exists():
        import shutil

        shutil.rmtree(final_dir)
    student.save_pretrained(str(final_dir))
    logger.info("Saved SentenceTransformer to %s (load with SentenceTransformer(path))", final_dir)

    write_json(
        paths["outputs"] / "train_summary.json",
        {
            "verdict": verdict,
            "metric": final_metric,
            "score": score,
            "baseline": baseline_eval,
            "delta": delta,
            "baseline_metric": baseline_metric,
            "smoke_test": smoke,
            "student_final": str(final_dir),
        },
    )


if __name__ == "__main__":
    main()
