#!/usr/bin/env python3
"""GPU OFFLINE: distill mMiniLM against PCA-reduced Harrier embeddings.

Requires artifacts from 02b_fit_pca_and_encode_torch.py. No internet / no Hub push.

Defaults use cosine EmbedDistillLoss + fp32 (MSE+Normalize+TF32/bf16 was NaN-prone
on raw mMiniLM).

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


def autocast_ctx(enabled: bool = False):
    """Only autocast when explicitly enabled — do NOT default to bf16 on H100."""
    import torch

    if not enabled or not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def configure_matmul(allow_tf32: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    # TF32 reduces matmul precision on H100; disable for fragile distill runs
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    logger.info("TF32 matmul=%s cudnn_tf32=%s", allow_tf32, allow_tf32)


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
        for key, value in results.items():
            if isinstance(value, (int, float)):
                return key, float(value)
        raise RuntimeError(f"No numeric metrics in evaluator results: {results}")
    candidates.sort(key=lambda x: -x[2])
    return candidates[0][0], candidates[0][1]


def probe_finite_loss(student, loss_fn, dataset, batch_size: int = 8) -> None:
    """Run a few manual train steps; abort early if loss/grads go non-finite."""
    import torch

    student.train()
    device = student.device
    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=1e-5,
        weight_decay=0.01,
    )
    n = min(batch_size * 3, len(dataset))
    metric = getattr(loss_fn, "distance_metric", "mse")

    for step in range(3):
        start = step * batch_size
        end = min(start + batch_size, n)
        if start >= end:
            break
        sentences = [dataset[i]["sentence"] for i in range(start, end)]
        labels = torch.tensor(
            [dataset[i]["label"] for i in range(start, end)],
            dtype=torch.float32,
            device=device,
        )
        feats = student.tokenize(sentences)
        feats = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in feats.items()}
        out = student(feats)["sentence_embedding"]
        teacher = labels.to(device=out.device, dtype=out.dtype)

        if metric == "cosine":
            step_loss = (1 - torch.nn.functional.cosine_similarity(out, teacher, dim=-1)).mean()
        elif metric == "l2":
            step_loss = torch.norm(out - teacher, dim=-1).mean()
        else:
            step_loss = torch.nn.functional.mse_loss(out, teacher)

        if not torch.isfinite(step_loss):
            raise SystemExit(
                f"Stability probe failed at step {step}: loss={step_loss}. "
                "Check labels for NaN and keep bf16/TF32 disabled."
            )
        opt.zero_grad(set_to_none=True)
        step_loss.backward()
        grads = [p.grad for p in student.parameters() if p.grad is not None]
        if not grads:
            raise SystemExit("Stability probe: no gradients produced")
        gn = torch.norm(torch.stack([g.detach().float().norm() for g in grads]))
        if not torch.isfinite(gn):
            raise SystemExit(
                f"Stability probe failed at step {step}: grad_norm={gn}, loss={float(step_loss)}. "
                "Config should use distance_metric=cosine and normalize_during_training=false."
            )
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        logger.info(
            "Stability probe step %s: loss=%.6f grad_norm=%.6f OK",
            step,
            float(step_loss.detach()),
            float(gn),
        )
    student.zero_grad(set_to_none=True)
    logger.info("Stability probe passed (finite loss+grads)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill student from PCA Harrier labels")
    p.add_argument("--config", default=None)
    p.add_argument("--bundle-dir", default=None)
    p.add_argument(
        "--eval-only",
        type=str,
        default=None,
        help="Skip training; load this saved model and run evaluators only.",
    )
    p.add_argument(
        "--skip-baseline-eval",
        action="store_true",
        help="Skip pre-train NanoBEIR/STS baseline (faster; still eval during/after train).",
    )
    p.add_argument(
        "--skip-stability-probe",
        action="store_true",
        help="Skip the 3-step finite loss/grad probe before trainer.train().",
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

    import numpy as np
    import torch
    from datasets import load_from_disk
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerModelCardData,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import EmbedDistillLoss, MSELoss
    from sentence_transformers.sentence_transformer.modules import Normalize

    allow_tf32 = bool(cfg.get("allow_tf32", False))
    configure_matmul(allow_tf32)

    use_bf16 = bool(cfg.get("bf16", False)) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = bool(cfg.get("fp16", False)) and torch.cuda.is_available() and not use_bf16
    mixed = use_bf16 or use_fp16

    evaluator, sts_eval, nano_eval = build_evaluator(paths, cfg, smoke)

    if args.eval_only:
        logger.info("Eval-only mode: loading %s", args.eval_only)
        model = SentenceTransformer(args.eval_only)
        with autocast_ctx(enabled=mixed):
            results = evaluator(model)
        metric_name, score = pick_metric(results, sts_eval, nano_eval)
        logger.info("Eval-only %s=%.4f", metric_name, score)
        return

    for required in ("student", "train_mse", "eval_mse", "stsb"):
        if not paths[required].exists():
            raise SystemExit(f"Missing {paths[required]} — run prior pipeline steps first")

    logger.info("Loading student from %s", paths["student"])
    # CRITICAL: this checkpoint often loads as float16 by default. AdamW without a
    # GradScaler then NaNs on the first step (loss/grad_norm become nan).
    student = SentenceTransformer(
        str(paths["student"]),
        model_kwargs={"torch_dtype": torch.float32},
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="mMiniLMv2-L12-H384 distilled from harrier-oss-v1-0.6b via PCA-384",
        ),
    )
    student = student.float()  # belt-and-suspenders if any submodule stayed fp16
    param_dtypes = {str(p.dtype) for p in student.parameters()}
    logger.info("Student parameter dtypes: %s", sorted(param_dtypes))
    if any(dt != "torch.float32" for dt in param_dtypes):
        raise SystemExit(
            f"Student weights are not float32 ({param_dtypes}). "
            "Refusing to train — fp16 Adam without GradScaler produces NaNs."
        )
    max_seq_length = cfg.get("max_seq_length")
    if max_seq_length is not None:
        student.max_seq_length = int(max_seq_length)
        logger.info("Set student.max_seq_length=%s", max_seq_length)

    # Normalize during training backprops through L2-norm and was NaN-prone with MSE.
    # Cosine loss already L2-normalizes internally; append Normalize after training for save.
    normalize_during_training = bool(cfg.get("normalize_during_training", False))
    if normalize_during_training and not any(isinstance(m, Normalize) for m in student):
        student.append(Normalize())
        logger.info("Appended Normalize() for training (normalize_during_training=true)")
    elif not normalize_during_training:
        logger.info("Training WITHOUT Normalize module (will append before save)")

    train_dataset = load_from_disk(str(paths["train_mse"]))
    eval_dataset = load_from_disk(str(paths["eval_mse"]))
    if smoke:
        train_dataset = train_dataset.select(range(min(64, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(16, len(eval_dataset))))

    def _nonempty(example):
        t = example["sentence"]
        return isinstance(t, str) and bool(t.strip())

    before = len(train_dataset)
    train_dataset = train_dataset.filter(_nonempty)
    eval_dataset = eval_dataset.filter(_nonempty)
    if len(train_dataset) < before:
        logger.info("Filtered empty train sentences: %s → %s", before, len(train_dataset))

    label0 = train_dataset[0]["label"]
    label_dim = len(label0) if isinstance(label0, list) else int(np.asarray(label0).shape[-1])
    student_dim = student.get_embedding_dimension()
    if label_dim != student_dim:
        raise SystemExit(
            f"Label dim {label_dim} != student dim {student_dim}. "
            "Re-run 02b_fit_pca_and_encode_torch.py with matching pca_dim."
        )

    # Full-ish label scan (cap at 50k for speed)
    scan_n = min(50_000, len(train_dataset))
    sample_labels = np.asarray(
        [train_dataset[i]["label"] for i in range(scan_n)], dtype=np.float32
    )
    stats = label_stats(sample_labels)
    logger.info("Label preflight (n=%s): %s", scan_n, stats)
    if not stats["finite"]:
        raise SystemExit(
            f"Non-finite labels in train_mse (nan={stats['nan']}, inf={stats['inf']}). "
            "Re-run 02b and check teacher embeddings."
        )
    if stats["norm_min"] < 0.5 or stats["norm_max"] > 1.5:
        logger.warning(
            "Label L2 norms not near 1.0 (min=%.4f max=%.4f).",
            stats["norm_min"],
            stats["norm_max"],
        )

    distance_metric = str(cfg.get("distance_metric", "cosine")).lower()
    if distance_metric == "mse":
        loss = MSELoss(model=student)
    else:
        loss = EmbedDistillLoss(model=student, distance_metric=distance_metric)
    logger.info("Using EmbedDistill/MSE distance_metric=%s", distance_metric)

    if not args.skip_stability_probe and not smoke:
        logger.info("Running stability probe (3 Adam steps) before full training ...")
        # Probe on a fresh copy of weights state — actually probe mutates student.
        # Save state, probe, reload if we want clean start. Simpler: probe then continue
        # (3 steps of warmup are fine). Or clone. We'll probe then continue from probed weights.
        probe_finite_loss(student, loss, train_dataset, batch_size=min(8, len(train_dataset)))

    baseline_eval = 0.0
    baseline_metric = "skipped"
    baseline_results: dict = {}
    metric_key = "eval_sts-dev_spearman_cosine"
    if args.skip_baseline_eval:
        logger.info("Skipping pre-train baseline eval (--skip-baseline-eval)")
    else:
        logger.info("Student performance before distillation (fp32 eval):")
        with autocast_ctx(enabled=mixed):
            baseline_results = evaluator(student)
        baseline_metric, baseline_eval = pick_metric(baseline_results, sts_eval, nano_eval)
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
    if mixed:
        logger.warning(
            "Mixed precision enabled (bf16=%s fp16=%s). Strongly prefer both false if NaNs persist.",
            use_bf16,
            use_fp16,
        )
    else:
        logger.info("Training in fp32")

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
        logging_steps=1 if smoke else max(1, int(0.01 * 100)),  # at least every step early
        logging_first_step=True,
        logging_nan_inf_filter=False,
        load_best_model_at_end=not smoke and not args.skip_baseline_eval,
        metric_for_best_model=metric_key,
        greater_is_better=True,
        report_to=report_to_list,
        logging_dir=str(tb_dir),
        run_name="mminilm-harrier-pca-mse",
        seed=int(cfg.get("seed", 12)),
        remove_unused_columns=False,
        dataloader_drop_last=True,
        optim="adamw_torch",
    )
    # Prefer frequent early logging: use fraction again for long runs
    if not smoke:
        args_train.logging_steps = 0.01

    trainer = SentenceTransformerTrainer(
        model=student,
        args=args_train,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # Ensure Normalize on the saved model for cosine/dot-product inference
    if not any(isinstance(m, Normalize) for m in student):
        student.append(Normalize())
        logger.info("Appended Normalize() before save")

    logger.info("Student performance after distillation:")
    with autocast_ctx(enabled=mixed):
        final_results = evaluator(student)
    final_metric, score = pick_metric(final_results, sts_eval, nano_eval)
    if baseline_metric != "skipped" and final_metric in final_results:
        # Prefer matching key from baseline_results when available
        pass
    delta = score - float(baseline_eval)
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
    logger.info("Saved SentenceTransformer to %s", final_dir)

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
            "distance_metric": distance_metric,
            "normalize_during_training": normalize_during_training,
            "allow_tf32": allow_tf32,
            "bf16": use_bf16,
            "fp16": use_fp16,
            "student_final": str(final_dir),
        },
    )


if __name__ == "__main__":
    main()
