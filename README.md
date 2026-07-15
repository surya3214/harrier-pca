# Harrier PCA → mMiniLM Distillation (Air-Gapped)

Distill [`microsoft/harrier-oss-v1-0.6b`](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) (1024-d) into [`nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large`](https://huggingface.co/nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large) (384-d) via **PCA + MSELoss**, targeting higher STS and retrieval scores.

Designed for two machines:

| Machine | Role |
|---|---|
| **Online** (internet, CPU OK) | Download models/datasets; upload results |
| **H100 offline** (no internet) | Fit PCA, encode teacher labels, train student |

## Setup (both machines)

```bash
pip install -r requirements.txt
```

Edit [`configs/default.yaml`](configs/default.yaml) for caps, batch sizes, and Hub repo ids.

## Bundle layout

```
offline_bundle/
  models/harrier-oss-v1-0.6b/
  models/mMiniLMv2-L12-H384/
  datasets/corpus/          # text + prompt_name
  datasets/stsb/
  datasets/NanoBEIR-en/
  artifacts/pca_384.npz
  artifacts/train_mse/      # sentence, label[384]
  artifacts/eval_mse/
  outputs/teacher-pca-384/  # Harrier+PCA SentenceTransformer (384-d)
  outputs/student-final/    # loadable SentenceTransformer
  logs/tensorboard/         # TensorBoard event files
  logs/
  MANIFEST.json
```

## Runbook

### 1. Online — download

```bash
python scripts/01_download_bundle.py --bundle-dir offline_bundle
```

Copy the entire `offline_bundle/` directory to the air-gapped GPU host (USB / scp / shared FS).

### 2. Offline H100 — PCA + teacher labels

**Prefer the torch script** (avoids sklearn/OpenMP hangs after CUDA encode):

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Optional smoke test
SMOKE_TEST=1 python scripts/02b_fit_pca_and_encode_torch.py --bundle-dir offline_bundle

python scripts/02b_fit_pca_and_encode_torch.py --bundle-dir offline_bundle

# If encode already finished and teacher_emb_1024.npy exists:
# python scripts/02b_fit_pca_and_encode_torch.py --bundle-dir offline_bundle --skip-encode
```

Fallback (sklearn, also hardened with thread caps + free teacher before `fit`):

```bash
python scripts/02_fit_pca_and_encode.py --bundle-dir offline_bundle
```

Both scripts write the same artifacts (`pca_384.npz`, `train_mse/`, `eval_mse/`) for `03_train_student.py`.
`02b` also exports `outputs/teacher-pca-384/` (Harrier + PCA as a SentenceTransformer).

If you already have `pca_384.npz` and only need the ST export:

```bash
python scripts/05_export_pca_teacher.py --bundle-dir offline_bundle
```

Harrier prompts used while encoding:

- STS / NLI text → `sts_query`
- Retrieval queries → `web_search_query`
- Passages / answers → no prompt

Teacher `max_seq_length` defaults to **512** (see `configs/default.yaml`) so PCA targets match the student window.

### 3. Offline H100 — distill student

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

SMOKE_TEST=1 python scripts/03_train_student.py --bundle-dir offline_bundle   # optional
python scripts/03_train_student.py --bundle-dir offline_bundle
```

If training logs `loss≈0.005` then `grad_norm=nan` / `loss=0`, that first loss is healthy (~`2/384` for unit targets); the `0` is usually a hidden NaN (Trainer filters NaN logs).

**Current defaults (anti-NaN):** force **float32** weights on load (this MiniLM checkpoint often loads as fp16, and fp16 Adam without GradScaler NaNs immediately), `distance_metric: cosine`, `normalize_during_training: false`, `bf16/fp16: false`, `allow_tf32: false`, `lr: 1e-5`. A 3-step stability probe aborts if loss/grads are non-finite.

```bash
python scripts/03_train_student.py --bundle-dir offline_bundle
# optional: skip heavy pre-train NanoBEIR baseline
# python scripts/03_train_student.py --bundle-dir offline_bundle --skip-baseline-eval
```

Saves a standard SentenceTransformer folder at `offline_bundle/outputs/student-final/`.

#### TensorBoard

Training writes events under `offline_bundle/logs/tensorboard/` (`report_to: tensorboard` in config).

On the GPU host (or after copying `logs/tensorboard` to a machine with a browser):

```bash
pip install tensorboard   # if not already from requirements.txt
tensorboard --logdir offline_bundle/logs/tensorboard --port 6006 --bind_all
```

Then open `http://<host>:6006` (SSH tunnel if needed: `ssh -L 6006:localhost:6006 user@gpu-host`).

Disable with `report_to: none` in `configs/default.yaml`.

Load student later (any machine, no Hub needed):

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("offline_bundle/outputs/student-final/")
```

Load PCA teacher (384-d Harrier) for custom eval:

```python
from sentence_transformers import SentenceTransformer
teacher = SentenceTransformer("offline_bundle/outputs/teacher-pca-384/")
# same prompts as Harrier, e.g. prompt_name="web_search_query" / "sts_query"
emb = teacher.encode(["example query"], prompt_name="sts_query")
print(emb.shape)  # (1, 384)
```

### 4. Online — upload

Copy `outputs/` and `artifacts/` back to the online machine, then:

```bash
hf auth login   # if needed
python scripts/04_upload_results.py --bundle-dir offline_bundle
# python scripts/04_upload_results.py --private
```

Default Hub targets (override in config or CLI):

- Student: `surya3214/mminilm-h384-distilled-harrier-pca`
- PCA teacher ST: `surya3214/harrier-pca-384`
- PCA artifacts dataset: `surya3214/harrier-pca-384-artifacts`

## Training data (capped)

| Source | Role | Prompt |
|---|---|---|
| `sentence-transformers/all-nli` (`pair`) | STS signal | `sts_query` |
| `sentence-transformers/gooaq` | Web QA retrieval | query / document |
| `sentence-transformers/natural-questions` | Open-domain retrieval | query / document |
| `sentence-transformers/msmarco` (`queries` + streamed `corpus`) | Passage retrieval | query / document |

Eval: STS-B validation + NanoBEIR (`msmarco`, `nfcorpus`, `nq`).

## Scripts

| Script | Machine | Purpose |
|---|---|---|
| [`scripts/01_download_bundle.py`](scripts/01_download_bundle.py) | Online | Models + capped corpora + eval sets |
| [`scripts/02b_fit_pca_and_encode_torch.py`](scripts/02b_fit_pca_and_encode_torch.py) | GPU | **Preferred** — `torch.pca_lowrank` PCA + MSE labels + PCA teacher ST |
| [`scripts/02_fit_pca_and_encode.py`](scripts/02_fit_pca_and_encode.py) | GPU | Fallback sklearn PCA (thread-capped) |
| [`scripts/05_export_pca_teacher.py`](scripts/05_export_pca_teacher.py) | GPU | Export Harrier+PCA as SentenceTransformer only |
| [`scripts/03_train_student.py`](scripts/03_train_student.py) | GPU | MSE distillation + STS/NanoBEIR + TensorBoard |
| [`scripts/04_upload_results.py`](scripts/04_upload_results.py) | Online | Push student + PCA teacher + artifacts to Hub |
