# CMTarget-LLM — Agent Guide

## What this is

Drug–Target Interaction (DTI) prediction via transfer learning. Pre-train on DrugBank, fine-tune on a target dataset (e.g. HIT), then predict. Model = BERT embeddings + optional self-attention + Qwen2-style MoE + cross-attention + scorer (Cosine/MF/GMF).

## Entry point

```
python main.py --task {train_tune,train,tune,predict}
```

All config is CLI args (see `main.py:20-48`). No config file.

## Key commands

```bash
# Pre-train → fine-tune in one run
python main.py --task train_tune -s drugbank -t hit

# Pre-train only (source domain)
python main.py --task train -s drugbank

# Fine-tune only (load pre-trained checkpoint)
python main.py --task tune -t hit --model_path logs/.../checkpoints/pretrain.pt

# Predict
python main.py --task predict --model_path logs/.../checkpoints/fineTune.pt

# Background run (Linux server)
nohup python -u main.py --task tune > myrun.log 2>&1 &
tail -f myrun.log
```

## Data

- CSV at `data/dataset/{source_name}/train.csv` and `test.csv` with columns: `compound`, `protein`, `label`
- Tokenization is done online per-batch via HuggingFace tokenizers (no pre-tokenized cache)

## Models / artifacts

| Path | Contents |
|------|----------|
| `logs/{timestamp}/checkpoints/pretrain.pt` | Best pre-trained checkpoint (by F1) |
| `logs/{timestamp}/checkpoints/fineTune.pt` | Best fine-tuned checkpoint (by F1) |
| `logs/{timestamp}/checkpoints/pretrain_checkpoint_epoch{N}.pt` | Periodic pre-train checkpoint (every `--checkpoint_interval` epochs) |
| `logs/{timestamp}/checkpoints/fintune_checkpoint_epoch{N}.pt` | Periodic fine-tune checkpoint |
| `logs/{timestamp}/{Training,FineTuning,Predicting}/` | Logs, loss/metrics plots, ROC curve, output.csv |

## Model structure

```
protein_ids ─→ BertEmbeddings ─→ SelfAttention* ─→ MoE* ─→ cross_attention* ─→ SelfAttentionPooling ─→ scorer ─→ logits
drug_ids ─────→ BertEmbeddings ─→ SelfAttention* ─→ MoE* ─→ cross_attention* ─→ SelfAttentionPooling ─→ scorer ─→ logits
```

Components marked `*` are controlled by `--use_selfatt`, `--use_moe`, `--use_cross_att` (ablation flags).

## Important quirks

- **CUDA device**: Hardcoded `os.environ["CUDA_VISIBLE_DEVICES"] = '0'` in `main.py:14`. Change for multi-GPU.
- **HF models require pre-download**: `FeatureExtract.py` loads ChemBERTa and ProtBERT with `local_files_only=True`. You must download them first:
  ```bash
  export HF_ENDPOINT="https://hf-mirror.com"
  huggingface-cli download --resume-download seyonec/ChemBERTa-zinc-base-v1 --local-dir ./embedding/ChemBERTa
  huggingface-cli download --resume-download Rostlab/prot_bert --local-dir ./embedding/ProBert
  ```
  **Or** change `local_files_only=True` to `False`.
- **Hardcoded paths in `FeatureExtract.py:27-33`** — `local_model_path` and Word2Vec path are absolute on a specific server. Update for new env.
- **Einops** (`rearrange`, `einops`) used in `embedding/dataset.py` but not listed in `setup.py` — install separately or add.
- **Loss balancing**: `MultiTaskLossWrapper` uses learnable `log_vars` (uncertainty weighting). Must be in optimizer params (see `trainer/CMTargetTrainer.py:60-66`).
- **Scheduler**: `CyclicLR` (not the typical cosine/step).
- **`peft` (LoRA)** used in `FineTunner.py` for parameter-efficient fine-tuning. Only `W_Q, W_K, W_V, gate_proj, up_proj` get LoRA adapters by default.

## Scoring methods (`--score_way`)

| Value | Behavior |
|-------|----------|
| `Cosine` (default) | Cosine similarity of pooled embeddings |
| `MF` | Element-wise product → sum |
| `GMF` | Element-wise product → linear → logits |

## Ablation experiments

Flags `--use_moe`, `--use_selfatt`, `--use_cross_att` toggle modules. Results from prior runs documented in `.dust/README_archive.md`. Removing cross-attention had minimal impact; removing MoE or self-attention hurt more.

## Setup

```bash
conda create -n cmtarget python=3.9
conda activate cmtarget
pip install -e .           # installs deps from setup.py
pip install einops         # missing dep — install separately
```

## No tests / CI

No test suite, no CI, no pre-commit, no linter config, no Dockerfile.
