# VUM: Vision Unified Models

A unified Vision Transformer that jointly performs **masked image modeling** (understanding) and **flow-matching diffusion** (generation) within a single architecture. VUM uses a shared encoder with task-specific decoders to eliminate gradient conflict between the two objectives.

## Architecture

- **Shared Encoder**: ViT with RMSNorm, SwiGLU MLP, RoPE, and Flash Attention
- **Decoder (MAE)**: Lightweight ViT decoder for masked patch reconstruction
- **Decoder (Generation)**: ViT decoder for diffusion denoising
- **Training**: Each batch is split into MAE samples (t=0) and diffusion samples (t>0) with configurable ratios

| Model | Encoder Blocks | Decoder Blocks | Hidden Dim | Heads |
|-------|---------------|---------------|-----------|-------|
| VUM-B | 12 | 12 | 768 | 16 |
| VUM-L | 24 | 24 | 1024 | 16 |

## Installation

```bash
conda create --name vum python=3.12
conda activate vum

pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install diffusers transformers accelerate
pip install flash-attn --no-build-isolation
pip install omegaconf termcolor einops safetensors
pip install git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity
pip install -e .
```

Optional (Flash Attention 3 for Hopper GPUs):
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper && python setup.py install
```

## Training

```bash
bash scripts/train.sh
```

## Sampling

```bash
bash scripts/sample.sh
```

## Evaluation

### Generation (FID/IS)


```bash
python projects/evaluate/adm_evaluator.py <ref_stats.npz>  <generated_samples>
```

### Understanding

**Linear Probing:**
```bash
bash scripts/evaluations/lp.sh
```

**ImageNet Fine-tuning:**
```bash
bash scripts/evaluations/ft_B.sh
```

### Downstream Tasks

**COCO Object Detection (Mask R-CNN):**
```bash
cd projects/evaluate/det
bash run_train.sh
```

**ADE20K Semantic Segmentation (UperNet):**
```bash
cd projects/evaluate/seg
bash run_train.sh
```

## Project Structure

```
VUM/
├── configs/
│   ├── vum_base.yaml           # ViT-Base config
│   └── vum_large.yaml          # ViT-Large config
├── vum/                        # Core library
│   ├── models/
│   │   ├── vum.py              # VUM model (encoder + dual decoders)
│   │   ├── modules/layers.py   # RMSNorm, SwiGLU, Attention, PatchEmbed
│   │   └── modules/rope.py     # Rotary position embeddings
│   ├── schedulers/
│   │   ├── hybrid_sched.py     # MAE + Diffusion scheduler
│   │   └── transport.py        # Flow-matching interpolant
│   ├── data/                   # ImageNet dataloader
│   └── utils/                  # Training utilities (DDP, EMA, optimizers)
├── projects/
│   ├── train/trainer.py        # Training script
│   ├── sample/sample_c2i_ddp.py  # Class-conditional sampling
│   └── evaluate/               # Evaluation scripts
│       ├── linear_probing.py
│       ├── imagenet_ft.py
│       ├── det/                # COCO detection
│       └── seg/                # ADE20K segmentation
└── setup.py
```
