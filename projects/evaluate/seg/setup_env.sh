#!/bin/bash
# Setup conda environment for semantic segmentation evaluation (mmseg 0.x + UperNet)
# Uses mmcv 1.x API (mmcv.runner, mmcv.parallel, etc.)

set -e

export http_proxy=http://10.70.18.5:8412
export https_proxy=http://10.70.18.5:8412

ENV_NAME=mmseg_eval

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y
source activate $ENV_NAME

# PyTorch 2.0.1 + CUDA 11.8 (well-tested with mmcv-full 1.7.x)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# mmcv-full 1.7.x (provides mmcv.runner, mmcv.parallel, etc.)
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# mmsegmentation 0.30 (compatible with mmcv-full 1.7.x)
pip install mmsegmentation==0.30.0

# Additional dependencies for our backbone
pip install safetensors einops timm==0.9.16
pip install flash-attn --no-build-isolation

# apex for fp16 training
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --global-option="--cpp_ext" --global-option="--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git 2>/dev/null || echo "Apex install failed (optional)"

echo ""
echo "Environment $ENV_NAME setup complete!"
echo "To use: source activate $ENV_NAME"
echo ""
echo "Run segmentation training:"
echo "  cd projects/evaluate/seg"
echo "  bash tools/dist_train.sh configs/upernet_ugvit_large_512_slide_160k_ade20k.py 8"
