#!/bin/bash
# UperNet + UG-ViT-Large on ADE20K (160k iterations)
# Usage: bash projects/evaluate/seg/run_train.sh

set -e

export http_proxy=http://10.70.18.5:8412
export https_proxy=http://10.70.18.5:8412

# conda activate vum

GPUS=1
PORT=${PORT:-7956}
WORK_DIR=workdir/seg_ade20k_ugvit_large_160k
CKPT=/path/to/checkpoint.safetensors

cd $(dirname $0)/../..

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py configs/upernet_ugvit_large_512_slide_160k_ade20k.py \
    --launcher pytorch \
    --work-dir $WORK_DIR \
    --options model.backbone.pretrained=$CKPT
