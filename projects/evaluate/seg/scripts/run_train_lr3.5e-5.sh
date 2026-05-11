#!/bin/bash
set -e

export http_proxy=http://10.70.18.5:8412
export https_proxy=http://10.70.18.5:8412

# conda activate vum

GPUS=8
PORT=${PORT:-7961}
WORK_DIR=workdir/seg_ade20k_ugvit_large_160k_lr3.5e-5
CKPT=/path/to/checkpoint.safetensors

cd $(dirname $0)/../..

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py configs/upernet_ugvit_large_512_slide_160k_ade20k_lr3.5e-5.py \
    --launcher pytorch \
    --work-dir $WORK_DIR \
    --options model.backbone.pretrained=$CKPT
