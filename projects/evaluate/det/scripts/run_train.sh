#!/bin/bash
# Mask R-CNN + UG-ViT-Large on COCO (12 epochs)
# Usage: bash projects/evaluate/det/run_train.sh
# NOTE: Requires COCO dataset. Set COCO_ROOT below.

set -e

export http_proxy=http://10.70.18.5:8412
export https_proxy=http://10.70.18.5:8412

# conda activate vum

GPUS=8
PORT=${PORT:-7957}
WORK_DIR=workdir/det_coco_ugvit_large_1x
CKPT=/path/to/checkpoint.safetensors

# TODO: Set COCO data root if not at default location (data/coco)
# COCO_ROOT=/path/to/coco

cd $(dirname $0)/../..

OMP_NUM_THREADS=2 python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py configs/mask_rcnn_ugvit_large_fpn_coco_1x.py \
    --launcher pytorch \
    --work-dir $WORK_DIR \
    --no-validate \
    --cfg-options model.pretrained=$CKPT
