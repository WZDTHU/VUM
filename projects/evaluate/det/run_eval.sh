#!/bin/bash
# Mask R-CNN + UG-ViT-Large on COCO evaluation
# Usage: bash projects/evaluate/det/run_eval.sh

set -e

export http_proxy=http://10.70.18.5:8412
export https_proxy=http://10.70.18.5:8412

# conda activate vum

GPUS=1
PORT=${PORT:-7959}
WORK_DIR=workdir/det_coco_ugvit_large_1x_lr2.5e-4
CHECKPOINT=${WORK_DIR}/latest.pth

cd $(dirname $0)/../..

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/test.py configs/mask_rcnn_ugvit_large_fpn_coco_1x.py \
    $CHECKPOINT \
    --launcher pytorch \
    --eval bbox segm
