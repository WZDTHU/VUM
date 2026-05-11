#!/bin/bash
set -e

export http_proxy=http://10.70.18.5:8412
export https_proxy=http://10.70.18.5:8412

# conda activate vum

GPUS=8
PORT=${PORT:-7964}
WORK_DIR=workdir/det_coco_ugvit_large_1x_lr2.5e-4
CKPT=/path/to/checkpoint.safetensors

cd $(dirname $0)/../..

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py configs/mask_rcnn_ugvit_large_fpn_coco_1x_lr2.5e-4.py \
    --launcher pytorch \
    --work-dir $WORK_DIR \
    --no-validate \
    --cfg-options model.pretrained=$CKPT
