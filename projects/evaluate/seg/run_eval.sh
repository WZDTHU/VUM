#!/bin/bash
# UperNet + UG-ViT-Large on ADE20K evaluation
# Usage: bash projects/evaluate/seg/run_eval.sh

set -e

export http_proxy=http://10.70.18.5:8412
export https_proxy=http://10.70.18.5:8412

source activate mmseg_eval

GPUS=8
PORT=${PORT:-7958}
WORK_DIR=workdir/seg_ade20k_ugvit_large_160k
CHECKPOINT=${WORK_DIR}/iter_160000.pth

cd $(dirname $0)/../..

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/test.py configs/upernet_ugvit_large_512_slide_160k_ade20k.py \
    $CHECKPOINT \
    --launcher pytorch \
    --eval mIoU
