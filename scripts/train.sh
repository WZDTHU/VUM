nnodes=1
node_rank=0
nproc_per_node=1
master_addr="localhost"
master_port=45689

# conda activate vum
cd $(dirname $0)/../..

TORCHLAUNCHER="torchrun \
    --nnodes=$nnodes \
    --node_rank=$node_rank \
    --nproc_per_node=$nproc_per_node \
    --master_addr=$master_addr \
    --master_port=$master_port \
    "
CMD=" \
    projects/train/trainer.py \
    --config configs/maskjit_base.yaml \
    --project_dir workdir/ \
    --seed 0 \
    "
bash -c "$TORCHLAUNCHER $CMD"