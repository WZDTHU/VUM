nnodes=1
node_rank=0
nproc_per_node=1
master_addr="localhost"
master_port=51965

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
    projects/sample/sample_c2i_ddp.py \
    --config configs/maskjit_large.yaml \
    --ckpt /path/to/data \
    --sample_dir workdir/samples \
    --height 256 \
    --width 256 \
    --per_proc_batch_size 128 \
    --T_max 1.0 \
    --T_min 0.0 \
    --cfg_scale 1.5 \
    --num_steps 50 \
    "
bash -c "$TORCHLAUNCHER $CMD";

