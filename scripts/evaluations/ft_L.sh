# conda activate vum
cd $(dirname $0)/../..

torchrun --nproc_per_node=8 projects/evaluate/imagenet_ft.py \
    --config configs/c2i/jit/large_model/maskjit_l_gen25_mae25_hyb50_sep_mm10_linear_ftp5.yaml \
    --finetune workdir/c2i/jit/large_model/maskjit_l_gen25_mae25_hyb50_sep_mm10_linear_ftp5/checkpoints/save-checkpoint-75000/ema.safetensors \
    --normalize_type diffusion \
    --global_pool true \
    --input_size 256 \
    --batch_size 128 \
    --epochs 50 \
    --warmup_epochs 5 \
    --blr 1e-3 \
    --layer_decay 0.75 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --reprob 0.25 \
    --imagenet_root /path/to/data \
    --output_dir workdir/c2i/jit/large_model/maskjit_l_gen25_mae25_hyb50_sep_mm10_linear_ftp5/evaluations
