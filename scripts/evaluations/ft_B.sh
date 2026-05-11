# conda activate vum
cd $(dirname $0)/../..

torchrun --nproc_per_node=8 projects/evaluate/imagenet_ft.py \
    --config configs/c2i/jit/clean_ablation/vary_mr/maskjit_b_gen25_mae25_hyb50_sep_mm10_linear.yaml \
    --finetune workdir/c2i/jit/clean_ablation/vary_mr/maskjit_b_gen25_mae25_hyb50_sep_mm10_linear/checkpoints/save-checkpoint-100000/ema.safetensors \
    --normalize_type diffusion \
    --global_pool true \
    --input_size 256 \
    --batch_size 128 \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 5e-4 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --reprob 0.25 \
    --output_dir workdir/c2i/jit/clean_ablation/vary_mr/maskjit_b_gen25_mae25_hyb50_sep_mm10_linear/evaluations
