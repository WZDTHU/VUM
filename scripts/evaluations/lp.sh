# conda activate vum
cd $(dirname $0)/../..

torchrun --nproc_per_node=8 projects/evaluate/linear_probing.py \
    --config configs/c2i/jit/clean_ablation/vary_mr/maskjit_b_gen25_mae25_hyb50_sep_mm10_linear.yaml \
    --ckpt workdir/c2i/jit/clean_ablation/vary_mr/maskjit_b_gen25_mae25_hyb50_sep_mm10_linear/checkpoints/save-checkpoint-100000/ema.safetensors \
    --layer_index 1 \
    --normalize_type diffusion \
    --return_class_token false \
    --learning_rate 0.001 \
    --optimizer adamw \
    --warmup_epochs 10 \
    --epochs 100 \
    --crop_size 256 \
    --batch_size 2048 \
    --output_dir workdir/c2i/jit/clean_ablation/vary_mr/maskjit_b_gen25_mae25_hyb50_sep_mm10_linear/evaluations
