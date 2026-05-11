# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL, AutoencoderDC
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import functools
import argparse
from omegaconf import OmegaConf
from vum.schedulers.transport import Transport
from vum.utils.misc_utils import (
    get_obj_from_str, instantiate_from_config, init_from_ckpt
)
from safetensors.torch import load_file

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # setup dtype
    dtype = torch.float32

    # Load model:
    config = OmegaConf.load(args.config)
    model_config = config.model 
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    
    model = instantiate_from_config(model_config.network).to(device=device, dtype=dtype)
    init_from_ckpt(model, checkpoint_dir=args.ckpt, ignore_keys=None, verbose=True)
    model.eval()  # important!
    model = torch.compile(model, fullgraph=False)
    transport = Transport(**OmegaConf.to_container(model_config.transport))
    scheduler = get_obj_from_str(model_config.scheduler.target)(
        transport=transport, **model_config.scheduler.get("params", dict())
    )
    
    # Create folder to save samples:
    model_name = args.ckpt.split('/')[-2].split('-')[-1]
    folder_name = f"{model_name}-{args.height}x{args.width}-T{args.T_min}-{args.T_max}-" \
                  f"Step-{args.num_steps}-sto-{args.stochasticity_ratio}-" \
                  f"cfg-{args.cfg_scale}-{args.guidance_low}-{args.guidance_high}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    class_labels = np.arange(0, args.num_classes).repeat(args.num_fid_samples // args.num_classes)
    class_labels = np.hstack([class_labels, np.zeros(total_samples-args.num_fid_samples)])

    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for i in pbar:
        
        # Sample inputs:
        z = torch.randn(
            (n, model.in_channels, args.height, args.width), 
            device=device, dtype=dtype
        )
        start = rank + total
        end = n * dist.get_world_size() + rank + total
        step = dist.get_world_size()
        y = torch.Tensor(class_labels[start:end:step]).to(dtype=torch.long, device=device)
        # y = torch.randint(0, 1000, (n,), device=device)
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
            
        # can_pass = True
        # for j in range(n):
        #     index = j * dist.get_world_size() + rank + total
        #     if not os.path.exists(f"{sample_folder_dir}/{index:06d}.png"):
        #         can_pass = False
        # if can_pass:
        #     total += global_batch_size
        #     print('total: ', total)
        #     continue

        # Sample images:
        
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            samples = scheduler.sample(
                model, y, y_null, z, 
                T_max=args.T_max, 
                T_min=args.T_min,
                num_steps=args.num_steps, 
                cfg_scale=args.cfg_scale, 
                cfg_low=args.guidance_low, 
                cfg_high=args.guidance_high,
                stochasticity_ratio=args.stochasticity_ratio,
            )[-1]
        samples = (samples + 1) / 2.
        samples = torch.clamp(
            255. * samples, 0, 255
        ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        # create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--config", type=str, default=None, help="Optional config to a SiT checkpoint.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample_dir", type=str, default="")


    # model
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    
    # number of samples
    parser.add_argument("--per_proc_batch_size", type=int, default=32)
    parser.add_argument("--num_fid_samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--T_max",  type=float, default=1.0)
    parser.add_argument("--T_min",  type=float, default=0.0)
    parser.add_argument("--cfg_scale",  type=float, default=1.5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_low", type=float, default=0.)
    parser.add_argument("--guidance_high", type=float, default=1.)
    parser.add_argument("--stochasticity_ratio", type=float, default=0.)

    
    

    args = parser.parse_args()
    main(args)
