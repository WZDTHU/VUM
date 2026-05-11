#!/usr/bin/env python3
"""
Linear probing evaluation script.

Usage:
    torchrun --nproc_per_node=8 projects/evaluate/linear_probing.py \
        --config path/to/config.yaml \
        --ckpt path/to/checkpoint \
        --imagenet_root /path/to/imagenet
"""


from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from vum.data.utils.image_normalize import get_normalize_transform
from vum.utils.misc_utils import (
    instantiate_from_config, init_from_ckpt
)
from vum.utils.train_utils import create_logger, format_time
from vum.models.c2i.jit_ori import JiT, JiTViT



# ============================================================================
# Distributed utilities
# ============================================================================

def is_main_process() -> bool:
    return dist.get_rank() == 0


def get_world_size() -> int:
    return dist.get_world_size()


def get_rank() -> int:
    return dist.get_rank()


# ============================================================================
# Transforms
# ============================================================================

def make_train_transform(crop_size: int = 256, normalize_type: str = "diffusion") -> transforms.Compose:
    """Training transform with RandomResizedCrop. normalize_type: 'mae' | 'diffusion'."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            crop_size,
            scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        get_normalize_transform(normalize_type),
    ])


def make_eval_transform(
    resize_size: int = 256, crop_size: int = 256, normalize_type: str = "diffusion"
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        get_normalize_transform(normalize_type),
    ])


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])

# ============================================================================
# Learning Rate Schedule (MAE/Kaiming style cosine decay with warmup)
# ============================================================================

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_per_epoch, args):
    """Cosine learning rate schedule with warmup (MAE style)."""
    # Warmup
    warmup_epochs = args.warmup_epochs
    if epoch < warmup_epochs:
        lr = args.lr * (step_in_epoch + epoch * total_steps_per_epoch) / (warmup_epochs * total_steps_per_epoch)
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs + step_in_epoch / total_steps_per_epoch) / (args.epochs - warmup_epochs)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ============================================================================
# Feature Extraction Model
# ============================================================================

class FeatureExtractor(nn.Module):
    """Wrapper that extracts single layer features from model."""

    def __init__(self, 
        model, layer_index: int, autocast_dtype: torch.dtype, 
        return_class_token: bool = False, noise_level: float = 0.0
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.layer_index = layer_index
        self.autocast_dtype = autocast_dtype
        self.return_class_token = return_class_token
        self.noise_level = noise_level

    
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from specified layer."""
        with torch.inference_mode():
            if hasattr(self.model, 'forward_features'):
                return self.model.forward_features(images, return_class_token=self.return_class_token).float()
            elif hasattr(self.model, 'get_intermediate_layers_feature'):
                # Use the method that returns features
                features = self.model.get_intermediate_layers_feature(
                    images, t=self.noise_level, n=1, return_class_token=self.return_class_token
                )
                return features.float()
            else:
                raise NotImplementedError("Backbone must have forward_features or forward_encoder method")


# ============================================================================
# Linear Classifier
# ============================================================================

class LinearClassifier(nn.Module):
    """Linear classifier on top of frozen features."""

    def __init__(self, in_dim: int, num_classes: int = 1000):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim, affine=False)
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


# ============================================================================
# Infinite Sampler
# ============================================================================

class InfiniteSampler(Sampler):
    """Wraps another sampler to yield an infinite stream of indices."""

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.epoch = 0

    def __iter__(self):
        while True:
            if hasattr(self.sampler, 'set_epoch'):
                self.sampler.set_epoch(self.epoch)
            yield from iter(self.sampler)
            self.epoch += 1

    def __len__(self):
        return int(1e18)


# ============================================================================
# Training
# ============================================================================

@torch.no_grad()
def evaluate(
    feature_model: nn.Module,
    classifier: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate classifier and return accuracy (multi-GPU aware)."""
    classifier.eval()

    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = feature_model(images)
        logits = classifier(features)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Aggregate across processes
    total_tensor = torch.tensor([total], device=device, dtype=torch.long)
    correct_tensor = torch.tensor([correct], device=device, dtype=torch.long)
    dist.all_reduce(total_tensor)
    dist.all_reduce(correct_tensor)
    total = total_tensor.item()
    correct = correct_tensor.item()

    accuracy = 100.0 * correct / total
    return accuracy


# ============================================================================
# Main
# ============================================================================

def test_linear_probing(args):
    """Run linear probing evaluation."""
    # Initialize DDP
    dist.init_process_group(backend='nccl')
    rank = get_rank()
    num_processes = get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')

    # Create output directories
    log_dir = os.path.join(args.output_dir, 'logs')
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    dist.barrier()

    # Setup logger
    logger = create_logger(log_dir)

        

    cudnn.benchmark = True

    # Load model
    if rank == 0:
        logger.info("Loading model...")
    cfg = OmegaConf.load(args.config)
    model_config = cfg.model

    model = instantiate_from_config(model_config.network).to(device=device)
    init_from_ckpt(model, checkpoint_dir=args.ckpt, ignore_keys=None, verbose=(rank == 0))

    # Convert JiT → JiTViT for linear probing (strip AdaLN, keep ViT backbone)
    if isinstance(model, JiT):
        if rank == 0:
            logger.info(f"Converting JiT to JiTViT (num_layers={args.num_layers})...")
        model = JiTViT.from_jit_model(
            model, num_layers=args.num_layers, verbose=(rank == 0)
        ).to(device=device)

    model.eval()
    model = torch.compile(model, fullgraph=False)

    # Get precision dtype
    if args.precision in ('bf16', 'bfloat16'):
        autocast_dtype = torch.bfloat16
    elif args.precision in ('fp16', 'float16'):
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    # Create feature extractor
    feature_model = FeatureExtractor(model, args.layer_index, autocast_dtype, args.return_class_token, args.noise_level)

    # Setup datasets
    imagenet_train_path = os.path.join(args.imagenet_root, "train")
    imagenet_val_path = os.path.join(args.imagenet_root, "val")

    train_transform = make_train_transform(args.crop_size, args.normalize_type)
    eval_transform = make_eval_transform(args.crop_size, args.crop_size, args.normalize_type)

    if rank == 0:
        logger.info(f"Loading train dataset: {imagenet_train_path}")
    train_dataset = ImageFolder(root=imagenet_train_path, transform=train_transform)
    num_classes = len(train_dataset.classes)

    if rank == 0:
        logger.info(f"Loading val dataset: {imagenet_val_path}")
        logger.info(f"Number of classes: {num_classes}")
    val_dataset = ImageFolder(root=imagenet_val_path, transform=eval_transform)

    # Create dataloaders with DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    infinite_sampler = InfiniteSampler(train_sampler)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=infinite_sampler,
        num_workers=args.num_workers,
        # pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        # pin_memory=True,
    )

    # Get sample output for dimension calculation
    if rank == 0:
        logger.info("Setting up classifier...")
    sample_image = train_dataset[0][0].unsqueeze(0).to(device)
    sample_output = feature_model(sample_image)
    in_dim = sample_output.shape[-1]
    if rank == 0:
        logger.info(f"Feature dimension: {in_dim}")

    # Create linear classifier with DDP
    classifier = LinearClassifier(in_dim, num_classes).to(device)
    classifier = nn.parallel.DistributedDataParallel(
        classifier, device_ids=[rank % torch.cuda.device_count()]
    )

    # Compute scaled learning rate
    total_batch_size = args.batch_size * num_processes
    args.lr = args.learning_rate * total_batch_size / args.base_batch_size # MAE style learning rate scaling
    args.min_lr = 0.0
    
    # Setup optimizer (no scheduler - we use manual lr adjustment)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
        classifier.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999), 
        weight_decay=0.01
    )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
        classifier.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=0.0 
    )
    elif args.optimizer == 'lars':
        optimizer = LARS(
            classifier.parameters(), 
            lr=args.lr, 
            weight_decay=0
        )
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")
    criterion = nn.CrossEntropyLoss()



    args.epoch_length = len(train_dataset) // total_batch_size
    max_iter = args.epochs * args.epoch_length

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Linear Probing Evaluation")
        logger.info("=" * 60)
        logger.info(f"Checkpoint: {args.ckpt}")
        logger.info(f"ImageNet root: {args.imagenet_root}")
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"World size: {num_processes}")
        logger.info(f"Precision: {args.precision}")
        logger.info(f"Batch size per GPU: {args.batch_size}, Total: {total_batch_size}")
        logger.info(f"Layer index: {args.layer_index}")
        logger.info(f"Num layers (JiT): {args.num_layers}")
        logger.info(f"Base batch size: {args.base_batch_size}, Scaled LR: {args.lr:.6f}")
        logger.info(f"Min LR: {args.min_lr}")
        logger.info(f"Warmup epochs: {args.warmup_epochs}")
        logger.info(f"Epochs: {args.epochs}, Epoch length: {args.epoch_length}")
        logger.info("=" * 60)
        logger.info("***** Running Linear Probing *****")
        logger.info(f"  Total optimization steps = {max_iter}")
        logger.info("=" * 60)
        

    # Training loop
    best_accuracy = 0.0
    global_step = 0
    start_time = timer()
    train_iter = iter(train_loader)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        classifier.train()

        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx in range(args.epoch_length):
            time_last_log = timer()

            # Adjust learning rate (MAE style cosine decay with warmup)
            lr = adjust_learning_rate(optimizer, epoch, batch_idx, args.epoch_length, args)

            images, labels = next(train_iter)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            features = feature_model(images)
            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            time_delta = timer() - time_last_log
            time_elapsed = timer() - start_time
            it_per_sec = 1 / time_delta
            eta_seconds = (max_iter - global_step) / it_per_sec

            # Logging (trainer_c2i style)
            if rank == 0 and global_step % 50 == 0:
                progress = f"{global_step}/{max_iter}"
                eta_str = f"{format_time(time_elapsed)}<{format_time(eta_seconds)}"
                rate_str = f"{it_per_sec:.2f}it/s"
                avg_loss = epoch_loss / epoch_steps
                logger.info(
                    f"steps: {progress} [{eta_str}, {rate_str}], "
                    f"lr={lr:.2e}, loss={loss.item():.4f}, avg_loss={avg_loss:.4f}"
                )

        # Log epoch summary
        if rank == 0:
            avg_epoch_loss = epoch_loss / epoch_steps
            logger.info(f"Epoch {epoch} finished, avg_loss={avg_epoch_loss:.4f}, running evaluation...")

        # Evaluate at end of each epoch
        accuracy = evaluate(feature_model, classifier, val_loader, device)

        if accuracy > best_accuracy:
            best_accuracy = accuracy

        if rank == 0:
            logger.info(f"Epoch {epoch} - Val Accuracy: {accuracy:.2f}% (Best: {best_accuracy:.2f}%)")

    # Final results
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Final Results:")
        logger.info(f"  Best Accuracy: {best_accuracy:.2f}%")
        logger.info(f"  Layer Index: {args.layer_index}")
        logger.info(f"  Num Layers (JiT): {args.num_layers}")
        logger.info(f"  Return Class Token: {args.return_class_token}")
        logger.info(f"  Crop Size: {args.crop_size}")
        logger.info(f"  Learning Rate: {args.learning_rate}")
        logger.info("=" * 60)

        # Save results
        results = {
            "best_accuracy": best_accuracy,
            "ckpt": args.ckpt,
            "noise_level": args.noise_level,
            "layer_index": args.layer_index,
            "num_layers": args.num_layers,
            "normalize_type": args.normalize_type,
            "learning_rate": args.learning_rate,
            "scaled_lr": args.lr,
            "epochs": args.epochs,
            "epoch_length": args.epoch_length,
            "return_class_token": args.return_class_token,
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
            "total_batch_size": total_batch_size,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_lr,
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        results_path = os.path.join(args.output_dir, f"results_{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {results_path}")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

    return {"accuracy": best_accuracy}


def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser(description="Linear probing evaluation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--imagenet_root", type=str,
                        default='/path/to/imagenet',
                        # default='/path/to/data',
                        help="Path to ImageNet dataset root")
    parser.add_argument("--output_dir", type=str, default="workdir/c2i/jit/linear_probing_results",
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of warmup epochs")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Crop size")
    parser.add_argument("--normalize_type", type=str, default="diffusion",
                        choices=["mae", "diffusion"],
                        help="Image normalize: 'mae' (ImageNet 0.485/0.229) or 'diffusion' (0.5/0.5)")
    parser.add_argument("--layer_index", type=int, default=1,
                        help="Layer index to extract features from (1=last, 2=second last, etc.)")
    parser.add_argument("--return_class_token", type=str2bool, default="false",
                        help="Whether to return class token")
    parser.add_argument("--noise_level", type=float, default=0.0,
                        help="Noise level added to the image, default to 0.0")
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Number of transformer layers to use from JiT (for layer ablation). "
                             "None means use all layers.")
    parser.add_argument("--base_batch_size", type=int, default=16384,
                        help="Base batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Base learning rate (will be scaled by batch size)")
    parser.add_argument("--min_lr", type=float, default=0.0,
                        help="Minimum learning rate for cosine decay")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd", "lars"],
                        help="Optimizer")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Precision for inference")
    args = parser.parse_args()

    test_linear_probing(args)


if __name__ == "__main__":
    main()

# Example usage:
# torchrun --nproc_per_node=1 projects/evaluate/linear_probing.py \
#     --config workdir/c2i/jit/diffusion_ablation/jit_b_p32_muon/configs/config.yaml \
#     --ckpt workdir/c2i/jit/diffusion_ablation/jit_b_p32_muon/checkpoints/checkpoint-100000/ema_model.safetensors \
#     --imagenet_root /path/to/imagenet \
#     --layer_index -1 \
#     --learning_rate 0.1 \
#     --warmup_epochs 10 \
#     --epochs 90 \
#     --output_dir workdir/c2i/jit/linear_probing_results