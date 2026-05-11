#!/usr/bin/env python3
"""
ImageNet finetuning script (MAE-style and JiT).

Supports both MAE-style models and JiT models. For JiT, converts to JiTViT
(strips AdaLN, keeps ViT backbone) before finetuning.

Reference: https://github.com/facebookresearch/mae

Usage:
    torchrun --nproc_per_node=8 projects/evaluate/imagenet_ft.py \
        --config path/to/config.yaml \
        --finetune path/to/checkpoint \
        --imagenet_root /path/to/imagenet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, datasets

from timm.data import create_transform
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy

from vum.data.utils.image_normalize import get_normalize_transform
from vum.utils.misc_utils import (
    instantiate_from_config, init_from_ckpt
)
from vum.utils.train_utils import create_logger, format_time
from vum.utils.optim_factory import param_groups_lrd
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


def all_reduce_mean(x):
    """Reduce and average across all processes."""
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    return x


# ============================================================================
# Transforms (MAE-style with strong augmentation)
# ============================================================================

def build_train_transform(args):
    """Build training transform with MAE-style augmentation.
    
    Includes:
    - RandomResizedCrop
    - RandomHorizontalFlip
    - AutoAugment (rand-m9-mstd0.5-inc1)
    - Random Erase
    """
    mean, std = get_normalize_stats(args.normalize_type)
    
    # Use timm's create_transform for consistent augmentation
    transform = create_transform(
        input_size=args.input_size,
        is_training=True,
        color_jitter=args.color_jitter if args.color_jitter else None,
        auto_augment=args.aa,
        interpolation='bicubic',
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        mean=mean,
        std=std,
    )
    return transform


def build_eval_transform(args):
    """Build evaluation transform."""
    mean, std = get_normalize_stats(args.normalize_type)
    
    # MAE uses 224 crop for 224 input, we follow the same pattern
    crop_pct = 224 / 256  # Standard crop ratio
    size = int(args.input_size / crop_pct)
    
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_normalize_stats(normalize_type: str):
    """Get normalization statistics."""
    if normalize_type == "mae":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif normalize_type == "diffusion":
        return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        raise ValueError(f"Unknown normalize_type: {normalize_type}")


# ============================================================================
# Learning Rate Schedule (MAE-style cosine decay with warmup)
# ============================================================================

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup (per-epoch)."""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_learning_rate_per_iter(optimizer, data_iter_step, total_steps, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup (per-iteration)."""
    progress = data_iter_step / total_steps + epoch
    if progress < args.warmup_epochs:
        lr = args.lr * progress / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (progress - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


# ============================================================================
# Layer-wise Learning Rate Decay
# ============================================================================

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay.
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    
    # Get number of layers from model
    num_layers = get_num_layers(model) + 1
    
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # No decay for bias and norm
        if p.ndim == 1 or n.endswith(".bias") or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
        
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)
        
        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
            
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
        
        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    
    return list(param_groups.values())


def get_num_layers(model):
    """Get number of transformer layers in the model."""
    # Try different common attribute names
    if hasattr(model, 'blocks'):
        return len(model.blocks)
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
        return len(model.backbone.blocks)
    elif hasattr(model, 'layers'):
        return len(model.layers)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'blocks'):
        return len(model.encoder.blocks)
    elif hasattr(model, 'model') and hasattr(model.model, 'blocks'):
        return len(model.model.blocks)
    else:
        # Default fallback
        return 12


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id.
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    Supports both MAE (patch_embed) and JiT (x_embedder) naming.
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif 'patch_embed' in name or 'x_embedder' in name:
        return 0
    elif 'blocks.' in name:
        # Extract block number (handles backbone.blocks.0, model.blocks.0, etc.)
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part == 'blocks' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1]) + 1
                except ValueError:
                    pass
        return num_layers
    else:
        return num_layers


# ============================================================================
# Vision Transformer for Classification (Flexible Wrapper)
# ============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class ViTForImageClassification(nn.Module):
    """
    Flexible wrapper for Vision Transformer models for image classification.
    
    Supports:
    - Global average pooling or cls_token for classification
    - Drop path rate (stochastic depth)
    - Mixed precision inference
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 1000,
        global_pool: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        autocast_dtype: torch.dtype = torch.bfloat16,
        noise_level: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.global_pool = global_pool
        self.autocast_dtype = autocast_dtype
        self.noise_level = noise_level
        
        # Get embedding dimension from backbone
        embed_dim = self._get_embed_dim()
        
        # Classification head
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes)
        
        
        # Initialize head
        self._init_head()
        
        # Set drop path rate if supported
        if drop_path_rate > 0:
            self._set_drop_path_rate(drop_path_rate)
    
    def _get_embed_dim(self) -> int:
        """Get embedding dimension from backbone."""
        if hasattr(self.backbone, 'embed_dim'):
            return self.backbone.embed_dim
        elif hasattr(self.backbone, 'hidden_size'):
            return self.backbone.hidden_size
        elif hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'hidden_size'):
            return self.backbone.config.hidden_size
        else:
            # Try to infer from a forward pass
            raise ValueError("Cannot determine embed_dim from backbone")
    
    def _init_head(self):
        """Initialize classification head with small weights."""
        nn.init.trunc_normal_(self.head.weight, std=2e-5)
        nn.init.zeros_(self.head.bias)
    
    def _set_drop_path_rate(self, drop_path_rate: float):
        """Set drop path rate for all blocks."""
        if hasattr(self.backbone, 'blocks'):
            num_blocks = len(self.backbone.blocks)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
            for i, block in enumerate(self.backbone.blocks):
                block.drop_path1 = DropPath(dpr[i])
                block.drop_path2 = DropPath(dpr[i])
                
    
    def no_weight_decay(self):
        """Return parameter names that should not have weight decay."""
        no_wd = {'pos_embed', 'cls_token'}
        if hasattr(self.backbone, 'no_weight_decay'):
            no_wd.update(self.backbone.no_weight_decay())
        return no_wd
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        # Check if backbone has get_intermediate_layers_feature method
        if hasattr(self.backbone, 'get_intermediate_layers_feature'):
            # Use the method that returns features
            features = self.backbone.get_intermediate_layers_feature(
                x, t=self.noise_level, n=1, return_class_token=not self.global_pool
            )
            return features
        elif hasattr(self.backbone, 'forward_features'):
            return self.backbone.forward_features(x, return_class_token=not self.global_pool)
        else:
            raise NotImplementedError("Backbone must have forward_features or forward_encoder method")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification."""
        with torch.amp.autocast(device_type='cuda', dtype=self.autocast_dtype):
            features = self.forward_features(x)
        
        features = features.float()  # Convert back to fp32 for head
        
        x = self.fc_norm(features)
        x = self.head_drop(x)
        x = self.head(x)
        return x


# ============================================================================
# Native Scaler for Mixed Precision Training
# ============================================================================

class NativeScalerWithGradNormCount:
    """Native AMP scaler with gradient norm tracking."""
    state_dict_key = "amp_scaler"
    
    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')
    
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm
    
    def state_dict(self):
        return self._scaler.state_dict()
    
    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """Get gradient norm."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    logger=None,
    args=None,
):
    """Train for one epoch."""
    model.train()
    
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    
    total_loss = 0.0
    num_batches = 0
    
    for data_iter_step, (samples, targets) in enumerate(data_loader):
        # Per-iteration LR schedule
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate_per_iter(optimizer, data_iter_step // accum_iter, len(data_loader) // accum_iter, epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        with torch.amp.autocast('cuda', dtype=args.autocast_dtype):
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        loss /= accum_iter
        grad_norm = loss_scaler(
            loss, optimizer, clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        
        total_loss += loss_value
        num_batches += 1
        
        # Logging
        if logger is not None and is_main_process() and data_iter_step % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch [{epoch}] Step [{data_iter_step}/{len(data_loader)}] loss={loss_value:.4f} lr={lr:.2e} grad_norm={grad_norm:.2e}")
    
    # Synchronize metrics
    avg_loss = all_reduce_mean(total_loss / num_batches)
    
    return {"loss": avg_loss}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    """Evaluate model on validation set."""
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    num_samples = 0
    
    for batch in data_loader:
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', dtype=args.autocast_dtype):
            output = model(images)
            loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_acc1 += acc1.item() * batch_size
        total_acc5 += acc5.item() * batch_size
        num_samples += batch_size
    
    # Gather from all processes
    total_loss_tensor = torch.tensor([total_loss], device=device)
    total_acc1_tensor = torch.tensor([total_acc1], device=device)
    total_acc5_tensor = torch.tensor([total_acc5], device=device)
    num_samples_tensor = torch.tensor([num_samples], device=device)
    
    dist.all_reduce(total_loss_tensor)
    dist.all_reduce(total_acc1_tensor)
    dist.all_reduce(total_acc5_tensor)
    dist.all_reduce(num_samples_tensor)
    
    avg_loss = total_loss_tensor.item() / num_samples_tensor.item()
    avg_acc1 = total_acc1_tensor.item() / num_samples_tensor.item()
    avg_acc5 = total_acc5_tensor.item() / num_samples_tensor.item()
    
    return {"loss": avg_loss, "acc1": avg_acc1, "acc5": avg_acc5}


# ============================================================================
# Remove Decoder Parameters (for finetuning encoder-only)
# ============================================================================

def remove_decoder_params(model: nn.Module, verbose: bool = False, logger=None):
    """
    Remove decoder-related parameters from the model.
    This is useful for MAE-style models where only the encoder is needed for finetuning.
    
    Removes parameters with names containing:
    - decoder_embed, decoder_pos_embed, decoder_blocks, decoder_norm, decoder_pred
    - mask_token
    """
    print(model)
    decoder_keywords = [
        'mask_token',
        't_embedder',
        'predictor',
        'projector',
        'decoder',
        'decoder_embed',
        'decoder_pos_embed',
        'decoder_blocks',
        'decoder_norm',
        'decoder_pred',
        'decoder_final',
    ]
    if hasattr(model, 'cls_token'):
        decoder_keywords.append('y_embedder')
    
    removed_params = []
    removed_size = 0
    
    for name in list(model._parameters.keys()):
        if any(kw in name for kw in decoder_keywords):
            param = model._parameters[name]
            if param is not None:
                removed_size += param.numel()
            del model._parameters[name]
            removed_params.append(name)
    
    for name in list(model._buffers.keys()):
        if any(kw in name for kw in decoder_keywords):
            del model._buffers[name]
            removed_params.append(name)
    
    # Remove decoder modules
    for name in list(model._modules.keys()):
        if any(kw in name for kw in decoder_keywords):
            module = model._modules[name]
            if module is not None:
                for p in module.parameters():
                    removed_size += p.numel()
            del model._modules[name]
            removed_params.append(name)
    
    if verbose and logger is not None:
        if removed_params:
            logger.info(f"Removed decoder parameters: {removed_params}")
            logger.info(f"Removed {removed_size / 1e6:.2f}M parameters")
        else:
            logger.info("No decoder parameters found to remove")
    elif verbose:
        if removed_params:
            print(f"Removed decoder parameters: {removed_params}")
            print(f"Removed {removed_size / 1e6:.2f}M parameters")


# ============================================================================
# Checkpoint Utilities
# ============================================================================

def save_checkpoint(args, epoch, model, model_without_ddp, optimizer, loss_scaler, best_acc1):
    """Save training checkpoint."""
    if not is_main_process():
        return
    
    checkpoint = {
        'epoch': epoch,
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': loss_scaler.state_dict(),
        'best_acc1': best_acc1,
        'args': vars(args),
    }
    
    # Save latest
    torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))
    
    # Save best
    if hasattr(args, 'is_best') and args.is_best:
        torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_best.pth'))


def load_checkpoint(args, model_without_ddp, optimizer, loss_scaler):
    """Load checkpoint for resuming training."""
    if not args.resume:
        return 0, 0.0
    
    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_scaler.load_state_dict(checkpoint['scaler'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_acc1 = checkpoint.get('best_acc1', 0.0)
    
    return start_epoch, best_acc1


# ============================================================================
# Main
# ============================================================================

def main_finetune(args):
    """Run ImageNet finetuning."""
    # Initialize DDP
    dist.init_process_group(backend='nccl')
    rank = get_rank()
    num_processes = get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    
    # Set seed for reproducibility
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    model_name = args.finetune.split('/')[-2].split('-')[-1] if args.finetune else 'scratch'
    bool2str = lambda x: 'global_pool' if x else 'cls_token'
    suffix = f'ft_{bool2str(args.global_pool)}'
    if args.num_layers is not None:
        suffix += f'_L{args.num_layers}'
    args.output_dir = os.path.join(args.output_dir, model_name, suffix)
    log_dir = os.path.join(args.output_dir, 'logs')
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
    dist.barrier()
    
    # Setup logger
    logger = create_logger(log_dir)
    
    cudnn.benchmark = True
    
    # ========== Build datasets ==========
    train_transform = build_train_transform(args)
    eval_transform = build_eval_transform(args)
    
    imagenet_train_path = os.path.join(args.imagenet_root, "train")
    imagenet_val_path = os.path.join(args.imagenet_root, "val")
    
    if rank == 0:
        logger.info(f"Loading train dataset: {imagenet_train_path}")
    train_dataset = datasets.ImageFolder(imagenet_train_path, transform=train_transform)
    
    if rank == 0:
        logger.info(f"Loading val dataset: {imagenet_val_path}")
    val_dataset = datasets.ImageFolder(imagenet_val_path, transform=eval_transform)
    
    num_classes = len(train_dataset.classes)
    
    # Samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # ========== Build Mixup ==========
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        if rank == 0:
            logger.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=num_classes,
        )
    
    # ========== Build model ==========
    if rank == 0:
        logger.info("Loading backbone model...")
    
    cfg = OmegaConf.load(args.config)
    model_config = cfg.model
    
    backbone = instantiate_from_config(model_config.network).to(device=device)
    
    # Load pretrained weights
    if args.finetune:
        init_from_ckpt(backbone, checkpoint_dir=args.finetune, ignore_keys=None, verbose=(rank == 0))
    
    # JiT: convert to JiTViT (strip AdaLN, keep ViT backbone). MAE: remove decoder params.
    if isinstance(backbone, JiT):
        if rank == 0:
            logger.info(f"Converting JiT to JiTViT (num_layers={args.num_layers})...")
        backbone = JiTViT.from_jit_model(
            backbone, num_layers=args.num_layers, verbose=(rank == 0)
        ).to(device=device)
    else:
        remove_decoder_params(backbone, verbose=(rank == 0), logger=logger)
    
    # Get precision dtype
    if args.precision in ('bf16', 'bfloat16'):
        autocast_dtype = torch.bfloat16
    elif args.precision in ('fp16', 'float16'):
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32
    args.autocast_dtype = autocast_dtype
    
    # Create classification model
    model = ViTForImageClassification(
        backbone=backbone,
        num_classes=num_classes,
        global_pool=args.global_pool,
        drop_path_rate=args.drop_path,
        autocast_dtype=autocast_dtype,
        noise_level=args.noise_level,
    ).to(device)
    
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if rank == 0:
        logger.info(f"Number of params: {n_parameters / 1.e6:.2f}M")
    
    # ========== Compute effective batch size and LR ==========
    eff_batch_size = args.batch_size * args.accum_iter * num_processes
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    
    if rank == 0:
        logger.info(f"Base LR: {args.lr * 256 / eff_batch_size:.2e}")
        logger.info(f"Actual LR: {args.lr:.2e}")
        logger.info(f"Accumulate grad iterations: {args.accum_iter}")
        logger.info(f"Effective batch size: {eff_batch_size}")
    
    # ========== DDP ==========
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank % torch.cuda.device_count()])
    model_without_ddp = model.module
    
    # ========== Build optimizer with layer-wise LR decay ==========
    param_groups = param_groups_lrd(
        model_without_ddp,
        weight_decay=args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScalerWithGradNormCount()
    
    # ========== Build criterion ==========
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    if rank == 0:
        logger.info(f"Criterion: {criterion}")
    
    # ========== Resume from checkpoint ==========
    start_epoch, best_acc1 = load_checkpoint(args, model_without_ddp, optimizer, loss_scaler)
    
    # ========== Logging ==========
    if rank == 0:
        logger.info("=" * 60)
        logger.info("ImageNet Finetuning (MAE-style)")
        logger.info("=" * 60)
        logger.info(f"Config: {args.config}")
        logger.info(f"Pretrained: {args.finetune}")
        logger.info(f"ImageNet root: {args.imagenet_root}")
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"World size: {num_processes}")
        logger.info(f"Precision: {args.precision}")
        logger.info(f"Input size: {args.input_size}")
        logger.info(f"Batch size per GPU: {args.batch_size}, Total: {eff_batch_size}")
        logger.info(f"Epochs: {args.epochs}, Warmup: {args.warmup_epochs}")
        logger.info(f"LR: {args.lr:.2e}, Min LR: {args.min_lr:.2e}")
        logger.info(f"Layer decay: {args.layer_decay}, Weight decay: {args.weight_decay}")
        logger.info(f"Drop path: {args.drop_path}")
        logger.info(f"Global pool: {args.global_pool}")
        logger.info(f"AutoAugment: {args.aa}")
        logger.info(f"Mixup: {args.mixup}, CutMix: {args.cutmix}")
        logger.info(f"Label smoothing: {args.smoothing}")
        logger.info(f"Random Erase: prob={args.reprob}, mode={args.remode}, count={args.recount}")
        logger.info("=" * 60)
    
    # ========== Eval only ==========
    if args.eval:
        test_stats = evaluate(val_loader, model, device, args)
        if rank == 0:
            logger.info(f"Accuracy of the network on {len(val_dataset)} test images: {test_stats['acc1']:.1f}%")
        dist.barrier()
        dist.destroy_process_group()
        return
    
    # ========== Training loop ==========
    if rank == 0:
        logger.info(f"Start training for {args.epochs} epochs")
    
    start_time = timer()
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, train_loader,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            logger=logger, args=args,
        )
        
        # Save checkpoint
        save_checkpoint(args, epoch, model, model_without_ddp, optimizer, loss_scaler, best_acc1)
        
        # Evaluate
        test_stats = evaluate(val_loader, model, device, args)
        
        if rank == 0:
            logger.info(f"Epoch {epoch} - Acc@1: {test_stats['acc1']:.2f}% Acc@5: {test_stats['acc5']:.2f}% Loss: {test_stats['loss']:.4f}")
        
        # Track best
        if test_stats["acc1"] > best_acc1:
            best_acc1 = test_stats["acc1"]
            args.is_best = True
            save_checkpoint(args, epoch, model, model_without_ddp, optimizer, loss_scaler, best_acc1)
            args.is_best = False
        
        if rank == 0:
            logger.info(f"Max accuracy: {best_acc1:.2f}%")
    
    total_time = timer() - start_time
    
    # ========== Final results ==========
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"Total time: {format_time(total_time)}")
        logger.info(f"Best Acc@1: {best_acc1:.2f}%")
        logger.info("=" * 60)
        
        # Save results
        results = {
            "best_acc1": best_acc1,
            "config": args.config,
            "finetune": args.finetune,
            "epochs": args.epochs,
            "lr": args.lr,
            "layer_decay": args.layer_decay,
            "weight_decay": args.weight_decay,
            "drop_path": args.drop_path,
            "input_size": args.input_size,
            "batch_size": args.batch_size,
            "eff_batch_size": eff_batch_size,
            "mixup": args.mixup,
            "cutmix": args.cutmix,
            "smoothing": args.smoothing,
            "aa": args.aa,
            "reprob": args.reprob,
        }
        if args.num_layers is not None:
            results["num_layers"] = args.num_layers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        results_path = os.path.join(args.output_dir, f"results_{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {results_path}")
    
    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


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
    
    parser = argparse.ArgumentParser(description="ImageNet finetuning (MAE-style)")
    
    # Model
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config")
    parser.add_argument("--finetune", type=str, default='',
                        help="Path to pretrained checkpoint")
    parser.add_argument("--input_size", type=int, default=256,
                        help="Input image size")
    parser.add_argument("--drop_path", type=float, default=0.1,
                        help="Drop path rate")
    parser.add_argument("--global_pool", type=str2bool, default="true",
                        help="Use global pooling (True) or cls_token (False)")
    parser.add_argument("--noise_level", type=float, default=0.0,
                        help="Noise level for diffusion models")
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Number of transformer layers to use from JiT (for layer ablation). "
                             "None means use all layers. Only used when backbone is JiT.")
    
    # Data
    parser.add_argument("--imagenet_root", type=str,
                        default='/path/to/imagenet',
                        # default='/path/to/data',
                        help="Path to ImageNet dataset root")
    parser.add_argument("--normalize_type", type=str, default="mae",
                        choices=["mae", "diffusion"],
                        help="Image normalization type")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--accum_iter", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs")
    
    # Optimizer
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=1e-3,
                        help="Base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--layer_decay", type=float, default=0.75,
                        help="Layer-wise lr decay")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="Gradient clipping norm")
    
    # Augmentation (MAE defaults)
    parser.add_argument("--color_jitter", type=float, default=None,
                        help="Color jitter factor")
    parser.add_argument("--aa", type=str, default='rand-m9-mstd0.5-inc1',
                        help="AutoAugment policy")
    parser.add_argument("--smoothing", type=float, default=0.1,
                        help="Label smoothing")
    
    # Random Erase
    parser.add_argument("--reprob", type=float, default=0.25,
                        help="Random erase probability")
    parser.add_argument("--remode", type=str, default='pixel',
                        help="Random erase mode")
    parser.add_argument("--recount", type=int, default=1,
                        help="Random erase count")
    
    # Mixup
    parser.add_argument("--mixup", type=float, default=0.8,
                        help="Mixup alpha (0 to disable)")
    parser.add_argument("--cutmix", type=float, default=1.0,
                        help="CutMix alpha (0 to disable)")
    parser.add_argument("--cutmix_minmax", type=float, nargs='+', default=None,
                        help="CutMix min/max ratio")
    parser.add_argument("--mixup_prob", type=float, default=1.0,
                        help="Probability of applying mixup or cutmix")
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5,
                        help="Probability of switching mixup to cutmix")
    parser.add_argument("--mixup_mode", type=str, default='batch',
                        help="Mixup mode: batch, pair, or elem")
    
    # Runtime
    parser.add_argument("--output_dir", type=str, default="workdir/imagenet_ft",
                        help="Output directory")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Training precision")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default='',
                        help="Resume from checkpoint")
    parser.add_argument("--eval", action='store_true',
                        help="Evaluation only")
    
    args = parser.parse_args()
    
    main_finetune(args)


if __name__ == "__main__":
    main()

# Example usage (MAE):
# torchrun --nproc_per_node=8 projects/evaluate/imagenet_ft.py \
#     --config path/to/mae_config.yaml \
#     --finetune path/to/mae_checkpoint \
#     --normalize_type mae \
#     --input_size 224 \
#     --batch_size 128 \
#     --epochs 100

# Example usage (JiT):
# torchrun --nproc_per_node=8 projects/evaluate/imagenet_ft.py \
#     --config workdir/c2i/jit/configs/config.yaml \
#     --finetune workdir/c2i/jit/checkpoints/checkpoint-xxx/ema_model.safetensors \
#     --normalize_type diffusion \
#     --input_size 256 \
#     --batch_size 128 \
#     --epochs 100
