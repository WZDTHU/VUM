import shutil, os, argparse
from copy import deepcopy
from omegaconf import OmegaConf
from contextlib import nullcontext
from timeit import default_timer as timer
import torch
import torch.nn.functional as F
import torch.distributed as dist


from vum.schedulers.transport import Transport
from vum.data import TrainDataLoader
from vum.utils.lr_scheduler import get_scheduler
from vum.utils.optimizer import get_optimizer
from vum.utils.misc_utils import (
    get_obj_from_str, instantiate_from_config, init_from_ckpt
)
from vum.utils.train_utils import (
    setup_ddp, setup_model, update_ema, load_checkpoints, save_checkpoints, create_logger, format_time
)
from vum.utils.gpu_memory_monitor import build_gpu_memory_monitor


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----General Training Arguments----
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="The config file for training.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="t2i_linear_attention",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    args = parser.parse_args()
    return args


def main(args):
    rank, num_processes, device = setup_ddp(args.seed)

    project_dir = args.project_dir
    config = OmegaConf.load(args.config)
    model_config = config.model 
    data_config = config.data
    train_config = config.training
    use_fsdp = getattr(train_config, 'use_fsdp', False)
    grad_ckpt = getattr(train_config, 'gradient_checkpointing', False)

    config_dir = os.path.join(project_dir, 'configs')
    checkpoint_dir = os.path.join(project_dir, 'checkpoints')
    log_dir = os.path.join(project_dir, 'logs')
    sample_dir = os.path.join(project_dir, 'samples')
    if rank == 0:
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        OmegaConf.save(config=config, f=os.path.join(config_dir, "config.yaml"))
    
    logger = create_logger(log_dir)
    
    total_batch_size = (
        data_config.dataloader.batch_size * 
        num_processes * train_config.gradient_accumulation_steps
    )
    learning_rate = (
        train_config.learning_rate * 
        total_batch_size / train_config.learning_rate_base_batch_size
    )
    
    # prepare model, dataloader, optimizer and scheduler
    # For large models, we need to use FSDP2 before moving to GPU to avoid OOM
    # Create model on CPU
    logger.info("Creating model on CPU...")
    model = instantiate_from_config(model_config.network)
    model.train()
    if getattr(model_config, 'ckpt', None) != None:
        # Handle both absolute paths and relative paths
        ckpt_path = model_config.ckpt if os.path.isabs(model_config.ckpt) else os.path.join(checkpoint_dir, model_config.ckpt)
        init_from_ckpt(model, checkpoint_dir=ckpt_path, verbose=True)
    ema_model = deepcopy(model)
    ema_model.train().requires_grad_(False)
    # model parameters
    logger.info(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    projector_params = sum(p.numel() for n, p in model.named_parameters() if 'projector' in n or 'predictor' in n)
    logger.info(f'total_params: {total_params/1e6:.3f}M, trainable_params: {trainable_params/1e6:.3f}M, train ratio: {trainable_params/total_params*100:.3f}%')
    logger.info(f'model_params: {total_params/1e6-projector_params/1e6:.3f}M, projector_params: {projector_params/1e6:.3f}M, projector_ratio: {projector_params/total_params*100:.3f}%')
    # model setup 
    model, ema_model = setup_model(model, ema_model, device, grad_ckpt, use_fsdp)
    
    
    # optimizer
    optimizer = get_optimizer(
        train_config.optimizer.target, model, lr=learning_rate, wd=train_config.optimizer.weight_decay
    )
    # LR Scheduler creation
    lr_scheduler = get_scheduler(
        train_config.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=train_config.lr_warmup_steps,
        num_training_steps=train_config.max_train_steps,
    )

    
    # Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    resume_steps = 0
    if train_config.resume_from_checkpoint:
        # normal read with safety check
        if train_config.resume_from_checkpoint != "latest":
            resume_from = os.path.basename(train_config.resume_from_checkpoint)
        else:   # Get the most recent checkpoint
            dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            resume_from = os.path.join(checkpoint_dir, dirs[-1]) if len(dirs) > 0 else None
        if resume_from is None:
            logger.info(
                f"Checkpoint '{train_config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            train_config.resume_from_checkpoint = None
        else:
            load_checkpoints(resume_from, model, ema_model, optimizer, lr_scheduler, use_fsdp, rank)
            dist.barrier()
            resume_steps = int(resume_from.split("-")[-1]) # gs not calculate the gradient_accumulation_steps
            logger.info(f"Resuming from steps: {resume_steps}")
    
    # train dataloader
    get_train_data_loader = TrainDataLoader(data_config)
    train_data_loader = get_train_data_loader.train_dataloader(
        rank=rank, world_size=num_processes, 
        global_batch_size=total_batch_size, max_steps=train_config.max_train_steps, 
        resume_steps=resume_steps, seed=args.seed
    )
    
    

    # transport 
    transport = Transport(**OmegaConf.to_container(model_config.transport))
    loss_fn = get_obj_from_str(model_config.scheduler.target)(
        transport=transport, **model_config.scheduler.get("params", dict())
    )

    
    
    # last step: compile model
    if getattr(train_config, 'compile_model', False):
        logger.info('compile model for acceleration')
        model = torch.compile(model)
        ema_model = torch.compile(ema_model)
    

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # TODO: support wandb tracker
    
    # initialize GPU memory monitor before applying parallelisms to the model
    gpu_memory_monitor = build_gpu_memory_monitor(logger)
    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

    # 15. Train!
    logger.info("  ***** Running training *****")
    logger.info(f"  Dataset Length = {len(get_train_data_loader)}")
    logger.info(f"  Num steps each epoch = {len(get_train_data_loader)/total_batch_size}")
    logger.info(f"  Instantaneous batch size per device = {data_config.dataloader.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Optimizer = {train_config.optimizer.target}")
    logger.info(f"  Maximum learning rate = {learning_rate}")
    logger.info(f"  LR scheduelr = {train_config.lr_scheduler}, with warmup steps = {train_config.lr_warmup_steps}")
    logger.info(f"  Gradient Accumulation steps = {train_config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {train_config.max_train_steps}, epochs={train_config.max_train_steps/(len(get_train_data_loader)/total_batch_size)}")
    logger.info(f"  GPU memory usage for model: {gpu_mem_stats.max_reserved_gib:.2f}GiB ({gpu_mem_stats.max_reserved_pct:.2f}%)")

    # CFG dropout for class labels
    dropout_prob = getattr(model_config, 'class_dropout_prob', 0.0)
    num_classes = model_config.network.params.num_classes
        
    train_data_iter = iter(train_data_loader)
    start_time = timer()
    for global_steps in range(resume_steps+1, train_config.max_train_steps+1):
        time_last_log = timer()
        batch = next(train_data_iter)
        time_data_load = timer() - time_last_log

        # === Data Processing ===
        data_proc_start = timer()
        
        # Load images and labels for mae
        batch_image = batch['image'].to(device, torch.float32, non_blocking=True)
        batch_label = batch['label'].to(device, torch.int, non_blocking=True)
        

        
        if dropout_prob > 0:
            drop_ids = torch.rand(batch_label.shape[0], device=device) < dropout_prob
            dropped_label = torch.where(drop_ids, num_classes, batch_label)
        else:
            dropped_label = batch_label
        
        null_label = torch.ones_like(dropped_label) * num_classes
        
        time_data_proc = timer() - data_proc_start
        
        # === Feature Encoding ===
        feat_enc_start = timer()
        h_target = None
        time_feat_enc = timer() - feat_enc_start
        
        # === Build Model Kwargs ===
        model_kwargs = dict(y=dropped_label)
        null_kwargs = dict(y=null_label)
        
        if global_steps <= train_config.lr_warmup_steps:
            epochs = 0
        else:
            epochs = (global_steps - train_config.lr_warmup_steps) // len(get_train_data_loader)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, loss_dict = loss_fn(
                model, ema_model, batch_image, h_target, model_kwargs, null_kwargs, epochs
            )
        
        sync_now = global_steps % train_config.gradient_accumulation_steps == 0
        if use_fsdp:    # FSDP2: 显式设置状态，然后给一个空的 context 让下面的代码跑通
            model.set_requires_gradient_sync(sync_now)
            sync_ctx = nullcontext()
        else:           # DDP / FSDP1: 使用 no_sync 上下文管理器
            sync_ctx = nullcontext() if sync_now else model.no_sync()
        with sync_ctx:
            loss = loss / train_config.gradient_accumulation_steps
            loss.backward()
        if sync_now:
            if use_fsdp:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.max_grad_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), train_config.max_grad_norm
                )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            update_ema(ema_model, model, train_config.ema_decay)
        
        time_delta = timer() - time_last_log
        time_data_load_pct = 100 * time_data_load / time_delta
        time_data_proc_pct = 100 * time_data_proc / time_delta
        time_feat_enc_pct = 100 * time_feat_enc / time_delta
        time_elapsed = timer() - start_time 
        it_per_sec = 1 / time_delta
        eta_seconds = (train_config.max_train_steps - global_steps) / it_per_sec
        
        logs = {"lr": lr_scheduler.get_last_lr()[0]}
        logs.update(loss_dict)
        if sync_now and train_config.max_grad_norm > 0:
            logs.update({'grad_norm': grad_norm.item()})
        logs.update({# time and status
            "data_load(s)": time_data_load,
            "data_load(%)": time_data_load_pct,
            "data_proc(s)": time_data_proc,
            "data_proc(%)": time_data_proc_pct,
            "feat_enc(s)": time_feat_enc,
            "feat_enc(%)": time_feat_enc_pct,
        })    
            
        if rank == 0:
            progress = f"{global_steps}/{train_config.max_train_steps}"
            eta_str = f"{format_time(time_elapsed)}<{format_time(eta_seconds)}"
            rate_str = f"{it_per_sec:.2f}it/s"        
            log_parts = []
            for k, v in logs.items():
                if k == "lr": log_parts.append(f"{k}={v:.2e}")
                else: log_parts.append(f"{k}={v:.3f}")
            log_metrics_str = ", ".join(log_parts)
            logger.info(f"steps: {progress} [{eta_str}, {rate_str}], {log_metrics_str}")
            
        
        if global_steps % train_config.checkpointing_steps == 0:
            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
            dist.barrier()
            if rank == 0 and train_config.checkpoints_total_limit is not None:
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                if len(checkpoints) >= train_config.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - train_config.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(checkpoint_dir, removing_checkpoint)
                        try:
                            shutil.rmtree(removing_checkpoint)
                        except:
                            pass
            dist.barrier()
            save_dir = os.path.join(checkpoint_dir, f"checkpoint-{global_steps}")
            save_checkpoints(save_dir, model, ema_model, optimizer, lr_scheduler, use_fsdp, rank)
            if rank == 0: logger.info(f"Saved state to {save_dir}")
            torch.cuda.empty_cache()
            dist.barrier()
            

        if global_steps in train_config.checkpoint_list or global_steps == train_config.max_train_steps:
            dist.barrier()
            save_dir = os.path.join(checkpoint_dir, f"save-checkpoint-{global_steps}")
            save_checkpoints(save_dir, model, ema_model, optimizer, lr_scheduler, use_fsdp, rank)
            if rank == 0: logger.info(f"Saved state to {save_dir}")
            torch.cuda.empty_cache()
            dist.barrier()
            
        
        
    # Create the pipeline using using the trained modules and save it.
    dist.barrier()
    if rank == 0:
        print("Training is Done!")
    dist.barrier()
    dist.destroy_process_group()
    
    
    

if __name__ == "__main__":
    args = parse_args()
    main(args)


    