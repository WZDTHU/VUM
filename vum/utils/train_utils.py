import os, torch, random, logging, functools
import torch.distributed as dist
import numpy as np
from collections import OrderedDict
from safetensors.torch import save_file, load_file
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict, set_optimizer_state_dict, get_model_state_dict, get_optimizer_state_dict, StateDictOptions
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

def setup_ddp(global_seed):
    # Setup Distributed Data Parallel (DDP)
    # 1. Retrieve Global Rank, Local Rank, and World Size
    # These environment variables are automatically set by torchrun or most multi-node launchers
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) 
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    # 2. Assign the GPU device: For multi-node setups, local_rank MUST be used
    # to ensure each process on a single node binds to a unique GPU
    device = local_rank
    torch.cuda.set_device(device)
    # 3. Initialize the process group
    # Explicitly specifying device_id silences the barrier() warning and provides robust support for multi-node communication
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{device}"))
    # 4. Set deterministic seeds
    # Offset the global seed by rank to ensure diverse data augmentation/shuffling across shards
    seed = global_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    # Performance and Precision configurations
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    # High precision for Tensor Cores (TF32) on Ampere or newer GPUs (A100, H100)
    torch.backends.cuda.matmul.fp32_precision = 'tf32' 
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    return rank, num_processes, device


def setup_fsdp2(model, device):
    torch.cuda.set_device(device)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    for module in model.fsdp_wrap_module_list: 
        # 通过显式设置分片层级来控制缓存占用，对每一层（Transformer Block）分别进行 fully_shard。这样 FSDP 每次只会“缓存”当前层和预取的下一层参数，计算完立即释放。
        fully_shard(module, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    return model

def setup_model(model, ema_model, device, grad_ckpt, use_fsdp):
    if grad_ckpt:
        checkpointing_list = model.checkpointing_wrap_module_list
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list,
        )
    if use_fsdp:
        model = setup_fsdp2(model, device)
        ema_model = setup_fsdp2(ema_model, device)
    else:
        model = DDP(model.to(device), device_ids=[device])
        ema_model = ema_model.to(device)
    return model, ema_model


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    # 如果被ddp包裹了，取内部模型
    real_model = model.module if hasattr(model, "module") else model
    real_ema = ema_model.module if hasattr(ema_model, "module") else ema_model
    # 如果被 torch.compile 包裹了，取内部原始模型
    real_model = real_model._orig_mod if hasattr(real_model, "_orig_mod") else real_model
    real_ema = real_ema._orig_mod if hasattr(real_ema, "_orig_mod") else real_ema

    ema_params = OrderedDict(real_ema.named_parameters())
    model_params = OrderedDict(real_model.named_parameters())

    # Normalize keys by removing _checkpoint_wrapped_module
    def normalize_key(k):
        return k.replace("._checkpoint_wrapped_module", "")
    model_params = {normalize_key(k): v for k, v in model_params.items()}
    
    
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def load_checkpoints(resume_from, model, ema_model, optimizer, lr_scheduler, use_fsdp, rank):
    if use_fsdp: # 只有 Rank 0 读取文件，然后通过 FSDP 自动分发（Scatter）到各卡
        options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
        model_full_dict = load_file(os.path.join(resume_from, 'model.safetensors')) if rank == 0 else None
        set_model_state_dict(model, model_state_dict=model_full_dict, options=options)
        ema_full_dict = load_file(os.path.join(resume_from, 'ema.safetensors')) if rank == 0 else None
        set_model_state_dict(ema_model, model_state_dict=ema_full_dict, options=options)
        optim_full_dict = torch.load(os.path.join(resume_from, 'optimizer.bin'), map_location="cpu") if rank == 0 else None
        set_optimizer_state_dict(model, optimizer, optim_state_dict=optim_full_dict, options=options)
    else:
        model.module.load_state_dict(load_file(os.path.join(resume_from, 'model.safetensors')))
        ema_model.load_state_dict(load_file(os.path.join(resume_from, 'ema.safetensors')))
        optimizer.load_state_dict(torch.load(os.path.join(resume_from, 'optimizer.bin'), weights_only=False))
    lr_scheduler.load_state_dict(torch.load(os.path.join(resume_from, 'lr_scheduler.bin'), weights_only=False))


def save_checkpoints(save_dir, model, ema_model, optimizer, lr_scheduler, use_fsdp, rank):
    # 如果被ddp包裹了，取内部模型
    real_model = model.module if hasattr(model, "module") else model
    real_ema = ema_model.module if hasattr(ema_model, "module") else ema_model
    # 如果被 torch.compile 包裹了，取内部原始模型
    real_model = real_model._orig_mod if hasattr(real_model, "_orig_mod") else real_model
    real_ema = real_ema._orig_mod if hasattr(real_ema, "_orig_mod") else real_ema
    if use_fsdp:
        # 注意：以下三个函数必须在所有 Rank 上同时运行
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state = get_model_state_dict(real_model, options=options)
        ema_state = get_model_state_dict(real_ema, options=options)
        optim_state = get_optimizer_state_dict(real_model, optimizer, options=options)
        
        # 只有 Rank 0 负责写入磁盘
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            save_file(model_state, os.path.join(save_dir, "model.safetensors"))
            save_file(ema_state, os.path.join(save_dir, "ema.safetensors"))
            torch.save(optim_state, os.path.join(save_dir, "optimizer.bin"))
    else:
        # DDP 模式下通常只有 Rank 0 有完整 state_dict，但为了保险也可以统一处理
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            save_file(real_model.state_dict(), os.path.join(save_dir, "model.safetensors"))
            save_file(real_ema.state_dict(), os.path.join(save_dir, "ema.safetensors"))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))

    # lr_scheduler 通常每张卡都一样，只需 Rank 0 保存
    if rank == 0:
        torch.save(lr_scheduler.state_dict(), os.path.join(save_dir, "lr_scheduler.bin"))

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    from datetime import datetime
    if dist.get_rank() == 0:  # real logger
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log_{timestamp}.txt", mode='w'),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def format_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"      

