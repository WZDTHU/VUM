import torch
import importlib.util

from torch import Tensor
from typing import List, Tuple
from itertools import chain


def is_flash_attn_available():
    return importlib.util.find_spec("flash_attn") is not None


def is_flash_attn_3_available():
    return importlib.util.find_spec("flash_attn_interface") is not None



def ckpt_wrapper(module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    

def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

def modulate(x, shift, scale):
    return x * (1 + scale) + shift
    

def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype



def val2list(x: list or tuple or any, repeat_time=1) -> list:  # type: ignore
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:  # type: ignore
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)





def dynamic_thresholding(x, p=0.995, clip_val=1.0):
    (b, c, h, w), ori_dtype = x.shape, x.dtype
    x_flattened = x.view(b, -1).to(torch.float32)
    
    # 计算每张图的第 p 百分位数（绝对值）
    s = torch.quantile(torch.abs(x_flattened), p, dim=1, keepdim=True)
    # 取 s 和 clip_val 中较大的那个
    s = torch.clamp(s, min=clip_val)
    
    # 归一化：如果 s 很大，x 就会被整体压缩
    x = x / s.view(b, 1, 1, 1)
    return x.to(ori_dtype)


def norm_alignment(x, target_std, target_mean):
    # 计算当前 batch 的均值和方差
    mu = x.mean(dim=(1,2,3), keepdim=True)
    sigma = x.std(dim=(1,2,3), keepdim=True)
    
    # 标准化并重新缩放到目标分布
    # target_std 设为 0.8 而不是 1.0 是为了给极端像素留出余地
    x = (x - mu) / (sigma + 1e-6) * target_std + target_mean
    return x





