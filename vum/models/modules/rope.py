# --------------------------------------------------------
# FiT: A Flexible Vision Transformer for Image Generation
#
# Based on the following repository
# https://github.com/lucidrains/rotary-embedding-torch
# https://github.com/jquesnelle/yarn/blob/HEAD/scaled_rope
# https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=b80b3f37
# --------------------------------------------------------

import math
import torch
from torch import nn
from einops import repeat

class NativeRotaryEmbed(nn.Module):
    def __init__(
        self,
        rotary_dim: int,
        base: int = 10000,
        initial_size: tuple[int, int] = (32, 32),
        ntk_alpha: float = 2.0,
        ntk_beta: float = 4.0,
        device: torch.device = None,
    ):
        """
        Initializes the 2D Vision Rotary Position Embedding with YaRN scaling.
        This module computes the `cos` and `sin` frequency tensors for RoPE.

        Args:
            rotary_dim (int): The feature dimension of each attention head. Must be divisible by 4.
            base (int): The base value for the sinusoidal frequencies.
            initial_size (tuple[int, int] or tuple[int, int, int]): 
                The (height, width) of the patch grid the model was pre-trained on.
                or The (time, height, width) of the patch grid the model was pre-trained on.
            ntk_alpha (float): The 'alpha' parameter for NTK-by-parts scaling.
            ntk_beta (float): The 'beta' parameter for NTK-by-parts scaling.
            device (torch.device | None): The device to create tensors on.
        """
        super().__init__()
        if rotary_dim % (2*len(initial_size)) != 0:
            raise ValueError(f"rotary_dim must be divisible by 4 for 2D RoPE and 6 for 3D RoPE, but got {rotary_dim}")
        if not ntk_beta > ntk_alpha:
            raise ValueError("YaRN requires ntk_beta > ntk_alpha to create a valid ramp.")

        self.rotary_dim = rotary_dim
        self.dim_for_axis = rotary_dim // len(initial_size)
        self.base = base
        self.initial_size = initial_size
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device
        self._cos_sin_cache = {}

    def _compute_yarn_params_for_dim(
        self, current_len: int, initial_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # (This internal helper function remains the same as before)
        dim_for_axis = self.dim_for_axis
        freq_dim = dim_for_axis // 2
        scaling_factor = current_len / initial_len if current_len > initial_len else 1.0

        if scaling_factor > 1.0:
            conc_val = 0.1 * math.log(scaling_factor) + 1.0
        else:
            conc_val = 1.0
        concentration = torch.tensor(conc_val, device=self.device, dtype=torch.float32)

        freq = self.base ** (
            torch.arange(0, dim_for_axis, 2, device=self.device, dtype=torch.float32)
            / dim_for_axis
        )
        
        if scaling_factor > 1.0:
            log_arg_beta = initial_len / (self.ntk_beta * 2 * math.pi)
            log_arg_alpha = initial_len / (self.ntk_alpha * 2 * math.pi)
            if log_arg_beta <= 0 or log_arg_alpha <= 0:
                 raise ValueError(f"Log arguments for NTK scaling must be positive.")
            low = (freq_dim * math.log(log_arg_beta) / math.log(self.base))
            high = (freq_dim * math.log(log_arg_alpha) / math.log(self.base))
            assert low < high < freq_dim - 1, f"Invalid NTK parameters. Values: low={low:.2f}, high={high:.2f}, freq_dim-1={freq_dim-1}"
            interpolation = 1.0 / ( scaling_factor * freq)
            extrapolation = 1.0 / freq
            ramp = (torch.arange(freq_dim, device=self.device, dtype=torch.float32) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            inv_freq = 1.0 / freq
        return concentration, inv_freq

    def forward(
        self, image_size: tuple[int, int], dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes and returns the cos and sin frequency tensors for the given image size.

        Args:
            image_size (tuple[int, int]): The (height, width) of the patch grid.
            dtype (torch.dtype): The desired data type for the output tensors (e.g., torch.bfloat16).

        Returns:
            A tuple of (cos, sin) tensors, each of shape (H*W, head_dim / 2).
        """
        key = (image_size, dtype)
        if key in self._cos_sin_cache:
            return self._cos_sin_cache[key]
        
        h, w = image_size
        initial_h, initial_w = self.initial_size

        conc_h, inv_freq_h = self._compute_yarn_params_for_dim(h, initial_h)
        conc_w, inv_freq_w = self._compute_yarn_params_for_dim(w, initial_w)

        t_h = torch.arange(h, device=self.device, dtype=torch.float32)
        t_w = torch.arange(w, device=self.device, dtype=torch.float32)

        freqs_h = torch.einsum("i,j->ij", t_h, inv_freq_h)
        freqs_w = torch.einsum("i,j->ij", t_w, inv_freq_w)

        cos_h, sin_h = freqs_h.cos() * conc_h, freqs_h.sin() * conc_h
        cos_w, sin_w = freqs_w.cos() * conc_w, freqs_w.sin() * conc_w
        
        cos_h_exp = cos_h.unsqueeze(1).expand(-1, w, -1)
        sin_h_exp = sin_h.unsqueeze(1).expand(-1, w, -1)
        cos_w_exp = cos_w.unsqueeze(0).expand(h, -1, -1)
        sin_w_exp = sin_w.unsqueeze(0).expand(h, -1, -1)

        cos = torch.cat((cos_h_exp, cos_w_exp), dim=-1).reshape(-1, self.dim_for_axis)
        sin = torch.cat((sin_h_exp, sin_w_exp), dim=-1).reshape(-1, self.dim_for_axis)

        # Cast to the model's dtype at the last moment and cache
        cos = cos.to(dtype)
        sin = sin.to(dtype)
        self._cos_sin_cache[key] = (cos, sin)

        return cos, sin

    def get_2d_rope(self, image_size, grid, dtype: torch.dtype = torch.float32):
        h, w = image_size
        initial_h, initial_w = self.initial_size

        
        conc_h, inv_freq_h = self._compute_yarn_params_for_dim(h, initial_h)
        freqs_h = torch.einsum("i,j->ij", grid[0], inv_freq_h)
        cos_h, sin_h = freqs_h.cos() * conc_h, freqs_h.sin() * conc_h

        conc_w, inv_freq_w = self._compute_yarn_params_for_dim(w, initial_w)
        freqs_w = torch.einsum("i,j->ij", grid[1], inv_freq_w)
        cos_w, sin_w = freqs_w.cos() * conc_w, freqs_w.sin() * conc_w
        
        cos = torch.cat([cos_h, cos_w], dim=-1).to(dtype)
        sin = torch.cat([sin_h, sin_w], dim=-1).to(dtype)

        return cos, sin
    

    def get_3d_rope(self, mm_size, grid, dtype: torch.dtype = torch.float32):
        
        t, h, w = mm_size
        initial_t, initial_h, initial_w = self.initial_size

        conc_t, inv_freq_t = self._compute_yarn_params_for_dim(t, initial_t)
        freqs_t = torch.einsum("i,j->ij", grid[0], inv_freq_t)
        cos_t, sin_t = freqs_t.cos() * conc_t, freqs_t.sin() * conc_t
        
        conc_h, inv_freq_h = self._compute_yarn_params_for_dim(h, initial_h)
        freqs_h = torch.einsum("i,j->ij", grid[1], inv_freq_h)
        cos_h, sin_h = freqs_h.cos() * conc_h, freqs_h.sin() * conc_h
        
        conc_w, inv_freq_w = self._compute_yarn_params_for_dim(w, initial_w)
        freqs_w = torch.einsum("i,j->ij", grid[2], inv_freq_w)
        cos_w, sin_w = freqs_w.cos() * conc_w, freqs_w.sin() * conc_w
        
        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1).to(dtype)
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1).to(dtype)

        return cos, sin
    

def apply_rotary_pos_emb(
    x: torch.Tensor,
    rope_embed: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 2D YaRN Rotary Position Embedding to a query or key tensor.
    This version correctly handles broadcasting for multi-head attention.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, N_heads, D_head).
        rope_embed (torch.Tensor): Combined cos/sin tensor, shape (B, N, D_head).

    Returns:
        torch.Tensor: The tensor with applied rotary embeddings, same shape as input.
    """
    orig_dtype = x.dtype
    cos, sin = rope_embed.chunk(2, dim=-1)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    
    return torch.cat((o1, o2), dim=-1).to(orig_dtype)