from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from vum.models.modules.rope import apply_rotary_pos_emb
from vum.models.funcs import modulate, is_flash_attn_available, is_flash_attn_3_available
if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
if is_flash_attn_3_available():
    from flash_attn_interface import flash_attn_func, flash_attn_varlen_func


# Qwen3Next zero-centerd RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3Next is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class SwiGLU(nn.Module):
    """ SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using Linear layer instead of Conv2d.
    Equivalent to PatchEmbed when kernel_size == stride.
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        # Input dim: patch_size * patch_size * in_chans
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class BottleneckPatchEmbed(nn.Module):
    """
    Image to Patch Embedding with bottleneck using Linear layers instead of Conv2d.
    Equivalent to BottleneckPatchEmbed when kernel_size == stride.
    """
    def __init__(self, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        # proj1: patch_size * patch_size * in_chans -> pca_dim (no bias)
        # proj2: pca_dim -> embed_dim (with bias)
        self.proj1 = nn.Linear(patch_size * patch_size * in_chans, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=bias)

    def forward(self, x):
        x = self.proj2(self.proj1(x))
        return x


class TimestepEmbed(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb




class LabelEmbed(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings




class ContextEmbed(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, mllm_feat_dim, hidden_size):
        super().__init__()
        self.norm = RMSNorm(mllm_feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mllm_feat_dim, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        

    def forward(self, cap_feats):
        '''
        cfg is also essential in text-to-image generation
        '''
        cap_feats = self.mlp(self.norm(cap_feats))
        return cap_feats




class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = True, 
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            headwise_attn_output_gate: bool = False,
            elementwise_attn_output_gate: bool = False,
            **kwargs
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate

        if headwise_attn_output_gate:
            self.qkvg = nn.Linear(dim, dim * 3 + self.num_heads, bias=qkv_bias)
        elif elementwise_attn_output_gate:
            self.qkvg = nn.Linear(dim, dim * 4, bias=qkv_bias)
        else:
            self.qkvg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

        self.proj = nn.Linear(dim, dim)
        

    def forward(self, x: torch.Tensor, rotary_embed: torch.Tensor, cu_seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cu_seqlens == None:
            return self.forward_batch(x, rotary_embed)
        else:
            return self.forward_pack(x, rotary_embed, cu_seqlens)

    def forward_batch(self, x: torch.Tensor, rotary_embed: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkvg = self.qkvg(x)
        ori_dtype = qkvg.dtype
        if self.headwise_attn_output_gate:
            qkv, g = qkvg.split([3 * self.head_dim * self.num_heads, self.num_heads], dim=-1)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4).contiguous()
            g = g.reshape(B, N, self.num_heads, 1)
        elif self.elementwise_attn_output_gate:
            qkvg = qkvg.reshape(B, N, 4, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4).contiguous()
            qkv, g = qkvg.split([3, 1], dim=0)
        else:
            qkv = qkvg.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4).contiguous()
        # q, k, v: (B, N, n_head, d_head)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        rotary_ndims = rotary_embed.shape[-1]
        if rotary_ndims != self.head_dim:
            q_rot, q_pass = q[..., : rotary_ndims], q[..., rotary_ndims :]
            k_rot, k_pass = k[..., : rotary_ndims], k[..., rotary_ndims :]
            q_rot = apply_rotary_pos_emb(q_rot, rotary_embed)
            k_rot = apply_rotary_pos_emb(k_rot, rotary_embed)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
        else:
            q = apply_rotary_pos_emb(q, rotary_embed)
            k = apply_rotary_pos_emb(k, rotary_embed)
            
        x = flash_attn_func(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            # dropout_p=self.attn_drop if self.training else 0.,
        ).to(ori_dtype)
        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            x = x * torch.sigmoid(g)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = torch.dropout(x, self.proj_drop, self.training)
        return x
    
    def forward_pack(self, x: torch.Tensor, rotary_embed: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        
        qkvg = self.qkvg(x)
        ori_dtype = qkvg.dtype
        if self.headwise_attn_output_gate:
            qkv, g = qkvg.split([3 * self.head_dim * self.num_heads, self.num_heads], dim=-1)
            qkv = qkv.reshape(L, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).contiguous()
            g = g.reshape(L, self.num_heads, 1)
        elif self.elementwise_attn_output_gate:
            qkvg = qkvg.reshape(L, 4, self.num_heads, self.head_dim).permute(1, 0, 2, 3).contiguous()
            qkv, g = qkvg.split([3, 1], dim=0)
        else:
            qkv = qkvg.reshape(L, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).contiguous()
        # q, k, v: (L, n_head, d_head)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        rotary_ndims = rotary_embed.shape[-1]
        if rotary_ndims != self.head_dim:
            q_rot, q_pass = q[..., : rotary_ndims], q[..., rotary_ndims :]
            k_rot, k_pass = k[..., : rotary_ndims], k[..., rotary_ndims :]
            q_rot = apply_rotary_pos_emb(q_rot, rotary_embed)
            k_rot = apply_rotary_pos_emb(k_rot, rotary_embed)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
        else:
            q = apply_rotary_pos_emb(q, rotary_embed)
            k = apply_rotary_pos_emb(k, rotary_embed)
        
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        x = flash_attn_varlen_func(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        ).to(ori_dtype)
        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            x = x * torch.sigmoid(g)

        x = x.reshape(L, C)
        x = self.proj(x)
        x = torch.dropout(x, self.proj_drop, self.training)
        return x




class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop


        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        

    def forward(self, x: torch.Tensor, y: torch.Tensor, rotary_embed: torch.Tensor, cu_seqlens: Optional[Tuple[torch.Tensor]] = None) -> torch.Tensor:
        if cu_seqlens == None:
            return self.forward_batch(x, y, rotary_embed)
        else:
            return self.forward_pack(x, y, rotary_embed, cu_seqlens)

    def forward_batch(self, x: torch.Tensor, y: torch.Tensor, rotary_embed: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = y.shape
        # q: (B, N, n_head, d_head), k, v: (B, M, n_head, d_head)
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        kv = self.kv(y).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)

        ori_dtype = q.dtype
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_pos_emb(q, rotary_embed)
        q, k = q.to(ori_dtype), k.to(ori_dtype)

        x = flash_attn_func(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            # dropout_p=self.attn_drop if self.training else 0.,
        ).to(ori_dtype)
        x = x.reshape(B, N, C)

        x = self.proj(x)
        x = torch.dropout(x, self.proj_drop, self.training)
        return x

    def forward_pack(self, x: torch.Tensor, y: torch.Tensor, rotary_embed: torch.Tensor, cu_seqlens: Tuple[torch.Tensor]) -> torch.Tensor:
        N, C = x.shape
        M, _ = y.shape
        # q: (N, n_head, d_head), k, v: (M, n_head, d_head)
        q = self.q(x).reshape(N, self.num_heads, self.head_dim)
        kv = self.kv(y).reshape(M, 2, self.num_heads, self.head_dim).permute(1, 0, 2, 3)

        ori_dtype = q.dtype
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_pos_emb(q, rotary_embed)
        q, k = q.to(ori_dtype), k.to(ori_dtype)

        x_cu_seqlens, y_cu_seqlens = cu_seqlens
        x_max_seqlen = (x_cu_seqlens[1:] - x_cu_seqlens[:-1]).max().item()
        y_max_seqlen = (y_cu_seqlens[1:] - y_cu_seqlens[:-1]).max().item()


        x = flash_attn_varlen_func(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            x_cu_seqlens, y_cu_seqlens, x_max_seqlen, y_max_seqlen
        ).to(ori_dtype)
        
        x = x.reshape(N, C)
        x = self.proj(x)
        x = torch.dropout(x, self.proj_drop, self.training)
        return x







class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, out_channels, adaln_embed_dim=None):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        if adaln_embed_dim == None:
            adaln_embed_dim = hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_embed_dim, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



   

class ModulatedConv2d(nn.Module):
    """StyleGAN2-style Modulated Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        
        # 基础卷积权重: [out, in, k, k]
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='linear')
        
        # 风格映射层：将 style 映射为 per-channel 缩放因子
        self.style_linear = nn.Linear(style_dim, in_channels)
        # 初始化为 identity mapping (输出 1)
        nn.init.ones_(self.style_linear.weight.data[:in_channels, :in_channels] if style_dim >= in_channels else self.style_linear.weight.data)
        nn.init.zeros_(self.style_linear.bias)
        
    def forward(self, x, style):
        """
        x: (B, in_channels, H, W)
        style: (B, style_dim) - global style vector
        """
        B, C, H, W = x.shape
        
        # 1. Style modulation: s_i in StyleGAN2
        # s: [B, 1, in_channels, 1, 1]
        s = self.style_linear(style).view(B, 1, self.in_channels, 1, 1)
        
        # 2. Modulation: w' = s * w
        # w: [B, out, in, k, k]
        w = self.weight.unsqueeze(0) * s
        
        # 3. Demodulation: 归一化以稳定训练
        if self.demodulate:
            sigma = torch.rsqrt(w.pow(2).sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
            w = w * sigma
            
        # 4. 分组卷积实现 batch-wise 不同权重
        x = x.view(1, B * C, H, W)
        w = w.view(B * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        out = F.conv2d(x, w, groups=B, padding=self.kernel_size // 2)
        return out.view(B, self.out_channels, H, W)


class RefinerUNet(nn.Module):
    """
    U-Net with hybrid conditioning and configurable upsampling strategy.
    
    Architecture:
    - Encoder: 4x stride=2 downsampling (H -> H/16)
    - Bottleneck: spatial concat (x_spatial) + style modulation (x_semantic) via ModulatedConv2d
    - Decoder: configurable upsampling + skip connections
    - Output: (B, 3, H, W) -> patchify -> project to (B, L, out_dim)
    
    Args:
        style_dim: dimension of transformer features for style modulation
        out_dim: output projection dimension (should match decoder out_channels)
        patch_size: patch size for patchify operation
        upsample_mode: upsampling strategy for the decoder path
            - 'conv_transpose': ConvTranspose2d (fast, may have checkerboard artifacts)
            - 'interpolate': F.interpolate + Conv2d (flexible sizes, smooth)
            - 'pixel_shuffle': Conv2d + PixelShuffle (learnable sub-pixel, no checkerboard)
    """
    UPSAMPLE_MODES = ('conv_transpose', 'interpolate', 'pixel_shuffle')
    
    def __init__(self, style_dim=768, out_dim=768, patch_size=16, upsample_mode='conv_transpose'):
        super().__init__()
        assert upsample_mode in self.UPSAMPLE_MODES, \
            f"upsample_mode must be one of {self.UPSAMPLE_MODES}, got '{upsample_mode}'"
        self.style_dim = style_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.upsample_mode = upsample_mode
        
        # --- Encoder Path (shared across all modes) ---
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.SiLU())
        self.down1 = nn.Conv2d(64, 64, 2, stride=2)      # H -> H/2
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.SiLU())
        self.down2 = nn.Conv2d(128, 128, 2, stride=2)    # H/2 -> H/4
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.SiLU())
        self.down3 = nn.Conv2d(256, 256, 2, stride=2)    # H/4 -> H/8
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.SiLU())
        self.down4 = nn.Conv2d(512, 512, 2, stride=2)    # H/8 -> H/16

        # --- Bottleneck ---
        self.bottleneck = ModulatedConv2d(512 + style_dim, 512, 3, style_dim)

        # --- Decoder Path (up blocks differ per mode, dec blocks are shared) ---
        self.up4, self.up3, self.up2, self.up1 = self._build_up_blocks(upsample_mode)
        self.dec4 = nn.Sequential(nn.Conv2d(512 + 512, 256, 3, padding=1), nn.SiLU())
        self.dec3 = nn.Sequential(nn.Conv2d(256 + 256, 128, 3, padding=1), nn.SiLU())
        self.dec2 = nn.Sequential(nn.Conv2d(128 + 128, 64, 3, padding=1), nn.SiLU())
        self.dec1 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.SiLU())

        # --- Output ---
        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)
        self.out_proj = nn.Linear(patch_size * patch_size * 3, out_dim)
        
        self._init_weights()

    @staticmethod
    def _build_up_blocks(mode):
        """Build upsampling blocks based on the selected mode."""
        channels = [512, 256, 128, 64]
        if mode == 'conv_transpose':
            return nn.ModuleList([
                nn.ConvTranspose2d(c, c, 2, stride=2) for c in channels
            ])
        elif mode == 'interpolate':
            return nn.ModuleList([
                nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU()) for c in channels
            ])
        elif mode == 'pixel_shuffle':
            return nn.ModuleList([
                nn.Sequential(nn.Conv2d(c, c * 4, 3, padding=1), nn.PixelShuffle(2), nn.SiLU())
                for c in channels
            ])
    
    def _init_weights(self):
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _patchify(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // P) * (W // P), P * P * C)
        return x

    def _upsample_and_fuse(self, x, skip, up_block):
        """Upsample x to match skip's spatial size, then concatenate."""
        if self.upsample_mode == 'interpolate':
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = up_block(x)
        return torch.cat([x, skip], dim=1)

    def forward(self, ori_x, x_semantic, x_spatial):
        """
        ori_x: (B, 3, H, W) - original image
        x_semantic: (B, L, D) - semantic features for style modulation
        x_spatial: (B, L, D) - spatial features for spatial concat
        returns: (B, L, out_dim)
        """
        B, L, D = x_spatial.shape
        H_latent = W_latent = int(L ** 0.5)
        
        style_semantic = x_semantic.mean(dim=1)
        x_spatial_2d = x_spatial.transpose(1, 2).reshape(B, D, H_latent, W_latent)
        
        # --- Encoder ---
        e1 = self.enc1(ori_x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        latent = self.down4(e4)
        
        # --- Bottleneck ---
        b = F.silu(self.bottleneck(torch.cat([latent, x_spatial_2d], dim=1), style_semantic))
        
        # --- Decoder ---
        d4 = self.dec4(self._upsample_and_fuse(b, e4, self.up4))
        d3 = self.dec3(self._upsample_and_fuse(d4, e3, self.up3))
        d2 = self.dec2(self._upsample_and_fuse(d3, e2, self.up2))
        d1 = self.dec1(self._upsample_and_fuse(d2, e1, self.up1))
        
        # --- Output ---
        out = self.out_proj(self._patchify(self.final_conv(d1)))
        return out





# projector
class ProjectionLayer(nn.Module):
    def __init__(self, projection_layer_type, hidden_size, projector_dim, z_dim):
        super().__init__()
        assert projection_layer_type in ['mlp', 'conv']
        self.projection_layer_type = projection_layer_type 
        self.build_projection_layer(projection_layer_type, hidden_size, projector_dim, z_dim)

    def build_projection_layer(self, projection_layer_type, hidden_size, projector_dim, z_dim):
        if projection_layer_type == "mlp":
            self.projection_layer = nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )
        elif projection_layer_type == "conv":
            in_ch  = hidden_size
            out_ch = z_dim
            self.projection_layer = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"Unsupported projection layer type: {projection_layer_type}")

    def forward(self, x, hw: tuple[int, int] | None = None):
        """
        x: [B, T, D]
        hw: optional (H, W) for non-square token grids (mostly not used).
        """
        B, T, D = x.shape
        if self.projection_layer_type == "mlp":
            x_ = self.projection_layer(x.reshape(B * T, D))
            return x_.reshape(B, T, -1)

        elif self.projection_layer_type == "conv":
            if hw is None:
                H = W = int(math.isqrt(T))
                assert H * W == T, f"conv projector needs square grid or pass hw; got T={T}"
            else:
                H, W = hw
                assert H * W == T, f"Provided hw={hw} but T={T}"

            # [B, T, D] -> [B, D, H, W]
            x_ = x.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
            y  = self.projection_layer(x_)                  # [B, z_dim, H, W]
            y  = y.permute(0, 2, 3, 1).contiguous()         # [B, H, W, z_dim]
            return y.reshape(B, T, -1)


class SSLHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, num_layers=3, use_bn=False, weight_norm=False):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            # 中间层使用 hidden_dim
            target_out = out_dim if is_last else hidden_dim
            
            linear = nn.Linear(curr_dim, target_out)
            # 最后一层常用 Weight Norm 增强稳定性
            if is_last and weight_norm:
                linear.weight.data = linear.weight.data / linear.weight.data.norm(dim=0, keepdim=True)
                
            layers.append(linear)
            
            if not is_last:
                if use_bn: layers.append(nn.BatchNorm1d(target_out, affine=False))
                else: layers.append(nn.LayerNorm(target_out, elementwise_affine=False))
                layers.append(nn.GELU())
            curr_dim = target_out
            
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class MocoHead(nn.Module):
    def __init__(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        super().__init__()
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        self.mlp = nn.Sequential(*mlp)
    
    def forward(self, x):
        return self.mlp(x)