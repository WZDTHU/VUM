import sys
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from vum.models.modules.layers import RMSNorm, SwiGLU, Attention, BottleneckPatchEmbed
from vum.models.modules.rope import NativeRotaryEmbed


def _drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class ViTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(hidden_size, (mlp_hidden_dim * 2) // 3, drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rotary_embed, cu_seqlens=None):
        x = x + self.drop_path(self.attn(self.norm1(x), rotary_embed, cu_seqlens))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@BACKBONES.register_module()
class VUM(nn.Module):
    """
    Backbone adapter wrapping VUM (MaskJiT ic6) encoder for mmseg.
    Extracts intermediate features at specified layers and applies FPN.
    """
    def __init__(
        self,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        bottleneck_dim=128,
        n_cls_tokens=8,
        n_reg_tokens=4,
        in_chans=3,
        out_indices=[7, 11, 15, 23],
        frozen_stages=-1,
        drop_path_rate=0.0,
        use_checkpoint=False,
        pretrained=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.n_cls_tokens = n_cls_tokens
        self.n_reg_tokens = n_reg_tokens
        self.n_prefix = n_cls_tokens + n_reg_tokens
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint

        self.x_embedder = BottleneckPatchEmbed(
            patch_size=patch_size, in_chans=in_chans,
            pca_dim=bottleneck_dim, embed_dim=embed_dim, bias=True,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, n_cls_tokens, embed_dim))
        self.cls_posemb = nn.Parameter(torch.zeros(1, n_cls_tokens, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, n_reg_tokens, embed_dim))
        self.reg_posemb = nn.Parameter(torch.zeros(1, n_reg_tokens, embed_dim))

        self.enc_rope = NativeRotaryEmbed(rotary_dim=embed_dim // num_heads)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = RMSNorm(embed_dim, eps=1e-6)

        # FPN for multi-scale (patch_size=16)
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.init_weights(pretrained)

    def get_rope(self, h, w, device):
        grid_h = torch.arange(h)
        grid_w = torch.arange(w)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0).reshape(2, -1)
        grid_enc = torch.cat([torch.zeros([2, self.n_prefix]), grid], dim=-1)
        freqs_cos, freqs_sin = self.enc_rope.get_2d_rope((h, w), grid_enc)
        rope = torch.cat([freqs_cos, freqs_sin], dim=-1).to(device)
        return rope

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return
        logger = get_root_logger()
        logger.info(f'Loading VUM pretrained weights from {pretrained}')

        if pretrained.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(pretrained)
        else:
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

        encoder_keys = {}
        for k, v in state_dict.items():
            if k.startswith('x_embedder.') or k.startswith('blocks.') or k.startswith('norm.'):
                encoder_keys[k] = v
            elif k in ('cls_token', 'cls_posemb', 'reg_token', 'reg_posemb'):
                encoder_keys[k] = v

        msg = self.load_state_dict(encoder_keys, strict=False)
        logger.info(f'Loaded encoder weights. Missing: {msg.missing_keys[:10]}...')
        logger.info(f'Unexpected: {msg.unexpected_keys[:10]}')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.x_embedder.eval()
            for param in self.x_embedder.parameters():
                param.requires_grad = False
            self.cls_token.requires_grad = False
            self.cls_posemb.requires_grad = False
            self.reg_token.requires_grad = False
            self.reg_posemb.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        h, w = H // P, W // P

        # Patchify
        x = x.reshape(B, C, h, P, w, P).permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, P * P * C)
        x = self.x_embedder(x)

        # Prepend cls + reg tokens
        cls_tokens = (self.cls_token + self.cls_posemb).expand(B, -1, -1)
        reg_tokens = (self.reg_token + self.reg_posemb).expand(B, -1, -1)
        x = torch.cat([cls_tokens, reg_tokens, x], dim=1)

        # RoPE
        rope = self.get_rope(h, w, x.device)
        rotary_embed = rope.unsqueeze(1).unsqueeze(0).expand(B, -1, -1, -1)

        # Transformer blocks
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rotary_embed)
            else:
                x = blk(x, rotary_embed)
            if i in self.out_indices:
                xp = self.norm(x[:, self.n_prefix:, :]).permute(0, 2, 1).reshape(B, -1, h, w)
                features.append(xp.contiguous())

        # FPN
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return tuple(features)
