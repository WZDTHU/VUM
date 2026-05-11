import torch
import torch.nn as nn
from typing import List
from vum.models.funcs import get_parameter_dtype
from vum.models.modules.layers import (
    RMSNorm, SwiGLU, TimestepEmbed, LabelEmbed, BottleneckPatchEmbed, Attention, FinalLayer,
)
from vum.models.modules.rope import NativeRotaryEmbed
from vum.models.funcs import modulate


# ===================== Blocks =====================

class ViTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(hidden_size, (mlp_hidden_dim * 2) // 3, drop=proj_drop)

    def forward(self, x, rotary_embed, cu_seqlens=None):
        x = x + self.attn(self.norm1(x), rotary_embed, cu_seqlens)
        x = x + self.mlp(self.norm2(x))
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, adaln_embed_dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(hidden_size, (mlp_hidden_dim * 2) // 3, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_embed_dim, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, rotary_embed):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rotary_embed)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ===================== Decoders =====================

class ViTDecoder(nn.Module):
    """MAE decoder. Accepts pre-assembled full-length input via forward_full."""
    def __init__(self, n_cls_tokens, n_decoder_blocks, hidden_size, decoder_hidden_size,
                 num_decoder_heads, mlp_ratio, attn_drop, proj_drop, out_channels):
        super().__init__()
        self.n_cls_tokens = n_cls_tokens
        self.decoder_embedder = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))
        self.decoder_blocks = nn.ModuleList([
            ViTBlock(decoder_hidden_size, num_decoder_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop, proj_drop=proj_drop)
            for _ in range(n_decoder_blocks)
        ])
        self.decoder_norm = RMSNorm(decoder_hidden_size)
        self.decoder_pred = nn.Linear(decoder_hidden_size, out_channels, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x, rotary_embed, ids_restore):
        x = self.decoder_embedder(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + self.n_cls_tokens - x.shape[1], 1)
        x_ = torch.cat((x[:, self.n_cls_tokens:, :], mask_tokens), dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat((x[:, :self.n_cls_tokens, :], x_), dim=1)
        for blk in self.decoder_blocks:
            x = blk(x, rotary_embed)
        x = x[:, self.n_cls_tokens:, :]
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def forward_varlen(self, cls_out, patch_out, ids_keep, num_keep, masks, rotary_embed):
        """Decode with variable-length kept patches (no external assembly needed)."""
        Bg, max_keep, _ = patch_out.shape
        L = masks.shape[1]
        device = cls_out.device

        combined = torch.cat([cls_out, patch_out], dim=1)
        combined = self.decoder_embedder(combined)
        D_dec = combined.shape[-1]
        cls_dec = combined[:, :self.n_cls_tokens]
        patch_dec = combined[:, self.n_cls_tokens:]

        keep_valid = torch.arange(max_keep, device=device).unsqueeze(0) < num_keep.unsqueeze(1)
        batch_idx = torch.arange(Bg, device=device).unsqueeze(1).expand_as(ids_keep)

        full_patch = self.mask_token.expand(Bg, L, -1).clone().to(patch_dec.dtype)
        full_patch[batch_idx[keep_valid], ids_keep[keep_valid]] = patch_dec[keep_valid]

        x = torch.cat([cls_dec, full_patch], dim=1)
        for blk in self.decoder_blocks:
            x = blk(x, rotary_embed)
        x = x[:, self.n_cls_tokens:]
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x


class JiTDecoder(nn.Module):
    """Gen decoder with adaLN."""
    def __init__(self, n_cls_tokens, n_decoder_blocks, hidden_size, decoder_hidden_size,
                 num_decoder_heads, mlp_ratio, attn_drop, proj_drop, out_channels):
        super().__init__()
        self.n_cls_tokens = n_cls_tokens
        self.decoder_embedder = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))
        self.decoder_blocks = nn.ModuleList([
            JiTBlock(decoder_hidden_size, hidden_size, num_decoder_heads,
                     mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
            for _ in range(n_decoder_blocks)
        ])
        self.decoder_norm = RMSNorm(decoder_hidden_size)
        self.decoder_final = FinalLayer(decoder_hidden_size, out_channels, hidden_size)
        self.decoder_pred = nn.Linear(decoder_hidden_size, out_channels, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        for block in self.decoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder_final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder_final.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, rotary_embed, ids_restore, mask):
        x = self.decoder_embedder(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + self.n_cls_tokens - x.shape[1], 1)
        x_ = torch.cat((x[:, self.n_cls_tokens:, :], mask_tokens), dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat((x[:, :self.n_cls_tokens, :], x_), dim=1)
        for blk in self.decoder_blocks:
            x = blk(x, c, rotary_embed)
        x = x[:, self.n_cls_tokens:]
        x = self.decoder_norm(x)
        mask_expanded = mask.unsqueeze(-1)
        x_unmask = self.decoder_final(x, c)
        x_mask = self.decoder_pred(x)
        x = x_unmask * (1 - mask_expanded) + x_mask * mask_expanded
        return x

    def forward_varlen(self, cls_out, patch_out, ids_keep, num_keep, masks, c, rotary_embed):
        """Decode with variable-length kept patches (no external assembly needed)."""
        Bg, max_keep, _ = patch_out.shape
        L = masks.shape[1]
        device = cls_out.device

        combined = torch.cat([cls_out, patch_out], dim=1)
        combined = self.decoder_embedder(combined)
        D_dec = combined.shape[-1]
        cls_dec = combined[:, :self.n_cls_tokens]
        patch_dec = combined[:, self.n_cls_tokens:]

        keep_valid = torch.arange(max_keep, device=device).unsqueeze(0) < num_keep.unsqueeze(1)
        batch_idx = torch.arange(Bg, device=device).unsqueeze(1).expand_as(ids_keep)

        full_patch = self.mask_token.expand(Bg, L, -1).clone().to(patch_dec.dtype)
        full_patch[batch_idx[keep_valid], ids_keep[keep_valid]] = patch_dec[keep_valid]

        x = torch.cat([cls_dec, full_patch], dim=1)
        for blk in self.decoder_blocks:
            x = blk(x, c, rotary_embed)
        x = x[:, self.n_cls_tokens:]
        x = self.decoder_norm(x)
        mask_expanded = masks.unsqueeze(-1)
        x_unmask = self.decoder_final(x, c)
        x_mask = self.decoder_pred(x)
        x = x_unmask * (1 - mask_expanded) + x_mask * mask_expanded
        return x


# ===================== Model =====================

class VUM(nn.Module):
    def __init__(
        self,
        in_channels=3,
        patch_size=16,
        num_heads=16,
        num_classes=1000,
        hidden_size=1152,
        bottleneck_dim=128,
        n_blocks=12,
        decoder_hidden_size=1152,
        num_decoder_heads=16,
        n_decoder_blocks=12,
        mae_hidden_size=512,
        num_mae_heads=16,
        n_mae_blocks=8,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        n_cls_tokens=1,
        n_reg_tokens=0,
        mask_grid=1,
        # defaults for backward compat (overridden by scheduler)
        mae_ratio=0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.n_cls_tokens = n_cls_tokens
        self.n_reg_tokens = n_reg_tokens
        self.mask_grid = mask_grid
        self.mae_ratio = mae_ratio
        
        if self.mae_ratio > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, n_cls_tokens, hidden_size))
        self.cls_posemb = nn.Parameter(torch.zeros(1, n_cls_tokens, hidden_size))
        if self.n_reg_tokens > 0:
            if self.mae_ratio > 0:
                self.reg_token = nn.Parameter(torch.zeros(1, n_reg_tokens, hidden_size))
            self.reg_posemb = nn.Parameter(torch.zeros(1, n_reg_tokens, hidden_size))

        self.enc_rope = NativeRotaryEmbed(rotary_dim=hidden_size // num_heads)
        self.mae_rope = NativeRotaryEmbed(rotary_dim=mae_hidden_size // num_mae_heads)
        self.dec_rope = NativeRotaryEmbed(rotary_dim=decoder_hidden_size // num_decoder_heads)

        self.t_embedder = TimestepEmbed(hidden_size)
        self.y_embedder = LabelEmbed(num_classes, hidden_size)
        self.x_embedder = BottleneckPatchEmbed(patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
            for _ in range(self.n_blocks)
        ])
        self.norm = RMSNorm(hidden_size)

        if self.mae_ratio > 0.0: 
            self.decoder_mae = ViTDecoder(
                n_cls_tokens, n_mae_blocks, hidden_size, mae_hidden_size, num_mae_heads,
                mlp_ratio, attn_drop, proj_drop, patch_size ** 2 * self.out_channels,
            )
        self.decoder_gen = JiTDecoder(
            n_cls_tokens, n_decoder_blocks, hidden_size, decoder_hidden_size, num_decoder_heads,
            mlp_ratio, attn_drop, proj_drop, patch_size ** 2 * self.out_channels,
        )

        self.initialize_weights()

    # ---- init ----

    def initialize_weights(self):
        if self.mae_ratio > 0:
            torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.trunc_normal_(self.cls_posemb, std=0.02)
        if self.n_reg_tokens > 0:
            if self.mae_ratio > 0:
                torch.nn.init.normal_(self.reg_token, std=.02)
            torch.nn.init.trunc_normal_(self.reg_posemb, std=0.02)
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        self.apply(self._init_weights)
        self.decoder_gen.initialize_weights()
        if self.mae_ratio > 0.0: 
            self.decoder_mae.initialize_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ---- helpers ----

    def get_rope(self, h, w, device):
        grid_h = torch.arange(h)
        grid_w = torch.arange(w)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0).reshape(2, -1)

        grid_enc = torch.cat([torch.zeros([2, self.n_cls_tokens + self.n_reg_tokens]), grid], dim=-1)
        freqs_cos, freqs_sin = self.enc_rope.get_2d_rope((h, w), grid_enc)
        rope_enc = torch.cat([freqs_cos, freqs_sin], dim=-1).to(device)  # (N_full, D_rope)

        grid_dec = torch.cat([torch.zeros([2, self.n_cls_tokens]), grid], dim=-1)
        freqs_cos_mae, freqs_sin_mae = self.mae_rope.get_2d_rope((h, w), grid_dec)
        rope_mae = torch.cat([freqs_cos_mae, freqs_sin_mae], dim=-1).unsqueeze(1).unsqueeze(0).to(device)

        freqs_cos_dec, freqs_sin_dec = self.dec_rope.get_2d_rope((h, w), grid_dec)
        rope_dec = torch.cat([freqs_cos_dec, freqs_sin_dec], dim=-1).unsqueeze(1).unsqueeze(0).to(device)
        return rope_enc, rope_mae, rope_dec

    def unpatchify(self, x, H, W):
        c = self.out_channels
        p = self.patch_size
        h, w = H // p, W // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ---- batch masking (vectorized) ----

    def _batch_mask(self, mask_ratios, grid, h, w, device):
        """
        向量化的batch masking，每个sample可以有不同的mask_ratio，无Python for循环。

        核心思路：masking以 grid×grid 的block为单位（不是单个patch），
        先在block层面做随机shuffle+截断决定哪些block保留，
        再映射到pixel（patch）层面。

        Args:
            mask_ratios: (B,) 每个sample的mask比例，0表示不mask
            grid:        masking粒度，grid×grid个patch组成一个block
            h, w:        patch grid的行列数（非像素）

        Returns:
            masks:           (B, L)               1=masked, 0=kept（空间顺序）
            ids_keep_padded: (B, max_keep_patches)  被保留patch的空间索引（padding到最大值）
            num_keep_patches: (B,)                  每个sample实际保留的patch数
        """
        B = mask_ratios.shape[0]
        L = h * w                           # 总patch数
        g2 = grid * grid                    # 每个block包含的patch数
        num_blocks = (h // grid) * (w // grid)  # 总block数

        # ---- 1. 计算每个sample保留多少个block ----
        # mask_ratio=0 → 全部保留; mask_ratio>0 → 按比例计算并至少保留1个
        num_keep_blocks = torch.where(
            mask_ratios > 0,
            (num_blocks * (1 - mask_ratios)).round().long().clamp(min=1),
            torch.full((B,), num_blocks, device=device, dtype=torch.long),
        )
        num_keep_patches = num_keep_blocks * g2  # 乘以每个block的patch数

        # ---- 2. 对block做随机shuffle ----
        # 给每个block一个随机数，argsort后前num_keep_blocks[b]个就是被保留的block
        noise = torch.rand(B, num_blocks, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)   # (B, num_blocks)

        # ---- 3. 建立 block→pixel 的映射表 ----
        # patch_grid[block_idx] = 该block包含的g2个patch的空间索引
        # 例如 grid=2, h=w=4: block_0 包含 patch [0,1,4,5]（左上2×2）
        patch_grid = torch.arange(L, device=device).view(h, w)
        patch_grid = patch_grid.unfold(0, grid, grid).unfold(1, grid, grid)
        patch_grid = patch_grid.contiguous().view(num_blocks, g2)  # (num_blocks, g2)

        # ---- 4. 从block级别的keep→pixel级别的mask ----
        # Step A: 在shuffled顺序中，位置 < num_keep_blocks 的block被保留
        block_pos = torch.arange(num_blocks, device=device).unsqueeze(0)       # (1, NB)
        block_keep_shuffled = (block_pos < num_keep_blocks.unsqueeze(1))       # (B, NB) True=保留

        # Step B: unshuffle回原始block顺序，这样block_keep_orig[b, orig_block_idx] = 是否保留
        ids_unshuffle = torch.argsort(ids_shuffle, dim=1)
        block_keep_orig = torch.gather(block_keep_shuffled.long(), 1, ids_unshuffle)

        # Step C: 把block级keep扩展到patch级 → (B, num_blocks*g2)，但此时是block遍历顺序
        patch_keep = block_keep_orig.unsqueeze(-1).expand(-1, -1, g2).reshape(B, -1)

        # Step D: scatter到空间顺序。patch_grid.reshape(-1) 就是 block遍历顺序→空间索引 的映射
        pixel_indices = patch_grid.reshape(-1).unsqueeze(0).expand(B, -1)  # (B, L)
        masks = torch.ones(B, L, device=device)
        masks.scatter_(1, pixel_indices, 1.0 - patch_keep.float())
        # 结果: masks[b, spatial_pos] = 1(masked) 或 0(kept)

        # ---- 5. 收集被保留patch的空间索引 ----
        # 在shuffled顺序中，前 max_keep_blocks 个block是被保留的（对所有sample统一取最大值以便padding）
        # max_keep_blocks = num_keep_blocks.max().item()
        max_keep_blocks = num_keep_blocks.max()
        ids_keep_blocks = ids_shuffle[:, :max_keep_blocks]              # (B, max_keep_blocks)
        # 通过 patch_grid 映射到pixel索引
        ids_keep_padded = patch_grid[ids_keep_blocks].reshape(B, -1)    # (B, max_keep_patches)
        # 注意：对于 num_keep_blocks[b] < max_keep_blocks 的sample，
        # 尾部的索引是无效的padding，后续通过 num_keep_patches 区分有效/无效

        return masks, ids_keep_padded, num_keep_patches

    def forward_encoder_packed(self, patches, t, y, mask_ratios, n_mae, h, w):
        """
        Packed encoder — 全向量化，无Python for循环。

        整体流程:
          ① batch mask → 得到每个sample保留了哪些patch
          ② 构建prefix tokens (cls/reg) 和 condition
          ③ 构建RoPE
          ④ 拼成 padded tensor (B, max_seq_len, D) → boolean pack 成 (total_tokens, D)
          ⑤ 跑encoder blocks (flash_attn_varlen)
          ⑥ unpack 回 padded tensor

        流水线:
            patches (B, 256, D)    每个sample有256个patch
                    ↓
            _batch_mask         → masks (B,256), ids_keep (B, max_keep), num_keep (B,)
                    ↓
            gather kept         → x_kept (B, max_keep, D)
                    ↓
            cat [cls, reg, kept] → x_padded (B, n_prefix+max_keep, D)  ← 右侧有padding
                    ↓
            boolean pack        → x_packed (total_tokens, D)            ← 丢弃padding
                    ↓
            encoder blocks      → flash_attn_varlen(cu_seqlens)
                    ↓
            boolean unpack      → x_padded_out (B, max_seq, D)         ← 补回padding(零)
                    ↓
            split cls / patch   → cls_out (B, n_cls, D), patch_out (B, max_keep, D)

        "padded" 的含义:
          不同sample保留的patch数不同，所以用最大值做padding。
          padding部分在pack阶段被boolean mask丢弃，不参与attention计算。

        Returns:
            cls_out:          (B, n_cls, D)
            patch_out:        (B, max_keep_patches, D)     padded
            cond:             (B, 1, D)                   MAE samples为零向量
            masks:            (B, L)                      1=masked, 0=kept
            ids_keep_padded:  (B, max_keep_patches)
            num_keep_patches:  (B,)
        """
        B, L, D = patches.shape
        device = patches.device
        n_prefix = self.n_cls_tokens + self.n_reg_tokens

        # ---- Step 1: batch masking ----
        masks, ids_keep_padded, num_keep_patches = self._batch_mask(
            mask_ratios, self.mask_grid, h, w, device,
        )
        max_keep_patches = ids_keep_padded.shape[1]

        # 用ids_keep_padded从patches中gather出被保留的patch tokens
        x_kept = torch.gather(
            patches, 1, ids_keep_padded.unsqueeze(-1).expand(-1, -1, D),
        )  # (B, max_keep_patches, D)

        # ---- Step 2: 构建prefix tokens ----
        # batch的前 n_mae 个是MAE sample，后面是Gen sample
        # MAE: cls_token = learnable cls + posemb，不依赖t/y
        # Gen:  cls_token = y_emb + posemb，reg_token = t_emb + posemb
        cls_tokens = torch.empty(B, self.n_cls_tokens, D, device=device, dtype=patches.dtype)
        cond = torch.zeros(B, 1, D, device=device, dtype=patches.dtype)

        if n_mae > 0:
            cls_tokens[:n_mae] = (self.cls_token + self.cls_posemb).expand(n_mae, -1, -1)
            t_emb = self.t_embedder(t[n_mae:])
            y_emb = self.y_embedder(y[n_mae:])
            cond[n_mae:] = (t_emb + y_emb).unsqueeze(1)         # decoder用的adaLN condition
            cls_tokens[n_mae:] = y_emb.unsqueeze(1).expand(-1, self.n_cls_tokens, -1) + self.cls_posemb
        else:
            t_emb = self.t_embedder(t)
            y_emb = self.y_embedder(y)
            cond[:] = (t_emb + y_emb).unsqueeze(1)
            cls_tokens = y_emb.unsqueeze(1).expand(-1, self.n_cls_tokens, -1) + self.cls_posemb

        reg_tokens = None
        if self.n_reg_tokens > 0:
            reg_tokens = torch.empty(B, self.n_reg_tokens, D, device=device, dtype=patches.dtype)
            if n_mae > 0:
                reg_tokens[:n_mae] = (self.reg_token + self.reg_posemb).expand(n_mae, -1, -1)
                reg_tokens[n_mae:] = t_emb.unsqueeze(1).expand(-1, self.n_reg_tokens, -1) + self.reg_posemb
            else:
                reg_tokens = t_emb.unsqueeze(1).expand(-1, self.n_reg_tokens, -1) + self.reg_posemb

        # ---- Step 3: 构建RoPE ----
        # rope_enc = [cls位置(zeros), reg位置(zeros), patch位置(2D coords)]
        # 对于被保留的patch，从rope_patches中按ids_keep_padded索引取出对应的RoPE
        rope_enc, _, _ = self.get_rope(h, w, device)
        D_rope = rope_enc.shape[-1]
        rope_prefix = rope_enc[:n_prefix]
        rope_patches = rope_enc[n_prefix:]
        rope_kept = rope_patches[ids_keep_padded]   # (B, max_keep_patches, D_rope)

        # ---- Step 4: padded → packed ----
        #
        # 4a. 拼出 padded tensor: [cls, reg, kept_patches]
        #     shape = (B, n_prefix + max_keep_patches, D)
        #     其中 max_keep_patches 按batch内最大值对齐
        if reg_tokens is not None:
            x_padded = torch.cat([cls_tokens, reg_tokens, x_kept], dim=1)
            rope_padded = torch.cat([rope_prefix.unsqueeze(0).expand(B, -1, -1), rope_kept], dim=1)
        else:
            x_padded = torch.cat([cls_tokens, x_kept], dim=1)
            rope_padded = torch.cat([rope_prefix.unsqueeze(0).expand(B, -1, -1), rope_kept], dim=1)

        max_seq_len = x_padded.shape[1]
        seq_lens = n_prefix + num_keep_patches   # (B,) 每个sample的真实序列长度

        # 4b. boolean mask提取有效token，丢弃padding
        #     valid[b, pos] = True if pos < seq_lens[b]
        #     x_padded[valid] 把(B, S, D)中有效位置按行优先展平成(total_tokens, D)
        pos = torch.arange(max_seq_len, device=device).unsqueeze(0)
        valid = pos < seq_lens.unsqueeze(1)         # (B, max_seq_len)

        x_packed = x_padded[valid]                   # (total_tokens, D)
        rope_packed = rope_padded[valid].unsqueeze(1) # (total_tokens, 1, D_rope) 中间的1是head维度用于broadcast

        # cu_seqlens: flash_attn_varlen 需要的累积序列长度
        # 例如 seq_lens=[100, 80, 120] → cu_seqlens=[0, 100, 180, 300]
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = seq_lens.cumsum(0).to(torch.int32)

        # ---- Step 5: 跑encoder transformer blocks ----
        for blk in self.blocks:
            x_packed = blk(x_packed, rope_packed, cu_seqlens)
        x_packed = self.norm(x_packed)

        # ---- Step 6: unpack ----
        # 把flat的packed结果写回padded tensor，再按位置切出cls和patch
        x_padded_out = torch.zeros(B, max_seq_len, D, device=device, dtype=x_packed.dtype)
        x_padded_out[valid] = x_packed

        cls_out = x_padded_out[:, :self.n_cls_tokens]
        # 跳过reg_tokens，取patch部分
        patch_out = x_padded_out[:, n_prefix:n_prefix + max_keep_patches]

        return cls_out, patch_out, cond, masks, ids_keep_padded, num_keep_patches

    # ---- batch encoder (for inference / forward_features) ----

    def random_masking(self, x, mask_ratio, grid, H, W):
        """Batch masking with uniform mask_ratio (for inference / forward_features)."""
        B, _, C = x.shape
        num_patches = (H // grid) * (W // grid)
        len_keep = int(num_patches * (1 - mask_ratio))
        noise = torch.rand(B, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:]
        g2 = grid * grid
        patch_grid = torch.arange(H * W, device=x.device).view(1, H, W)
        patch_grid = patch_grid.unfold(1, grid, grid).unfold(2, grid, grid)
        patch_grid = patch_grid.contiguous().view(1, -1, grid, grid)
        ids_keep_expanded = patch_grid[:, ids_keep].view(B, -1)
        ids_masked_expanded = patch_grid[:, ids_masked].view(B, -1)
        x_masked = torch.gather(x.view(B, -1, C), 1, ids_keep_expanded.unsqueeze(-1).repeat(1, 1, C))
        ids_restore = torch.argsort(torch.cat((ids_keep_expanded, ids_masked_expanded), dim=1), dim=1)
        mask = torch.ones([B, H * W], device=x.device)
        mask[:, :len_keep * g2] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep_expanded

    def forward_encoder(self, x, t, y, mask_ratio, forward_mae=False):
        """Batch encoder (uniform mask_ratio) for inference / forward_features."""
        B, C, H, W = x.shape
        P = self.patch_size
        h, w = H // P, W // P
        x = x.reshape(B, C, h, P, w, P).permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, P * P * C)
        x = self.x_embedder(x)

        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio, self.mask_grid, h, w)

        rope_enc, rope_mae, rope_dec = self.get_rope(h, w, x.device)
        rope_enc_4d = rope_enc.unsqueeze(1).unsqueeze(0).expand(B, -1, -1, -1)
        n_prefix = self.n_cls_tokens + self.n_reg_tokens
        rotary_embed = torch.gather(
            rope_enc_4d[:, n_prefix:],
            dim=1,
            index=ids_keep.view(B, -1, 1, 1).expand(-1, -1, 1, rope_enc_4d.shape[-1]),
        )
        rotary_embed = torch.cat([rope_enc_4d[:, :n_prefix], rotary_embed], dim=1)

        if forward_mae:
            c = None
            if self.mae_ratio > 0:
                cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_posemb
                reg_tokens = (self.reg_token.expand(B, -1, -1) + self.reg_posemb) if self.n_reg_tokens > 0 else None
            else:
                t = torch.zeros((x.shape[0])).to(x)
                y = torch.ones_like(t).to(torch.int) * 1000
                t_emb = self.t_embedder(t).unsqueeze(1)
                y_emb = self.y_embedder(y).unsqueeze(1)
                c = None
                cls_tokens = y_emb.repeat(1, self.n_cls_tokens, 1) + self.cls_posemb
                reg_tokens = (t_emb.repeat(1, self.n_reg_tokens, 1) + self.reg_posemb) if self.n_reg_tokens > 0 else None
        else:
            t_emb = self.t_embedder(t).unsqueeze(1)
            y_emb = self.y_embedder(y).unsqueeze(1)
            c = t_emb + y_emb
            cls_tokens = y_emb.repeat(1, self.n_cls_tokens, 1) + self.cls_posemb
            reg_tokens = (t_emb.repeat(1, self.n_reg_tokens, 1) + self.reg_posemb) if self.n_reg_tokens > 0 else None

        if reg_tokens is not None:
            x = torch.cat([cls_tokens, reg_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)
        for blk in self.blocks:
            x = blk(x, rotary_embed)
        x = torch.cat([x[:, :self.n_cls_tokens], x[:, self.n_cls_tokens + self.n_reg_tokens:]], dim=1)
        x = self.norm(x)

        if forward_mae:
            return x, c, rope_mae, mask, ids_restore
        return x, c, rope_dec, mask, ids_restore

    # ---- main forward ----

    def forward(self, x, t, y, mask_ratios=None, n_mae=0, **kwargs):
        """
        Training forward with per-sample mask ratios and packed encoder.

        Args:
            x: (B, C, H, W) images (noisy for gen, clean for mae)
            t: (B,) timesteps (first n_mae entries are unused / 0)
            y: (B,) class labels (first n_mae entries are unused)
            mask_ratios: (B,) per-sample mask ratios. If None, use legacy defaults.
            n_mae: number of pure-MAE samples at the start of the batch
        """
        B, C, H, W = x.shape
        P = self.patch_size
        h, w = H // P, W // P
        L = h * w

        if not self.training:   # diffusion sampling
            n_mae = 0
            mask_ratios = torch.zeros(B, device=x.device)


        # patchify + embed
        patches = x.reshape(B, C, h, P, w, P).permute(0, 2, 4, 3, 5, 1).reshape(B, L, P * P * C)
        patches = self.x_embedder(patches)

        # packed encoder
        cls_out, patch_out, cond, masks, ids_keep, num_keep = \
            self.forward_encoder_packed(patches, t, y, mask_ratios, n_mae, h, w)

        _, rope_mae, rope_dec = self.get_rope(h, w, x.device)

        # ---- MAE decoder ----
        if n_mae > 0:
            x_mae = self.decoder_mae.forward_varlen(
                cls_out[:n_mae], patch_out[:n_mae],
                ids_keep[:n_mae], num_keep[:n_mae], masks[:n_mae],
                rope_mae,
            )
            mask_mae = masks[:n_mae]

        # ---- Gen decoder ----
        n_gen = B - n_mae
        if n_gen > 0:
            cond_gen = cond[n_mae:]
            mask_gen = masks[n_mae:]
            x_gen = self.decoder_gen.forward_varlen(
                cls_out[n_mae:], patch_out[n_mae:],
                ids_keep[n_mae:], num_keep[n_mae:], masks[n_mae:],
                cond_gen, rope_dec,
            )

        # ---- combine ----
        if n_mae > 0 and n_gen > 0:
            x_out = torch.cat([x_mae, x_gen], dim=0)
            mask_out = torch.cat([mask_mae, mask_gen], dim=0)
        elif n_mae > 0:
            x_out = x_mae
            mask_out = mask_mae
        else:
            x_out = x_gen
            mask_out = mask_gen

        x_out = self.unpatchify(x_out, H, W)

        if not self.training:
            return x_out
        return x_out, mask_out

    # ---- feature extraction ----

    def forward_features(self, x, return_class_token=False):
        latent, _, _, _, _ = self.forward_encoder(x, None, None, 0.0, forward_mae=True)
        if return_class_token:
            return latent[:, :self.n_cls_tokens].mean(dim=1)
        return latent[:, self.n_cls_tokens:].mean(dim=1)

    @property
    def dtype(self) -> torch.dtype:
        return get_parameter_dtype(self)

    @property
    def fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.blocks)

    @property
    def checkpointing_wrap_module_list(self) -> List[nn.Module]:
        return list(self.blocks)
