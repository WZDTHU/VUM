"""
Flexible MAE + Diffusion scheduler (v2).

Splits each batch into three groups:
  1. Pure MAE  (t=0, mask_ratio > 0)         — first  n_mae   samples
  2. Hybrid    (t>0, mask_ratio > 0)          — middle n_hybrid samples
  3. Pure Diff (t>0, mask_ratio = 0)          — last   n_diff  samples

Mask ratios can be fixed or randomly sampled per group.
The model receives per-sample `mask_ratios` and `n_mae`.
"""
import math
import torch
import torch.nn.functional as F
from copy import deepcopy
from .transport import Transport
from vum.models.funcs import expand_t_like_x, dynamic_thresholding


@torch.no_grad()
def concat_all_gather(tensor):
    if not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


def _sample_ratio(mask_ratio, n, device, style="constant", t=None):
    """
    Sample n mask ratios.

    Args:
        mask_ratio: float (fixed) | [min, max] (uniform range)
        style:      "constant" | "uniform" | "linear" | "cosine"
        t:          (n,) timestep tensor, required for linear / cosine.

    Schedules (lo / hi are derived from mask_ratio):
        constant  → all = mask_ratio  (float) or max (range)
        uniform   → U[lo, hi]
        linear    → m(t) = hi − t·(hi − lo)
        cosine    → m(t) = lo + ½(hi − lo)(1 + cos(πt))
    """
    style = str(style).lower()
    
    if isinstance(mask_ratio, (int, float)):
        lo, hi = float(mask_ratio), float(mask_ratio)
    elif isinstance(mask_ratio, dict):
        lo, hi = float(mask_ratio['min']), float(mask_ratio['max'])
    else:
        lo, hi = float(mask_ratio[0]), float(mask_ratio[1])
        

    if style in ('constant', 'fixed'):
        return torch.full((n,), hi, device=device)
    if style == 'uniform':
        return torch.empty(n, device=device).uniform_(lo, hi)
    if style == 'linear':
        assert t is not None, "linear mask schedule requires timestep t"
        return hi - t * (hi - lo)
    if style == 'cosine':
        assert t is not None, "cosine mask schedule requires timestep t"
        return lo + 0.5 * (hi - lo) * (1.0 + torch.cos(torch.pi * t))
    if style == 'sqrt':
        assert t is not None, "sqrt mask schedule requires timestep t"
        return hi - torch.sqrt(t) * (hi - lo)
    if style == 'square':
        assert t is not None, "square mask schedule requires timestep t"
        return hi - t ** 2 * (hi - lo)
    raise ValueError(f"Unknown mask_ratio style: {style!r}")


class HybridSchedule:
    def __init__(
        self,
        transport: Transport,
        patch_size: int = 16,
        base_resolution: int = 256,
        # noise schedule
        T_max: float = 1.0,
        T_min: float = 0.004,
        # MAE: add small noise x + t*randn, t in [0, mae_noise_max], biased toward 0
        mae_noise_max: float = 0.0,  # 0 = no noise; e.g. 0.1 for light noise
        # Hybrid: separate P-distribution (mask+noise, typically lower noise than pure diff)
        P_mean_hybrid: float = None,  # None = use transport.P_mean
        P_std_hybrid: float = None,  # None = use transport.P_std
        # batch split ratios (the rest is hybrid)
        mae_ratio: float = 0.25,
        diff_ratio: float = 0.25,
        # mask ratio configs: float or [min, max]
        mask_ratio_mae=0.75,
        mask_ratio_hybrid=0.25,
        hybrid_mask_style: str = "constant",  # "constant" | "uniform" | "linear" | "cosine"
        # loss coefficients
        mae1_coeff: float = 1.0,
        mae2_coeff: float = 0.1,
        # targets
        norm_pix_loss: bool = True,
        target_noisy_pix: bool = True,
    ):
        self.transport = transport
        self.patch_size = patch_size
        self.base_resolution = base_resolution
        self.scale_factor = ((patch_size ** 2) / 256) ** 0.5
        self.T_max = T_max
        self.T_min = T_min
        self.mae_noise_max = mae_noise_max
        self.P_mean_hybrid = P_mean_hybrid if P_mean_hybrid is not None else transport.P_mean
        self.P_std_hybrid = P_std_hybrid if P_std_hybrid is not None else transport.P_std

        self.mae_ratio = mae_ratio
        self.diff_ratio = diff_ratio
        self.mask_ratio_mae = mask_ratio_mae
        self.mask_ratio_hybrid = mask_ratio_hybrid
        self.hybrid_mask_style = str(hybrid_mask_style).lower()

        self.mae1_coeff = mae1_coeff
        self.mae2_coeff = mae2_coeff
        self.norm_pix_loss = norm_pix_loss
        self.target_noisy_pix = target_noisy_pix
        

    # ---- prepare input ----

    def _sample_t_from_P(self, n, dtype, device, P_mean, P_std):
        """Sample t via sigmoid(N(P_mean, P_std))."""
        rnd = torch.randn(n, dtype=dtype, device=device)
        t = torch.sigmoid(rnd * P_std + P_mean).clamp_min(self.T_min)
        return t

    def prepare_input(self, x):
        """
        Returns: x_t, t_flat, mask_ratios, n_mae
            x_t: (B, C, H, W) noisy images
            t_flat: (B,) timesteps (0 for MAE samples)
            mask_ratios: (B,) per-sample mask ratios
            n_mae: int
        """
        (B, C, H, W), dtype, device = x.shape, x.dtype, x.device

        n_mae = round(self.mae_ratio * B)
        n_diff = round(self.diff_ratio * B)
        n_hybrid = B - n_mae - n_diff

        # sample timesteps: hybrid and diff use separate P-distributions
        t = torch.zeros(B, dtype=dtype, device=device)
        snr_scale = (H * W / (self.base_resolution ** 2)) ** 0.5
        if n_hybrid > 0:
            t_h = self._sample_t_from_P(n_hybrid, dtype, device, self.P_mean_hybrid, self.P_std_hybrid)
            t_h = snr_scale * t_h / (1 + (snr_scale - 1) * t_h)
            t[n_mae:n_mae + n_hybrid] = t_h
        if n_diff > 0:
            t_d = self.transport.sample_t(batch_size=n_diff, dtype=dtype, device=device).clamp_min(self.T_min)
            t_d = snr_scale * t_d / (1 + (snr_scale - 1) * t_d)
            t[n_mae + n_hybrid:] = t_d

        # sample mask ratios
        mask_ratios = torch.zeros(B, device=device)
        if n_mae > 0:
            mask_ratios[:n_mae] = _sample_ratio(self.mask_ratio_mae, n_mae, device)
        if n_hybrid > 0:
            mask_ratios[n_mae:n_mae + n_hybrid] = _sample_ratio(
                self.mask_ratio_hybrid, n_hybrid, device,
                style=self.hybrid_mask_style, t=t_h)
        

        # build x_t: MAE (optional light noise), hybrid+diff (interpolant)
        t_exp = expand_t_like_x(t, x)
        z = self.scale_factor * torch.randn_like(x)
        x_t = self.transport.interpolant(x, z, t_exp)
        if n_mae > 0 and self.mae_noise_max > 0:
            # MAE: x + t_small*randn, t_small in [0, mae_noise_max], biased toward 0
            t_small = self.mae_noise_max * (torch.rand(n_mae, device=device, dtype=dtype) ** 2)
            t_small = t_small.view(n_mae, 1, 1, 1)
            # x_t[:n_mae] = x[:n_mae] + t_small * torch.randn_like(x[:n_mae])
            x_t[:n_mae] = (1-t_small) * x[:n_mae] + t_small * torch.randn_like(x[:n_mae])

        return x_t, t_exp, mask_ratios, n_mae, n_diff

    # ---- helpers ----

    def _patchify(self, imgs):
        p = self.patch_size
        N, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(N, C, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(N, h * w, p * p * C)


   

    def adjust_mae_weight(self, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        max_w = 1.5
        min_w = 0.5
        weight = min_w + (max_w - min_w) * 0.5 * (1. + math.cos(math.pi * epoch / 800))
        return weight
    
    # ---- main loss ----

    def __call__(self, model, ema_model, x, h_target, model_kwargs, null_kwargs={}, epochs=0):

        x_t, t, mask_ratios, n_mae, n_diff = self.prepare_input(x)

        x_pred, mask = model(x_t, self.transport.c_noise(t).flatten(), model_kwargs['y'], mask_ratios=mask_ratios, n_mae=n_mae)


        x_pred = self._patchify(x_pred)
        x_patch = self._patchify(x)
        x_t_patch = self._patchify(x_t)

        def normalize_target(target):
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            return (target - mean) / (var + 1.e-6) ** 0.5

        # ---- Part 1: Pure MAE samples (first n_mae) ----
        if n_mae > 0:
            mask_mae = mask[:n_mae]
            mask_sum = mask_mae.sum(dim=-1).clamp(min=1)
            pred_mae = x_pred[:n_mae]
            target_mae = normalize_target(x_patch[:n_mae])
            mae_per_patch = ((pred_mae - target_mae) ** 2).mean(dim=-1)
            mae_loss_1 = (mae_per_patch * mask_mae).sum(dim=-1) / mask_sum
            mae_loss_1 = torch.nan_to_num(mae_loss_1, nan=0.0, posinf=1e6, neginf=-1e6).mean()
        else:
            mae_loss_1 = 0.0

        # ---- Part 2: Gen samples (hybrid + diff, from n_mae onward) ----
        n_gen = x_patch.shape[0] - n_mae

        if n_gen > 0:
            pred_gen = x_pred[n_mae:]
            target_gen = x_t_patch[n_mae:] if self.target_noisy_pix else x_patch[n_mae:]
            target_gen = normalize_target(target_gen) if self.norm_pix_loss else target_gen
            mask_gen = mask[n_mae:]
            unmask_gen = 1.0 - mask_gen

            mae_per_patch_2 = ((pred_gen - target_gen) ** 2).mean(dim=-1)
            mask_sum_2 = mask_gen.sum(dim=-1).clamp(min=1)
            mae_loss_2 = (mae_per_patch_2 * mask_gen).sum(dim=-1) / mask_sum_2
            mae_loss_2 = torch.nan_to_num(mae_loss_2, nan=0.0, posinf=1e6, neginf=-1e6).mean()

            v_pred = self.transport.c_out(x_pred[n_mae:], x_t_patch[n_mae:], t[n_mae:].squeeze(-1))
            v_target = self.transport.c_out(x_patch[n_mae:], x_t_patch[n_mae:], t[n_mae:].squeeze(-1))
            diff_per_patch = ((v_pred - v_target) ** 2).mean(dim=-1)
            unmask_sum = unmask_gen.sum(dim=-1).clamp(min=1)
            diffusion_loss = (diff_per_patch * unmask_gen).sum(dim=-1) / unmask_sum
            diffusion_loss = torch.nan_to_num(diffusion_loss, nan=0.0, posinf=1e6, neginf=-1e6).mean()
        else:
            mae_loss_2 = 0.0
            diffusion_loss = 0.0

       
        
        loss = diffusion_loss + self.mae1_coeff * mae_loss_1 + self.mae2_coeff * mae_loss_2
        loss_dict = dict(
            loss=loss.detach().item() if torch.is_tensor(loss) else loss,
            m1_loss=mae_loss_1.detach().item() if torch.is_tensor(mae_loss_1) else mae_loss_1,
            m2_loss=mae_loss_2.detach().item() if torch.is_tensor(mae_loss_2) else mae_loss_2,
            d_loss=diffusion_loss.detach().item() if torch.is_tensor(diffusion_loss) else diffusion_loss,
        )
        

        return loss, loss_dict

    # ---- sampling (same as v1, no masking) ----

    def forward_with_cfg(self, model, x_t, t, y, y_null, cfg_scale, cfg_low, cfg_high):
        apply_cfg = cfg_scale > 1.0 and t >= cfg_low and t <= cfg_high
        if apply_cfg:
            x_cur = torch.cat([x_t] * 2, dim=0)
            y_cur = torch.cat([y, y_null], dim=0)
        else:
            x_cur = x_t
            y_cur = y
        t_cur = torch.ones(x_cur.size(0)).to(x_cur) * self.transport.c_noise(t)
        x_pred = model(x_cur, t_cur, y_cur)
        if apply_cfg:
            x_cond, x_uncond = x_pred.chunk(2)
            x_pred = x_uncond + cfg_scale * (x_cond - x_uncond)
        x_pred = dynamic_thresholding(x_pred, p=0.99).clamp(-1., 1.)
        v_pred = self.transport.c_out(x_pred, x_t, self.transport.c_noise(t))
        return v_pred

    @torch.no_grad()
    def sample(self, model, y, y_null, z, T_max=1.0, T_min=0.0, num_steps=4,
               cfg_scale=1.0, cfg_low=0.0, cfg_high=1.0, stochasticity_ratio=0.0):
        z = self.scale_factor * z
        t_steps = torch.linspace(T_max, T_min, num_steps + 1, dtype=torch.float64).to(z)
        cfg_low = cfg_low * T_max
        cfg_high = cfg_high * T_max
        x_cur = deepcopy(z)
        samples = [z]
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            F_pred = self.forward_with_cfg(model, x_cur, t_cur, y, y_null, cfg_scale, cfg_low, cfg_high)
            s_ratio = stochasticity_ratio if (stochasticity_ratio > 0.0 and T_min < t_cur < T_max) else 0.0
            x_next = self.transport.from_x_t_to_x_r(x_cur, t_cur, t_next, F_pred, s_ratio)
            samples.append(x_next)
            x_cur = x_next
        return torch.stack(samples, dim=0).to(torch.float32)

