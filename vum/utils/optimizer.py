import math
import torch
from vum.utils.optim_factory import param_groups_wd


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X



class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        muon_param_groups=None,  # list of {'params': [...], 'wd': ...}
        adamw_param_groups=None,  # list of {'params': [...], 'wd': ...}
    ):
        param_groups = []
        
        # Handle muon params: either as param_groups (with per-group wd) or as a simple list
        if muon_param_groups is not None:
            for pg in muon_param_groups:
                if len(pg['params']) > 0:
                    param_groups.append({
                        "params": list(pg['params']),
                        "use_muon": True,
                        "lr": lr,
                        "wd": pg.get('wd', wd),
                        "momentum": momentum,
                        "nesterov": nesterov,
                        "ns_steps": ns_steps,
                    })
        elif muon_params is not None:
            param_groups.append({
                "params": list(muon_params),
                "use_muon": True,
                "lr": lr,
                "wd": wd,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
            })
        
        # Handle adamw params: either as param_groups (with per-group wd) or as a simple list
        if adamw_param_groups is not None:
            for pg in adamw_param_groups:
                if len(pg['params']) > 0:
                    param_groups.append({
                        "params": list(pg['params']),
                        "use_muon": False,
                        "lr": lr,
                        "wd": pg.get('wd', wd),
                        "adamw_betas": adamw_betas,
                        "adamw_eps": adamw_eps,
                    })
        elif adamw_params is not None:
            param_groups.append({
                "params": list(adamw_params),
                "use_muon": False,
                "lr": lr,
                "wd": wd,
                "adamw_betas": adamw_betas,
                "adamw_eps": adamw_eps,
            })

        super().__init__(param_groups, dict(lr=lr))
        
        # Verify muon params are 2D
        all_muon_params = []
        if muon_param_groups is not None:
            for pg in muon_param_groups:
                all_muon_params.extend(pg['params'])
        elif muon_params is not None:
            all_muon_params = list(muon_params)
        
        for p in all_muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            
            if group["use_muon"]:
                ############################
                #           Muon           #
                ############################
                momentum = group["momentum"]
                # generate weight updates
                for p in group["params"]:
                    # sanity check
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None

                    # calc update
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                    # scale update
                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                    # apply weight decay
                    p.data.mul_(1 - lr * wd)

                    # apply update
                    p.data.add_(u, alpha=-adjusted_lr)
            else:
                ############################
                #       AdamW backup       #
                ############################
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)
                    state["step"] += 1
                    step = state["step"]
                    buf1 = state["moment1"]
                    buf2 = state["moment2"]
                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)

                    g = buf1 / (eps + buf2.sqrt())

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    scale = bias_correction1 / bias_correction2**0.5
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(g, alpha=-lr / scale)

        return loss



def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "adamw_wd":
        param_groups = param_groups_wd(model, wd) # wd=0.05 in mae
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    elif optimizer_name == "muon":
        muon_params = [
            p for name, p in model.named_parameters() if is_muon_param(name, p)
        ]
        adamw_params = [
            p for name, p in model.named_parameters() if not is_muon_param(name, p)
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    elif optimizer_name == "muon_wd":
        # Separate muon_params into decay/no_decay based on param name
        # muon_params: 2D params (excluding cls_token, posemb, final_layer)
        # For weight decay: apply to 2D params that are not bias
        muon_decay = []
        muon_no_decay = []
        adamw_decay = []
        adamw_no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            
            # Determine if this param should have weight decay
            # No decay for: 1D params, bias, or special params
            should_decay = not (param.ndim <= 1 or name.endswith('.bias'))
            
            if is_muon_param(name, param):
                if should_decay:
                    muon_decay.append(param)
                else:
                    muon_no_decay.append(param)
            else:
                if should_decay:
                    adamw_decay.append(param)
                else:
                    adamw_no_decay.append(param)
        
        muon_param_groups = [
            {'params': muon_decay, 'wd': wd},
            {'params': muon_no_decay, 'wd': 0.},
        ]
        adamw_param_groups = [
            {'params': adamw_decay, 'wd': wd},
            {'params': adamw_no_decay, 'wd': 0.},
        ]
        
        return Muon(
            lr=lr,
            wd=wd,
            muon_param_groups=muon_param_groups,
            adamw_param_groups=adamw_param_groups,
        )
    else:
        assert 0, f"optimizer {optimizer_name} is not supported"


def is_muon_param(name, param):
    return (
        param.ndim == 2 and 
        "token" not in name and 
        "posemb" not in name and 
        "final" not in name and
        "pred" not in name
    )