import torch


class Transport:
    def __init__(self, P_mean=0.0, P_std=1.0):
        self.P_mean = P_mean
        self.P_std = P_std        

    def interpolant(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        alpha_t = 1 - t
        sigma_t = t
        x_t = alpha_t * x + sigma_t * z
        return x_t

    def sample_t(self, batch_size, dtype, device):
        rnd_normal = torch.randn((batch_size, ), dtype=dtype, device=device)
        t = torch.sigmoid(rnd_normal * self.P_std + self.P_mean)     # [0, 1]
        return t
    
    def c_noise(self, t: torch.Tensor):
        return t
    
    def c_out(self, F: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        return (x_t - F) / t
    
    
    def from_x_t_to_x_r(self, x_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor, F: torch.Tensor, s_ratio=0.0):
        A = 1
        B = - (t-r)
        x_r = A * x_t + B * F
        if s_ratio > 0.0:
            z = x_t + (1-t) * F # an approximation of z
            epsilon = torch.randn_like(z)
            dt = t-r
            x_r = x_r - s_ratio * z * dt + torch.sqrt(s_ratio*2*t*dt) * epsilon
        return x_r
    





