import torch
import torch.nn as nn
import torch.nn.functional as F

class NavierStokesURANS(nn.Module):
    def __init__(self, stats, re=5600):
        super(NavierStokesURANS, self).__init__()
        self.box_len = stats.get('box_len', [1.0, 1.0, 1.0])
        self.dt = stats.get('dt', 0.01) 
        
        self.nu_laminar = 1.0 / re
        self.residual_scale = 1e-6 
        
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel(kernel_size=5, sigma=1.0))

    def _create_gaussian_kernel(self, kernel_size, sigma):
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel2d = g.view(1, -1) * g.view(-1, 1)
        return kernel2d.view(1, 1, kernel_size, kernel_size)

    def smooth(self, f):
        f_padded = F.pad(f, (2, 2, 2, 2), mode='reflect')
        return F.conv2d(f_padded, self.gaussian_kernel)

    def sobel_grad(self, f, dim, dx=1.0, dy=1.0):
        if dim == 2: # d/dx
            kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=f.device).view(1, 1, 3, 3).float()
            grad = F.conv2d(f, kernel, padding=1) / 8.0
            return grad / dx
        else: # d/dy
            kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=f.device).view(1, 1, 3, 3).float()
            grad = F.conv2d(f, kernel, padding=1) / 8.0
            return grad / dy

    def compact_laplacian_1d(self, f, dim, d):
        if dim == 2: # d2/dx2
            kernel = torch.tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]], device=f.device).view(1, 1, 3, 3).float()
        else: # d2/dy2
            kernel = torch.tensor([[0, 1, 0], [0, -2, 0], [0, 1, 0]], device=f.device).view(1, 1, 3, 3).float()
        
        f_padded = F.pad(f, (1, 1, 1, 1), mode='reflect')
        grad2 = F.conv2d(f_padded, kernel)
        return grad2 / (d ** 2)

    def forward(self, pred_curr, pred_next, grid_coords, dx=None, dy=None, beta_mask=None):
        if dx is None or dy is None:
            dx = torch.abs(grid_coords[0, 0, 1, 0] - grid_coords[0, 0, 0, 0])
            dy = torch.abs(grid_coords[0, 1, 0, 1] - grid_coords[0, 0, 0, 1])
        
        dx = torch.clamp(dx, min=1e-6)
        dy = torch.clamp(dy, min=1e-6)

        u, v, p = pred_curr[:, 0:1], pred_curr[:, 1:2], pred_curr[:, 2:3]
        u_next, v_next = pred_next[:, 0:1], pred_next[:, 1:2]
        
        u_s = self.smooth(u)
        v_s = self.smooth(v)
        p_s = self.smooth(p)
        
        u_t = (u_next - u) / (self.dt + 1e-8)
        v_t = (v_next - v) / (self.dt + 1e-8)
        
        u_x = self.sobel_grad(u_s, 2, dx, dy); u_y = self.sobel_grad(u_s, 1, dx, dy)
        v_x = self.sobel_grad(v_s, 2, dx, dy); v_y = self.sobel_grad(v_s, 1, dx, dy)
        p_x = self.sobel_grad(p_s, 2, dx, dy); p_y = self.sobel_grad(p_s, 1, dx, dy)
        
        if pred_curr.shape[1] >= 4:
            nu_t_raw = pred_curr[:, 3:4]
            nu_t_s = self.smooth(nu_t_raw)
            if beta_mask is not None:
                nu_t = nu_t_s * beta_mask.detach() 
            else:
                nu_t = nu_t_s
        else:
            nu_t = 0.005

        nu_eff = self.nu_laminar + nu_t
        
        u_xx = self.compact_laplacian_1d(u_s, 2, dx)
        u_yy = self.compact_laplacian_1d(u_s, 1, dy)
        v_xx = self.compact_laplacian_1d(v_s, 2, dx)
        v_yy = self.compact_laplacian_1d(v_s, 1, dy)

        diff_u = nu_eff * (u_xx + u_yy)
        diff_v = nu_eff * (v_xx + v_yy)
        
        res_x = u_t + (u_s*u_x + v_s*u_y) + p_x - diff_u
        res_y = v_t + (u_s*v_x + v_s*v_y) + p_y - diff_v
        res_c = u_x + v_y
        
        return res_x * self.residual_scale, res_y * self.residual_scale, res_c * self.residual_scale