import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion

from dataset import CFDReconstructionDataset
from architectures import PIGU_Hybrid
from model import NavierStokesURANS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 2000
SENSOR_COUNT = 65

# ==========================================
# 🛑 Variant 1 配置：硬编码参数
# ==========================================
DATA_WEIGHT = 50.0      
PHYS_WEIGHT = 1.0       
WALL_WEIGHT = 5.0 
DT = 0.05
# 注意：V1 不再需要 GATE_REG_WEIGHT

TV_WEIGHT_UV = 1.0      
TV_WEIGHT_NUT = 0.1     

PATH_UNSTEADY = "../../Ablation Experiment/dataset_sin/flow_one_period.npy"
PATH_MEAN = "../../Ablation Experiment/dataset_sin/mean_flow_steady.npy"

OUTPUT_DIR = "results_variant1_hardcoded"

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print(f"[!] 启动 Variant 1: 硬编码空间门控实验 (无动态物理感知).")
    
    dataset = CFDReconstructionDataset(PATH_UNSTEADY, PATH_MEAN, SENSOR_COUNT, dt=DT)
    sensor_y = torch.tensor([p[0] for p in dataset.sensor_indices], device=DEVICE)
    sensor_x = torch.tensor([p[1] for p in dataset.sensor_indices], device=DEVICE)
    
    stats_max = dataset.stats['max'].to(DEVICE)
    stats_min = dataset.stats['min'].to(DEVICE)
    box_len = dataset.stats['box_len']
    W, H = dataset.W, dataset.H
    
    dx_val = box_len[0] / (W - 1)
    dy_val = box_len[1] / (H - 1)
    dx_tensor = torch.tensor(dx_val, device=DEVICE)
    dy_tensor = torch.tensor(dy_val, device=DEVICE)
    
    print("[*] 正在解析几何边界生成 PDE 安全掩码...")
    raw_mean = np.load(PATH_MEAN).astype(np.float32)
    tree = cKDTree(raw_mean[:, :2])
    grid_x_np = np.linspace(raw_mean[:, 0].min(), raw_mean[:, 0].max(), W)
    grid_y_np = np.linspace(raw_mean[:, 1].min(), raw_mean[:, 1].max(), H)
    grid_X, grid_Y = np.meshgrid(grid_x_np, grid_y_np)
    grid_pts = np.stack([grid_X.ravel(), grid_Y.ravel()], axis=-1)
    dist, _ = tree.query(grid_pts)
    is_solid = (dist > max(dx_val, dy_val) * 2.5).reshape(H, W)
    safe_fluid = binary_erosion(~is_solid, iterations=3)
    pde_safe_mask = torch.from_numpy(safe_fluid).float().to(DEVICE).view(1, 1, H, W)

    def denormalize_batch(norm_tensor):
        uvp_norm = norm_tensor[:, :3, :, :]
        uvp_phys = (uvp_norm + 1) / 2 * (stats_max - stats_min) + stats_min
        if norm_tensor.shape[1] == 4:
            nu_t_raw = norm_tensor[:, 3:4, :, :] 
            return torch.cat([uvp_phys, nu_t_raw], dim=1)
        return uvp_phys

    # 加载 V1 架构 (Projector 内部已改为硬编码)
    model = PIGU_Hybrid(sensor_in_dim=3, sensor_count=SENSOR_COUNT, out_dim=4).to(DEVICE)
    pde_engine = NavierStokesURANS(dataset.stats).to(DEVICE)
    sparse_loss_fn = torch.nn.L1Loss() 
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    wall_mask = torch.zeros(1, 1, H, W).to(DEVICE)
    wall_mask[..., 0, :] = 1.0 
    wall_mask[..., -1, :] = 1.0 
    
    print(f"[*] 训练开始...")
    
    for epoch in range(EPOCHS):
        total_sparse, total_phys = 0, 0
        warmup = min(1.0, epoch / 300.0)
        
        for batch in loader:
            batch = [x.to(DEVICE) for x in batch]
            s_val_t, s_pos, grid_pos_norm, s_val_next, mean_flow = batch[:5]
            
            # 前向传播
            pred_norm_t = model(s_val_t, s_pos, grid_pos_norm, base_flow=mean_flow)
            pred_norm_next = model(s_val_next, s_pos, grid_pos_norm, base_flow=mean_flow)
            
            # 数据损失
            p_s_t = pred_norm_t[:, :3, sensor_y, sensor_x].permute(0, 2, 1) 
            p_s_n = pred_norm_next[:, :3, sensor_y, sensor_x].permute(0, 2, 1)
            loss_data = sparse_loss_fn(p_s_t, s_val_t) + sparse_loss_fn(p_s_n, s_val_next)
            
            # 反归一化
            phys_t = denormalize_batch(pred_norm_t)
            phys_n = denormalize_batch(pred_norm_next)
            
            # 🛑 V1 核心逻辑：这里手动定义一个 beta_mask 传给 PDE 引擎
            # 确保 PDE 引擎在计算时，物理逻辑与架构中的硬编码掩码一致
            grid_x = grid_pos_norm[:, 0:1, :, :]
            beta_mask = torch.sigmoid(50.0 * (grid_x - (-0.2))) 
            
            res_x, res_y, res_c = pde_engine(phys_t, phys_n, None, dx=dx_tensor, dy=dy_tensor, beta_mask=beta_mask)
            
            loss_phys = F.smooth_l1_loss(res_x * pde_safe_mask, torch.zeros_like(res_x)) + \
                        F.smooth_l1_loss(res_y * pde_safe_mask, torch.zeros_like(res_y)) + \
                        F.smooth_l1_loss(res_c * pde_safe_mask, torch.zeros_like(res_c))
            
            loss_wall = torch.mean((phys_t[:, 0:2] * wall_mask)**2)
            
            # 正则项
            uv = phys_t[:, 0:2]
            loss_tv = torch.mean(torch.abs(uv[:,:,:,1:]-uv[:,:,:,:-1])) + torch.mean(torch.abs(uv[:,:,1:,:]-uv[:,:,:-1,:]))
            
            # 总损失
            loss = DATA_WEIGHT * loss_data + PHYS_WEIGHT * warmup * loss_phys + \
                   WALL_WEIGHT * warmup * loss_wall + TV_WEIGHT_UV * loss_tv
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 🛑 关键修复点：
            # 此处删除了原本尝试对 model.projector.gate_scale 进行 .clamp_() 的代码行。
            # 因为 Variant 1 的架构中已经没有这个参数了。

            total_sparse += loss_data.item()
            total_phys += loss_phys.item()
            
        avg_sparse = total_sparse / len(loader)
        avg_phys = total_phys / len(loader)
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f"Ep {epoch:04d} | Data: {avg_sparse:.5f} | PDE: {avg_phys:.5f} | V1 Run")
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/v1_ep{epoch:03d}.pth")
            
            # 可视化
            with torch.no_grad():
                u_img = phys_t[0, 0].cpu().numpy()
                nut_img = (phys_t[0, 3] * beta_mask[0, 0]).cpu().numpy()
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].imshow(u_img, cmap='jet', origin='lower'); ax[0].set_title("U-Velocity")
                ax[1].imshow(nut_img, cmap='magma', origin='lower'); ax[1].set_title("Hardcoded Nu_t")
                plt.savefig(f"{OUTPUT_DIR}/vis_{epoch:03d}.png"); plt.close()

if __name__ == "__main__":
    main()