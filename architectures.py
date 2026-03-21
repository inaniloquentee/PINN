import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 高频傅里叶特征映射 (Hybrid Embedding)
# ==========================================
class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        self.register_buffer("B", torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        # x shape: [B, C, H, W] -> permute to [B, H, W, C] for matmul
        x_proj = torch.matmul(x.permute(0, 2, 3, 1), self.B)
        x_proj = 2 * math.pi * x_proj
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out.permute(0, 3, 1, 2) # return back to [B, C, H, W]

# ==========================================
# 2. 空间调制图到网格投影器 (Projector) - [V1 核心修改区]
# ==========================================
class SpatiallyModulatedProjector(nn.Module):
    def __init__(self, sensor_in_dim=3, sensor_count=65, grid_feat_dim=32):
        super().__init__()
        self.sensor_count = sensor_count
        self.grid_feat_dim = grid_feat_dim
        
        # 传感器特征提取
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_in_dim + 2, 64), # 3 val + 2 pos
            nn.SiLU(),
            nn.Linear(64, grid_feat_dim)
        )
        
        # 交叉注意力机制组件 (Q: Grid, K,V: Sensors)
        self.grid_proj = nn.Conv2d(2, grid_feat_dim, 1) # Grid pos (x,y)
        self.attn_scale = grid_feat_dim ** -0.5
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(grid_feat_dim, grid_feat_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(grid_feat_dim, grid_feat_dim, 3, padding=1)
        )

        # 🛑 [V1 核心删减]: 彻底删除了 self.gate_threshold 和 self.gate_scale 参数
        # 网络在此变体中失去了自寻优能力！

    def forward(self, s_val, s_pos, grid_pos_norm):
        B, N, C = s_val.shape
        _, _, H, W = grid_pos_norm.shape
        
        # 1. 准备传感器特征 (K, V)
        sensor_input = torch.cat([s_val, s_pos], dim=-1) # [B, N, 5]
        sensor_feat = self.sensor_mlp(sensor_input)      # [B, N, dim]
        
        # 2. 准备网格查询特征 (Q)
        grid_q = self.grid_proj(grid_pos_norm)           # [B, dim, H, W]
        grid_q_flat = grid_q.view(B, self.grid_feat_dim, -1).permute(0, 2, 1) # [B, H*W, dim]
        
        # 3. 计算交叉注意力 (Cross-Attention)
        attn_scores = torch.bmm(grid_q_flat, sensor_feat.permute(0, 2, 1)) * self.attn_scale # [B, H*W, N]
        attn_weights = F.softmax(attn_scores, dim=-1)    # [B, H*W, N]
        
        # 4. 融合特征映射到网格
        mapped_feat_flat = torch.bmm(attn_weights, sensor_feat) # [B, H*W, dim]
        mapped_feat = mapped_feat_flat.permute(0, 2, 1).view(B, self.grid_feat_dim, H, W)
        mapped_feat = self.out_conv(mapped_feat)

        # ====================================================
        # 🛑 [V1 核心修改]: 硬编码门控 (Hardcoded Spatial Gate)
        # ====================================================
        # 提取网格的 X 坐标通道 (假设 0 通道是 X, 范围 [-1, 1])
        grid_x = grid_pos_norm[:, 0:1, :, :] 
        
        # 强制在突扩台阶后方 (例如 x > -0.2) 开启特征，而不是根据物理平均流
        # 这里的 50.0 是一个极陡的斜率，用于模拟物理上清晰的切割线
        hardcoded_beta = torch.sigmoid(50.0 * (grid_x - (-0.2)))
        
        # 将硬编码掩码施加于提取的特征上
        modulated_feat = mapped_feat * hardcoded_beta
        
        return modulated_feat

# ==========================================
# 3. U-Net 基础组件
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ==========================================
# 4. 主模型：PIGU-Hybrid
# ==========================================
class PIGU_Hybrid(nn.Module):
    def __init__(self, sensor_in_dim=3, sensor_count=65, out_dim=4):
        super().__init__()
        
        self.projector = SpatiallyModulatedProjector(sensor_in_dim, sensor_count, grid_feat_dim=32)
        self.fourier_embed = GaussianFourierFeatureTransform(input_dim=2, mapping_size=16) # Output 32 channels (16 sin, 16 cos)
        
        # Encoder 输入通道: Projector(32) + Fourier(32) + BaseFlow(3) = 67
        in_channels = 32 + 32 + 3
        
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        # 输出通道: u', v', p', nu_t_raw
        self.outc = nn.Conv2d(64, out_dim, kernel_size=1)

    def forward(self, s_val, s_pos, grid_pos_norm, base_flow=None):
        # 1. 传感器图投影 (V1 中它使用内部的硬编码掩码)
        proj_feat = self.projector(s_val, s_pos, grid_pos_norm)
        
        # 2. 网格高频位置编码
        grid_fourier = self.fourier_embed(grid_pos_norm)
        
        # 3. 特征拼接 (在 V1 和 Baseline++ 中，平均流依然作为底层输入特征被喂给网络)
        if base_flow is not None:
            x = torch.cat([proj_feat, grid_fourier, base_flow], dim=1)
        else:
            # 兼容性冗余处理
            dummy_base = torch.zeros_like(grid_pos_norm[:, :3, :, :])
            x = torch.cat([proj_feat, grid_fourier, dummy_base], dim=1)

        # 4. U-Net 编解码过程
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = self.conv_up1(torch.cat([x, x3], dim=1))
        
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x2], dim=1))
        
        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x1], dim=1))
        
        # 5. 输出脉动量与涡粘性
        fluc_out = self.outc(x)
        
        # 6. 雷诺分解：全量流场 = 均值 + 脉动
        # 注意: 只有速度和压力需要叠加均值，第 4 通道涡粘性不需要
        if base_flow is not None and fluc_out.shape[1] >= 3:
            fluc_uvp = fluc_out[:, :3, :, :]
            base_uvp = base_flow[:, :3, :, :]
            final_uvp = fluc_uvp + base_uvp
            
            if fluc_out.shape[1] == 4:
                # 为了防止 Raw 涡粘性输出负值，可以加一个 Softplus 或绝对值
                # 这里为了兼容原始逻辑，直接切片拼接
                nu_t_raw = F.softplus(fluc_out[:, 3:4, :, :]) 
                return torch.cat([final_uvp, nu_t_raw], dim=1)
            else:
                return final_uvp
                
        return fluc_out