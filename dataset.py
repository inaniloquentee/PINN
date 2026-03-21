import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.interpolate import griddata
from scipy.linalg import qr

class CFDReconstructionDataset(Dataset):
    def __init__(self, unsteady_path, mean_path, sensor_count=65, target_grid=(128, 256), dt=0.01):
        super().__init__()
        print("[Dataset] Initializing v3.0 (Sparse Mode: 5 returns)...") # 打印版本号
        self.sensor_count = sensor_count
        self.H, self.W = target_grid
        self.dt = dt 
        
        try:
            raw_unsteady = np.load(unsteady_path).astype(np.float32)
            raw_mean = np.load(mean_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

        # 1. 预处理
        mean_cols = raw_mean.shape[1] if raw_mean.ndim > 1 else 1
        coords_mean = raw_mean[:, :2]
        x_min, y_min = coords_mean.min(axis=0)
        x_max, y_max = coords_mean.max(axis=0)
        self.box_len = [x_max - x_min, y_max - y_min, 1.0]
        
        grid_x = np.linspace(x_min, x_max, self.W)
        grid_y = np.linspace(y_min, y_max, self.H)
        self.grid_X, self.grid_Y = np.meshgrid(grid_x, grid_y)
        self.grid_coords = torch.stack([torch.tensor(self.grid_X), torch.tensor(self.grid_Y)], dim=-1).float()
        self.grid_coords_norm = self.grid_coords.clone()
        self.grid_coords_norm[..., 0] = 2 * (self.grid_coords[..., 0] - x_min) / (x_max - x_min) - 1
        self.grid_coords_norm[..., 1] = 2 * (self.grid_coords[..., 1] - y_min) / (y_max - y_min) - 1

        if mean_cols >= 6: mean_vals = raw_mean[:, 3:6]
        elif mean_cols == 5: mean_vals = raw_mean[:, 2:5]
        else: mean_vals = raw_mean[:, -3:]

        self.mean_data = griddata(coords_mean, mean_vals, (self.grid_X, self.grid_Y), method='linear', fill_value=0)
        self.mean_data = torch.from_numpy(self.mean_data).permute(2, 0, 1).float()

        unsteady_cols = raw_unsteady.shape[-1] if raw_unsteady.ndim > 1 else raw_mean.shape[-1]
        points_per_frame = raw_mean.shape[0]
        total_elements = raw_unsteady.size
        frame_size = points_per_frame * unsteady_cols
        num_frames = total_elements // frame_size
        raw_unsteady = raw_unsteady.flatten()[:num_frames * frame_size]
        raw_unsteady = raw_unsteady.reshape(num_frames, points_per_frame, unsteady_cols)

        process_frames = min(num_frames, 200)
        data_list = []
        for i in range(process_frames):
            frame_data = raw_unsteady[i]
            if unsteady_cols >= 6: values = frame_data[:, 3:6]
            elif unsteady_cols == 5: values = frame_data[:, 2:5]
            else: values = frame_data[:, :3]
            coords = frame_data[:, :2] if unsteady_cols >= 5 else coords_mean
            grid_val = griddata(coords, values, (self.grid_X, self.grid_Y), method='linear', fill_value=0)
            data_list.append(grid_val)
            
        self.data = np.stack(data_list, axis=0) 
        self.data = torch.from_numpy(self.data).permute(0, 3, 1, 2).float()

        self.stats = {}
        self.max_vals = torch.amax(self.data, dim=(0, 2, 3), keepdim=True)
        self.min_vals = torch.amin(self.data, dim=(0, 2, 3), keepdim=True)
        self.stats['max'] = self.max_vals
        self.stats['min'] = self.min_vals
        self.stats['dt'] = self.dt
        self.stats['box_len'] = self.box_len
        
        denom = self.max_vals - self.min_vals
        denom[denom < 1e-8] = 1.0
        self.data = 2 * (self.data - self.min_vals) / denom - 1
        self.mean_data = 2 * (self.mean_data - self.min_vals.squeeze(0)) / denom.squeeze(0) - 1
        
        print(f"[Dataset] Selecting {sensor_count} sensors (QR Pivot)...")
        self.sensor_indices = self.compute_qr_sensors(self.data, self.mean_data, sensor_count)

    def compute_qr_sensors(self, data, mean, num_sensors):
        T, C, H, W = data.shape
        fluctuations = (data[:, 0, :, :] - mean[0, :, :]).reshape(T, -1).numpy().T 
        k = min(T, num_sensors + 10) 
        U, _, _ = np.linalg.svd(fluctuations, full_matrices=False)
        Psi = U[:, :k] 
        _, _, P = qr(Psi.T, pivoting=True)
        best_indices_flat = P[:num_sensors]
        sensor_locs = []
        for idx in best_indices_flat:
            y = idx // W
            x = idx % W
            sensor_locs.append((y, x))
        return sensor_locs

    def denormalize(self, tensor):
        return (tensor + 1) / 2 * (self.stats['max'].to(tensor.device) - self.stats['min'].to(tensor.device)) + self.stats['min'].to(tensor.device)

    def __len__(self):
        return max(0, self.data.shape[0] - 1)

    def __getitem__(self, idx):
        if idx >= self.data.shape[0] - 1: idx = self.data.shape[0] - 2
        target_t = self.data[idx]
        target_next = self.data[idx + 1]
        
        vals_t, coords_t = [], []
        for y, x in self.sensor_indices:
            vals_t.append(target_t[:, y, x])
            coords_t.append(self.grid_coords_norm[y, x])
        
        vals_next = []
        for y, x in self.sensor_indices:
            vals_next.append(target_next[:, y, x])
        
        # 返回 5 个变量 (绝对不含全量数据)
        return (torch.stack(vals_t), torch.stack(coords_t), self.grid_coords_norm, 
                torch.stack(vals_next), self.mean_data)