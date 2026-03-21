import sys
import os
# 强行将子目录加入路径，解决找不到 dataset 的问题
sys.path.append(os.path.abspath("./baseline")) 

import torch
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation

from dataset import CFDReconstructionDataset
from architectures import PIGU_Hybrid
from model import NavierStokesURANS

# ==========================================
# 1. 评估全局配置
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SENSOR_COUNT = 65
DT = 0.05

VARIANTS = {
    "Baseline (Physics Dynamic Gate)": "./baseline/results_adv1b/model_ep1900.pth",
    "Variant 1 (Hardcoded Gate)": "./Variant_1 Hardcoded Gate/results/model_ep1900.pth",
    # "Variant 2 (w/o Spatial Gate)": "./Variant_2 w o Spatial Gate/results_ablation_v2/model_ep1900.pth",
    "Variant 4 (w/o PDE Loss)": "./Variant_4 PDE_Loss/results_ablation_v4/model_ep1900.pth",
}

PATH_UNSTEADY = "../../Ablation Experiment/dataset_sin/flow_one_period.npy"
PATH_MEAN = "../../Ablation Experiment/dataset_sin/mean_flow_steady.npy"
EVAL_FRAMES = 100 

def compute_vorticity(u, v, dx, dy):
    dv_dy, dv_dx = torch.gradient(v, spacing=(dy, dx), dim=(-2, -1))
    du_dy, du_dx = torch.gradient(u, spacing=(dy, dx), dim=(-2, -1))
    return dv_dx - du_dy

def get_boundary_mask(solid_mask):
    solid_np = solid_mask.cpu().numpy()
    dilated = binary_dilation(solid_np, iterations=2)
    boundary_np = dilated ^ solid_np  
    return torch.from_numpy(boundary_np).to(DEVICE)

def evaluate_variant(name, weight_path, dataset, pde_engine, dx_val, dy_val, valid_mask, boundary_mask):
    print(f"\n[*] Evaluating: {name}")
    
    model = PIGU_Hybrid(sensor_in_dim=3, sensor_count=SENSOR_COUNT).to(DEVICE)
    
    try:
        # [修改] 添加了 weights_only=True 消除安全警告
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    except Exception as e:
        print(f"[!] Failed to load weights for {name}. Architecture mismatch? Error: {e}")
        return None
        
    model.eval()

    metrics = {
        "L2_Velocity": [],
        "Continuity_Res_RMS": [],
        "Momentum_Res_RMS": [],
        "Vorticity_L2": [],
        "TKE_L2": [],
        "Boundary_Pressure_L2": [] 
    }
    
    stats_max = dataset.stats['max'].to(DEVICE)
    stats_min = dataset.stats['min'].to(DEVICE)
    
    def denormalize(norm_tensor):
        return (norm_tensor + 1) / 2 * (stats_max - stats_min) + stats_min

    dx_tensor = torch.tensor(dx_val, device=DEVICE)
    dy_tensor = torch.tensor(dy_val, device=DEVICE)

    num_eval = min(EVAL_FRAMES, len(dataset))
    
    with torch.no_grad():
        for idx in range(num_eval):
            s_val_t, s_pos, grid_pos_norm, s_val_next, mean_flow = dataset[idx]
            
            s_val_t = s_val_t.unsqueeze(0).to(DEVICE)
            s_pos = s_pos.unsqueeze(0).to(DEVICE)
            grid_pos_norm = grid_pos_norm.unsqueeze(0).to(DEVICE)
            s_val_next = s_val_next.unsqueeze(0).to(DEVICE)
            mean_flow = mean_flow.unsqueeze(0).to(DEVICE)
            
            true_norm_t = dataset.data[idx].unsqueeze(0).to(DEVICE)

            pred_norm_t = model(s_val_t, s_pos, grid_pos_norm, base_flow=mean_flow)
            pred_norm_next = model(s_val_next, s_pos, grid_pos_norm, base_flow=mean_flow)
            
            p_phys = denormalize(pred_norm_t)
            p_next_phys = denormalize(pred_norm_next)
            t_phys = denormalize(true_norm_t)
            mean_phys = denormalize(mean_flow)
            
            u_p, v_p, pres_p = p_phys[0, 0], p_phys[0, 1], p_phys[0, 2]
            u_t, v_t, pres_t = t_phys[0, 0], t_phys[0, 1], t_phys[0, 2]
            u_m, v_m = mean_phys[0, 0], mean_phys[0, 1]

            uv_p = p_phys[0, 0:2][:, valid_mask]
            uv_t = t_phys[0, 0:2][:, valid_mask]
            l2_err = torch.norm(uv_p - uv_t) / (torch.norm(uv_t) + 1e-8)
            metrics["L2_Velocity"].append(l2_err.item())

            res_x, res_y, res_c = pde_engine(p_phys, p_next_phys, None, dx=dx_tensor, dy=dy_tensor)
            resc_valid = res_c[0, 0][valid_mask]
            resx_valid = res_x[0, 0][valid_mask]
            resy_valid = res_y[0, 0][valid_mask]
            
            metrics["Continuity_Res_RMS"].append(torch.sqrt(torch.mean(resc_valid**2)).item())
            metrics["Momentum_Res_RMS"].append(torch.sqrt(torch.mean(resx_valid**2 + resy_valid**2)).item())

            vort_p = compute_vorticity(u_p, v_p, dx_val, dy_val)
            vort_t = compute_vorticity(u_t, v_t, dx_val, dy_val)
            vort_err = torch.norm(vort_p[valid_mask] - vort_t[valid_mask]) / (torch.norm(vort_t[valid_mask]) + 1e-8)
            metrics["Vorticity_L2"].append(vort_err.item())

            tke_p = 0.5 * ((u_p - u_m)**2 + (v_p - v_m)**2)
            tke_t = 0.5 * ((u_t - u_m)**2 + (v_t - v_m)**2)
            tke_err = torch.norm(tke_p[valid_mask] - tke_t[valid_mask]) / (torch.norm(tke_t[valid_mask]) + 1e-8)
            metrics["TKE_L2"].append(tke_err.item())

            pres_b_p = pres_p[boundary_mask]
            pres_b_t = pres_t[boundary_mask]
            pres_err = torch.norm(pres_b_p - pres_b_t) / (torch.norm(pres_b_t) + 1e-8)
            metrics["Boundary_Pressure_L2"].append(pres_err.item())

    result_row = {"Variant": name}
    for k, v in metrics.items():
        result_row[k] = np.mean(v)
        
    return result_row

def main():
    print("[*] Initializing Unified Evaluation Script...")
    dataset = CFDReconstructionDataset(PATH_UNSTEADY, PATH_MEAN, SENSOR_COUNT, dt=DT)
    pde_engine = NavierStokesURANS(dataset.stats).to(DEVICE)
    
    dx_val = dataset.box_len[0] / (dataset.W - 1)
    dy_val = dataset.box_len[1] / (dataset.H - 1)

    print("[*] Generating Physical Geometry Masks...")
    raw_mean = np.load(PATH_MEAN).astype(np.float32)
    coords_mean = raw_mean[:, :2]
    tree = cKDTree(coords_mean)
    grid_pts = np.stack([dataset.grid_X.ravel(), dataset.grid_Y.ravel()], axis=-1)
    dist, _ = tree.query(grid_pts)
    threshold = max(dx_val, dy_val) * 2.5 
    is_solid_wall = (dist > threshold).reshape(dataset.H, dataset.W)
    
    valid_mask = ~torch.from_numpy(is_solid_wall).to(DEVICE)
    boundary_mask = get_boundary_mask(torch.from_numpy(is_solid_wall))

    results_list = []
    
    for name, path in VARIANTS.items():
        if os.path.exists(path):
            res = evaluate_variant(name, path, dataset, pde_engine, dx_val, dy_val, valid_mask, boundary_mask)
            if res:
                results_list.append(res)
        else:
            print(f"[!] Skipping {name}: Weights not found at {path}")

    if results_list:
        df = pd.DataFrame(results_list)
        pd.options.display.float_format = '{:.2e}'.format  
        
        print("\n" + "="*80)
        print("🚀 Final Quantitative Ablation Study Results 🚀")
        print("="*80)
        
        # [核心修复] 使用更纯净的方法打印，不依赖外部包
        print(df.to_string(index=False, justify='center'))
        print("="*80)
        
        # 确保输出目录存在
        os.makedirs("eval_results", exist_ok=True)
        df.to_csv("eval_results/comprehensive_ablation_metrics.csv", index=False)
        print("[+] Results saved to eval_results/comprehensive_ablation_metrics.csv")
    else:
        print("[!] No valid results generated.")

if __name__ == "__main__":
    main()