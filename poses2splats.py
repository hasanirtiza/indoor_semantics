import cv2
from data.dataloader import TumRGBDDataset
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from scipy.spatial import KDTree
import json


import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils.utils import  pvec2mat

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr





class LightweightGaussianModel:
    def __init__(self, points, colors):
        self.means = points
        self.rgbs = colors
        #self.scales = np.ones_like(points) * 0.02
        self.scales = np.ones_like(points) * 0.05  # Make them fatter by default
        self.opacities = np.ones(len(points)) * 0.85 

    def compute_anisotropic_scales(self, k=16, isotropy_weight=0.3):
        print(f" Computing Covariance (PCA) for {len(self.means)} splats...")
        tree = KDTree(self.means)
        dists, indices = tree.query(self.means, k=k)
        
        new_scales = []
        for i, idx_list in enumerate(indices):
            neighbors = self.means[idx_list]
            centered = neighbors - np.mean(neighbors, axis=0)
            std = np.std(centered, axis=0) + 1e-6
            
            # Isotr. Penalty -> Blend raw anisotropy with avg. scale
            avg_scale = np.mean(std)
            regularized = (1.0 - isotropy_weight) * std + isotropy_weight * avg_scale
            new_scales.append(regularized)
            
        self.scales = np.array(new_scales)

    def save_json(self, filepath):
        data = []
        for i in range(len(self.means)):
            splat = {
                "mean": [float(x) for x in self.means[i]],
                "scale": [float(x) for x in self.scales[i]],
                "rgb": [float(x) for x in self.rgbs[i]],
                "opacity": float(self.opacities[i])
            }
            data.append(splat)
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f" Saved {len(data)} splats to {filepath}")


def render_preview(model, pose, K, width=640, height=480):
    print(f" Rendering preview...")
    
    w2c = np.linalg.inv(pose)
    R_cam = w2c[:3, :3]
    t_cam = w2c[:3, 3]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    
    p_cam = (model.means @ R_cam.T) + t_cam
    mask = p_cam[:, 2] > 0.2 
    
    p_cam = p_cam[mask]
    colors = model.rgbs[mask]
    scales = model.scales[mask]
    opacities = model.opacities[mask]
    
    if len(p_cam) == 0: return Image.new('RGB', (width, height))

    u = (p_cam[:, 0] * fx / p_cam[:, 2]) + cx
    v = (p_cam[:, 1] * fy / p_cam[:, 2]) + cy
    
    depths = p_cam[:, 2]
    indices = np.argsort(depths)[::-1]
    
    img = Image.new('RGBA', (width, height), (0,0,0,255))
    draw = ImageDraw.Draw(img)
    
    for idx in indices:
        x, y = u[idx], v[idx]
        z = depths[idx]
        
        # Skip off-screen
        if x < -50 or x > width+50 or y < -50 or y > height+50: continue

        raw_radius = (np.max(scales[idx]) * fx) / z
 
        radius = min(raw_radius, 40.0) 
        
        radius = max(1.0, radius)

        r, g, b = (colors[idx] * 255).astype(int)

        a_val = opacities[idx]
        if raw_radius > 30: 
            a_val *= 0.5 
            
        a = int(a_val * 255)
        
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=(r,g,b,a))
        
    return img


def evaluate_rendering_quality(model, gt_rgb_path, estimated_pose, K, width=640, height=480):
# quick way to check how well is you splatting
    print(f"\n--- Evaluating Quality against {os.path.basename(gt_rgb_path)} ---")

    gt_img = cv2.imread(gt_rgb_path)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    gt_img = cv2.resize(gt_img, (width, height)) # Ensure dimensions match
    render_pil = render_preview(model, estimated_pose, K, width, height)
    render_np = np.array(render_pil)
    render_rgb = render_np[:, :, :3]
    score_psnr = psnr(gt_img, render_rgb)
    score_ssim = ssim(gt_img, render_rgb, channel_axis=2, data_range=255)
    print(f"PSNR: {score_psnr:.2f} dB (Higher is better)")
    print(f"SSIM: {score_ssim:.4f} (1.0 is perfect)")
    diff = cv2.absdiff(gt_img, render_rgb)
    diff_viz = np.clip(diff * 3, 0, 255).astype(np.uint8)
    combined = np.hstack((gt_img, render_rgb, diff_viz))
    
    return combined, score_psnr, score_ssim




def poses2splats(smoothed_est_traj, dataset_params, dataset_path, depth_list, rgb_list, preview_idx=None): 
    # --- 1. Fuse Point Cloud from Poses ---
    print("Fusing point cloud from computed poses...")
    fused_points = []
    fused_colors = []

    K = dataset_params["camera_matrix"]
    inv_fx = 1.0 / K[0,0]
    inv_fy = 1.0 / K[1,1]
    cx, cy = K[0,2], K[1,2]

    FRAME_STRIDE = 20  # Process every 20th frame to keep it lightweight
    PIXEL_STEP = 8     # Process every 8th pixel

    for i in range(0, len(smoothed_est_traj), FRAME_STRIDE):
        depth_path = os.path.join(dataset_path, depth_list[i])
        rgb_path = os.path.join(dataset_path, rgb_list[i])
        
        d_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
        c_img = cv2.imread(rgb_path)
        c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB) 
        
        pose_vec = smoothed_est_traj[i]
        T_w_c = pvec2mat(pose_vec) # Cam -> World
        
        H, W = d_img.shape
        y_grid, x_grid = np.meshgrid(np.arange(0, H, PIXEL_STEP), np.arange(0, W, PIXEL_STEP), indexing='ij')
        
        z_val = d_img[0:H:PIXEL_STEP, 0:W:PIXEL_STEP]
        colors = c_img[0:H:PIXEL_STEP, 0:W:PIXEL_STEP] / 255.0 # Normalize 0-1
        
        mask = (z_val > dataset_params["min_depth"]) & (z_val < dataset_params["max_depth"])
        
        if np.sum(mask) == 0: continue
        
        z_val = z_val[mask]
        x_val = x_grid[mask]
        y_val = y_grid[mask]
        colors = colors[mask]
        
        x_cam = (x_val - cx) * z_val * inv_fx
        y_cam = (y_val - cy) * z_val * inv_fy
        z_cam = z_val
        
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1) # (N, 3)
        
        # T_w_c @ [x,y,z,1]
        R_w_c = T_w_c[:3, :3]
        t_w_c = T_w_c[:3, 3]
        
        pts_world = (pts_cam @ R_w_c.T) + t_w_c
        
        fused_points.append(pts_world)
        fused_colors.append(colors)

    global_points = np.vstack(fused_points)
    global_colors = np.vstack(fused_colors)

    print(f"Total Fused Points: {len(global_points)}")

    gs = LightweightGaussianModel(global_points, global_colors)

    gs.compute_anisotropic_scales(k=12, isotropy_weight=0.5)

    gs.save_json("results/splats.json")

    if preview_idx is None:
        mid_idx = len(smoothed_est_traj) // 2
    else:
        mid_idx = preview_idx
    preview_pose = pvec2mat(smoothed_est_traj[mid_idx])

    preview_img = render_preview(gs, preview_pose, K)


    gt_image_path = os.path.join(dataset_path, rgb_list[mid_idx])

    # Run Evaluation
    comparison_img, p_score, s_score = evaluate_rendering_quality(gs, gt_image_path, preview_pose, K)

    # Save the Comparison

    preview_img.save("results/splat_preview.png")
    Image.fromarray(comparison_img).save("results/quality_check.png")
    print("Preview saved to results/splat_preview.png")