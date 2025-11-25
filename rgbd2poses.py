import cv2
from data.dataloader import TumRGBDDataset
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils.utils import depth_to_3d, mat2pvec
from scipy.optimize import least_squares

def reprojection_residuals(dof, object_points, image_points, K):
    # dof: [rx, ry, rz, tx, ty, tz]
    rvec = dof[:3]
    tvec = dof[3:]
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
    return (projected.reshape(-1, 2) - image_points).ravel()

def refine_pose(obj_points, img_points, rvec_init, tvec_init, K):
    # Initial guess
    x0 = np.hstack((rvec_init.ravel(), tvec_init.ravel()))
    
    # Switch to 'trf' (Trust Region Reflective) which supports robust loss
    res = least_squares(
        reprojection_residuals, 
        x0, 
        args=(obj_points, img_points, K),
        method='trf',     # <--- Changed from 'lm' to 'trf'
        loss='soft_l1',   # Now this works! Ignores remaining outliers.
        f_scale=1.0,
        xtol=1e-8,        # Quit when changes are tiny
        ftol=1e-8
    )
    
    rvec_refined = res.x[:3].reshape(3, 1)
    tvec_refined = res.x[3:].reshape(3, 1)
    return rvec_refined, tvec_refined

def run_pnp(obj_points, img_points, camera_matrix, reprojectionError=2.0, confidence=0.99,
                                                     iterationsCount=200):

    retval, rvec1, tvec1, inliers = cv2.solvePnPRansac( obj_points, img_points, camera_matrix, 
                                                     None, reprojectionError=2.0, confidence=0.99,
                                                     iterationsCount=200,
                                                     flags=cv2.SOLVEPNP_ITERATIVE
    )

    return retval, rvec1, tvec1, inliers



def build_2d_3d_correspondance(kp1, kp2,  prev_depth, good_matches, dataset_params ):
    obj_points = []   # 3d
    img_points = []   # 2d
    H, W = prev_depth.shape
    for m in good_matches:  # Todo: later emrge this loop into first one of matches
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt

        u, v = int(round(u1)), int(round(v1))
        if u < dataset_params["edge"] or v < dataset_params["edge"] or u >= (W - dataset_params["edge"]) or v >= (H - dataset_params["edge"]):
            continue

            
        d = prev_depth[int(v), int(u)]
        if not np.isfinite(d) or d <= 0:
            continue
        if d < dataset_params["min_depth"] or d > dataset_params["max_depth"]:
            continue

        X = depth_to_3d((u1, v1), prev_depth, dataset_params["camera_matrix"]) 
        if X is None:
            continue
        obj_points.append(X.astype(np.float32))
        img_points.append(np.array([u2, v2], dtype=np.float32))

    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)

    return obj_points, img_points


def run_rgbd2poses(dataset_path, rgb_list, depth_list, gt_tum_pose, dataset_params):


    akaze = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Init(world <- cam_0);  GT in TUM convention
    T_w_prev_est      = gt_tum_pose[0].copy()   # or np.eye(4) .... for pure odom

    time_stamp = []
    time_stamp.append(0)

    # trajectory vars init with first pose
    gt_traj = []
    est_traj = []
    gt_traj.append(mat2pvec(gt_tum_pose[0]))
    est_traj.append(mat2pvec(gt_tum_pose[0]))




    for i in range(1, len(rgb_list)):
        prev_rgb = cv2.imread(os.path.join(dataset_path, rgb_list[i-1]))
        curr_rgb = cv2.imread(os.path.join(dataset_path, rgb_list[i]))
        prev_depth = cv2.imread(os.path.join(dataset_path, depth_list[i-1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
 #       curr_depth = cv2.imread(os.path.join(dataset_path, depth_list[i]),   cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0


        kp1, des1 = akaze.detectAndCompute(prev_rgb, None)
        kp2, des2 = akaze.detectAndCompute(curr_rgb, None)
        matches = bf.knnMatch(des1, des2, k=2)
        ratio_thresh = 0.75
        good_matches = []  
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        obj_points, img_points = build_2d_3d_correspondance(kp1, kp2,  prev_depth, good_matches, dataset_params)
        
        if len(obj_points) < 6:
            T_w_curr_est = T_w_prev_est.copy()

        retval, rvec1, tvec1, inliers = run_pnp(obj_points, img_points, dataset_params["camera_matrix"])
        obj_in = obj_points[inliers.ravel()]
        img_in = img_points[inliers.ravel()]

        rvec, tvec = refine_pose(obj_in, img_in, rvec1, tvec1, dataset_params["camera_matrix"])
        R, _ = cv2.Rodrigues(rvec)
        est_pose = np.eye(4)
        est_pose[:3, :3] = R
        est_pose[:3, 3] = tvec.flatten()
        
        # TUM convention: T_w_curr = T_w_prev * inv(T_curr_prev)
        # PnP gives T_prev_curr (Cam in Prev). 
        T_w_curr_est = T_w_prev_est @ np.linalg.inv(est_pose)

        
        gt_traj.append(mat2pvec(gt_tum_pose[i]))
        est_traj.append(mat2pvec(T_w_curr_est))
        time_stamp.append(i)
        T_w_prev_est = T_w_curr_est.copy()
        if i % 500 == 0:
            print(f"\rIteration: {i}", end="", flush=True)


    est_traj_array = np.array(est_traj)
    time_stamp_array = np.array(time_stamp).reshape(-1, 1)
    if est_traj_array.shape[0] == time_stamp_array.shape[0]:
        combined_poses = np.hstack((time_stamp_array, est_traj_array))
        return combined_poses
    else:
        print("Warning: Estimated trajectory and time stamp have different number of rows (potentially some frames missed), only returning poses")
        return est_traj_array
