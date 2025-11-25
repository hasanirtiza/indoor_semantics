import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import os

def match_nearest(img_timestamps, depth_timestamps, max_dt=0.05):
    # img_timestamps, depth_timestamps: 1D arrays (seconds)
    depth_idx = np.searchsorted(depth_timestamps, img_timestamps)
    pairs = []
    for i, t in enumerate(img_timestamps):
        j = depth_idx[i]
        candidates = []
        if j < len(depth_timestamps):
            candidates.append(j)
        if j-1 >= 0:
            candidates.append(j-1)
        best = None
        best_dt = float('inf')
        for c in candidates:
            dt = abs(depth_timestamps[c] - t)
            if dt < best_dt:
                best_dt = dt
                best = c
        if best is not None and best_dt <= max_dt:
            pairs.append((i, best, best_dt))
        else:
            print(f"Image at index {i:.3f} does not have a match")
            pairs.append((i, None, None))  
    return pairs


def associate_frames(tstamp_image, tstamp_depth, tstamp_pose=None, max_dt_depth=0.08, max_dt_pose=0.08):
    # associate frames
    associations = []
    tstamp_depth = np.array(tstamp_depth)
    if tstamp_pose is not None:
        tstamp_pose = np.array(tstamp_pose)

    for i, t in enumerate(tstamp_image):
        idx_d = np.searchsorted(tstamp_depth, t)
        candidates_d = []
        if idx_d < len(tstamp_depth): candidates_d.append(idx_d)
        if idx_d-1 >= 0: candidates_d.append(idx_d-1)
        best_d, dt_d = min(((c, abs(tstamp_depth[c]-t)) for c in candidates_d), key=lambda x: x[1])
        if dt_d > max_dt_depth:
            continue  # skip if too far
        
        if tstamp_pose is None:
            associations.append((i, best_d))
        else:
            # nearest pose (or interpolate)
            idx_p = np.searchsorted(tstamp_pose, t)
            candidates_p = []
            if idx_p < len(tstamp_pose): candidates_p.append(idx_p)
            if idx_p-1 >= 0: candidates_p.append(idx_p-1)
            best_p, dt_p = min(((c, abs(tstamp_pose[c]-t)) for c in candidates_p), key=lambda x: x[1])
            if dt_p > max_dt_pose:
                continue
            associations.append((i, best_d, best_p))
    return associations

def pvec2mat(pose_vec):
   # tum format -> 4x4 pose matrix
    pose_vec = np.asarray(pose_vec, dtype=np.float64)
    t = pose_vec[:3]              # translation
    q = pose_vec[3:]              # quaternion [qx, qy, qz, qw]
    
    R_mat = R.from_quat(q).as_matrix()
    
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

def mat2pvec(pose_mat):
   # 4x4 format -> tum
    pose_mat = np.asarray(pose_mat, dtype=np.float64)
    
    t = pose_mat[:3, 3]
    
    R_mat = pose_mat[:3, :3]
    
    q = R.from_matrix(R_mat).as_quat()
    
    pose_vec = np.hstack([t, q])
    return pose_vec

def split_pose(pose_mat):
    pose_mat = np.asarray(pose_mat, dtype=np.float64)
    return pose_mat[:3, 3], pose_mat[:3, :3]

def filter_associated_data(img_data, depth_data, pose_data, associations):
    # return only valid associations
    images, depths, poses = [], [], []
    inv_first_pose = None

    for assoc in associations:
        if len(assoc) == 2:  # image + depth only
            i, j = assoc
            images.append(img_data[i, 1])
            depths.append(depth_data[j, 1])
        else:  # image + depth + pose
            i, j, k = assoc

            images.append(img_data[i, :])
            depths.append(depth_data[j, :])
            this_pose = pvec2mat(pose_data[k, 1:].astype(np.float64)) # lets convet pose here it is cumbersome
            #if inv_first_pose is None:
            #    inv_first_pose = np.linalg.inv(this_pose)
            #    this_pose = np.eye(4)
            #else:
            #    this_pose = inv_first_pose @ this_pose
            poses.append(this_pose)  #
    
    return images, depths, poses


def bilinear_depth(depth, x, y):
    # depth imp. using bil
    H, W = depth.shape
    if x < 0 or y < 0 or x >= W-1 or y >= H-1: 
        return None
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    dx, dy = x - x0, y - y0
    v00 = depth[y0,   x0]
    v10 = depth[y0,   x0+1]
    v01 = depth[y0+1, x0]
    v11 = depth[y0+1, x0+1]
    if not np.isfinite(v00) or not np.isfinite(v10) or not np.isfinite(v01) or not np.isfinite(v11):
        return None
    return (v00*(1-dx)*(1-dy) + v10*dx*(1-dy) + v01*(1-dx)*dy + v11*dx*dy)

def depth_to_3d_bil(pt, depth, K, min_depth=0.1, max_depth=60.0):
     # depth imp. using bil
    u, v = pt
    H, W = depth.shape
    
    # Check bounds
    if u < 0 or v < 0 or u >= W-1 or v >= H-1:
        return None
    
    # Floor and ceil
    u0, v0 = int(np.floor(u)), int(np.floor(v))
    u1, v1 = u0 + 1, v0 + 1
    du, dv = u - u0, v - v0
    
    # Fetch depth at 4 neighbors
    z00, z10 = depth[v0, u0], depth[v0, u1]
    z01, z11 = depth[v1, u0], depth[v1, u1]
    
    # Check for invalid depth
    if not (np.isfinite(z00) and np.isfinite(z10) and np.isfinite(z01) and np.isfinite(z11)):
        return None
    
    # Bilinear interpolation
    z = (1-du)*(1-dv)*z00 + du*(1-dv)*z10 + (1-du)*dv*z01 + du*dv*z11
    
    if z <= 0 or z < min_depth or z > max_depth:
        return None
    
    # Backproject to 3D
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.array([x, y, z], dtype=np.float32)



def pose_matrix_to_init_pose(pose_matrix):
   
    assert pose_matrix.shape == (4, 4)

    # Extract rotation (3x3) and translation (3,)
    R_mat = pose_matrix[:3, :3]
    t_vec = pose_matrix[:3, 3]

    # Convert rotation matrix to quaternion (scipy returns [x, y, z, w])
    quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]

    # Reorder to poseLib convention: [w, x, y, z]
    q_init = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)

    # Translation vector
    t_init = t_vec.astype(float)

    return {"q": q_init, "t": t_init}



def depth_to_3d(pt, depth, K):
   
    u, v = int(pt[0]), int(pt[1])
    z = depth[v, u]
    if z == 0:  # skip missing depth
        return None
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1] 
    return np.array([x, y, z], dtype=np.float32)

def unproject_depth_to_3d(pt_3d, K):
   
    x, y, z = pt_3d
    if z <= 0:
        return None  # invalid depth
    u = (x * K[0, 0] / z) + K[0, 2]
    v = (y * K[1, 1] / z) + K[1, 2]
    return np.array([u, v], dtype=np.float32)


def project_3d_to_2d(points_3d, K, R=np.eye(3), t=np.zeros((3, 1))):

    points_3d = np.asarray(points_3d, dtype=np.float32)
    if points_3d.ndim == 1:
        points_3d = points_3d[np.newaxis, :]

    # Transform points into camera frame
    points_cam = (R @ points_3d.T + t).T

    # Avoid division by zero
    valid = points_cam[:, 2] > 1e-6
    points_cam = points_cam[valid]

    # Perspective projection
    uv = (K @ points_cam.T).T
    uv[:, 0] /= uv[:, 2]
    uv[:, 1] /= uv[:, 2]

    return uv[:, :2]




def sample_world_points_from_depth(depth, rgb, K, T_w_cam,
                                   max_points_per_frame=2000):
  
    H, W = depth.shape

    # valid depth mask
    mask = depth > 0
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    v_idx, u_idx = np.where(mask)
    num_valid = len(u_idx)

    # random subsample if too many
    N = min(max_points_per_frame, num_valid)
    sel = np.random.choice(num_valid, size=N, replace=False)
    u = u_idx[sel].astype(np.float32)
    v = v_idx[sel].astype(np.float32)

    z = depth[v_idx[sel], u_idx[sel]].astype(np.float32)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Backproject to camera coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts_cam = np.stack([x, y, z], axis=-1)  # (N,3)

    # cam -> world
    R = T_w_cam[:3, :3]
    t = T_w_cam[:3, 3]
    pts_world = (R @ pts_cam.T + t.reshape(3, 1)).T  # (N,3)

    # colors: cv2 gives BGR; convert to RGB for nicer output
    cols_bgr = rgb[v_idx[sel], u_idx[sel], :]      # (N,3)
    cols_rgb = cols_bgr[:, ::-1]                   # BGR -> RGB

    return pts_world.astype(np.float32), cols_rgb.astype(np.uint8)


def rotation_matrix_to_axis_angle(rot_matrix): 
    # Compute the eigenvalues and eigenvectors of the rotation matrix
    eigenvalues, eigenvectors = np.linalg.eig(rot_matrix)
    
    # Find the eigenvector corresponding to the eigenvalue of 1
    axis = eigenvectors[:, np.isclose(eigenvalues, 1.0)]
    
    # Ensure the axis is a unit vector
    axis = axis.ravel() / np.linalg.norm(axis)
    
    # Compute the angle of rotation
    angle = np.arccos((np.trace(rot_matrix) - 1) / 2)

    return axis, np.degrees(angle)


def rotation_matrix_to_axis_angle_stable(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    # If angle is very small, axis doesn't matter much
    if angle < 1e-6:
        return np.array([1, 0, 0]), 0.0

    # Standard axis formula
    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ])
    axis = axis / (2 * np.sin(angle))

    return axis, np.degrees(angle)



def save_poses_to_txt(timestamps, poses, filename):
   

    if len(timestamps) != len(poses):
        raise ValueError("timestamps and poses must have the same length")

    with open(filename, 'w') as f:
        for t, p in zip(timestamps, poses):
            if len(p) != 7:
                raise ValueError("Each pose must contain 7 values: tx ty tz qx qy qz qw")

            line = f"{t} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]}\n"
            f.write(line)




def quat_slerp(q0, q1, alpha):

    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)

    # if quats are opposite, flip one to take the shorter path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # if very close, lerp
    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)

def R_to_quat(R):
  
    m = R
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    else:
        if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)

def quat_to_R(q):
  
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx + yy)]
    ], dtype=np.float64)
    return R



def merge_pose_files(file_trans, file_quat, file_out="merged.txt"):
  

    with open(file_trans, 'r') as f1, open(file_quat, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    if len(lines1) != len(lines2):
        print("Warning: the two files have different number of lines. Will merge up to the shortest length.")

    merged_lines = []
    for l1, l2 in zip(lines1, lines2):
        p1 = l1.strip().split()
        p2 = l2.strip().split()

        # Extract components
        timestamp = p1[0]
        tx, ty, tz = p1[1], p1[2], p1[3]
        qx, qy, qz, qw = p2[4], p2[5], p2[6], p2[7]

        merged_lines.append(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

    # Write output file
    with open(file_out, "w") as fout:
        fout.writelines(merged_lines)

    print(f"Merged file written to {file_out}")




def read_al_feats(image_name, foler_path):
    kpts_file = os.path.join(foler_path, os.path.basename(image_name) +  ".c0.s0.f9.kp.bin")
    desc_file = os.path.join(foler_path, os.path.basename(image_name) +  ".c0.s0.f9.bin")
    al_keypoints = np.fromfile(kpts_file, dtype=np.float32)
    al_keypoints = al_keypoints.reshape(-1, 2)
    keypoints = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in al_keypoints]


    descriptors = np.fromfile(desc_file, dtype=np.uint8)
    descriptors = descriptors.reshape(-1, 128)


    return keypoints, descriptors, al_keypoints
