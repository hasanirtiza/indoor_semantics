#!/usr/bin/env python3
import sys
import argparse
import numpy as np


def read_trajectory_tum(path):
    """
    Read a TUM trajectory file.
    Format per line (space-separated):
        timestamp tx ty tz qx qy qz qw
    Returns: dict[float] -> 4x4 pose matrix (T_w_cam)
    """
    traj = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            T = pose_from_tum(tx, ty, tz, qx, qy, qz, qw)
            traj[t] = T
    return traj


def pose_from_tum(tx, ty, tz, qx, qy, qz, qw):
    """Build 4x4 pose matrix from translation + quaternion (TUM convention)."""
    R = quat_to_rot(qx, qy, qz, qw)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def quat_to_rot(qx, qy, qz, qw):
    """Convert quaternion (qx, qy, qz, qw) to 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=float)
    q /= n
    x, y, z, w = q

    R = np.array([
        [1.0 - 2.0 * (y * y + z * z),  2.0 * (x * y - z * w),      2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),        1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),        2.0 * (y * z + x * w),      1.0 - 2.0 * (x * x + y * y)]
    ], dtype=float)
    return R


def associate_trajectories(traj_ref, traj_est, max_diff=0.02):
    """
    Associate two trajectories by timestamp.
    Returns list of (t_ref, t_est) pairs where |t_ref - t_est| < max_diff.
    """
    ref_times = sorted(traj_ref.keys())
    est_times = sorted(traj_est.keys())

    i, j = 0, 0
    matches = []

    while i < len(ref_times) and j < len(est_times):
        t_ref = ref_times[i]
        t_est = est_times[j]
        dt = t_est - t_ref

        if abs(dt) <= max_diff:
            matches.append((t_ref, t_est))
            i += 1
            j += 1
        elif dt > 0:
            i += 1
        else:
            j += 1

    return matches


def align(model, data):
    """
    Align two trajectories using Horn's method (closed-form SE(3)).

    Args:
        model: 3xN array (first trajectory, e.g. estimated)
        data:  3xN array (second trajectory, e.g. ground truth)

    Returns:
        rot  -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (N,) after alignment
    """
    # Zero-center
    model_mean = model.mean(axis=1, keepdims=True)
    data_mean = data.mean(axis=1, keepdims=True)
    model_zerocentered = model - model_mean
    data_zerocentered = data - data_mean

    # Cross-covariance
    W = np.zeros((3, 3), dtype=float)
    for k in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, k], data_zerocentered[:, k])

    # SVD
    U, _, Vt = np.linalg.svd(W.T)
    S = np.eye(3, dtype=float)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    rot = U @ S @ Vt
    trans = data_mean - rot @ model_mean  # (3,1)

    # Apply transform
    model_aligned = rot @ model + trans
    alignment_error = model_aligned - data
    trans_error = np.linalg.norm(alignment_error, axis=0)  # (N,)

    return rot, trans, trans_error



def full_evaluate_ate2(gt_traj_file_name, est_traj_file_name, max_association_diff=0.02):
    """
    Compute ATE RMSE between two TUM-format trajectories.

    gt_traj, est_traj: dict[timestamp] -> 4x4 pose

    Returns:
        stats: dict with rmse, mean, median, etc.
        gt_pts: (N,3) ground truth positions (matched order)
        est_aligned_pts: (N,3) estimated positions after SE(3) alignment
    """
    print(f"Loading ground truth from: {gt_traj_file_name}")
    gt_traj = read_trajectory_tum(gt_traj_file_name)
    print(f"  Loaded {len(gt_traj)} poses.")

    print(f"Loading estimate from:     {est_traj_file_name}")
    est_traj = read_trajectory_tum(est_traj_file_name)
    print(f"  Loaded {len(est_traj)} poses.")

    matches = associate_trajectories(gt_traj, est_traj, max_diff=max_association_diff)
    if len(matches) == 0:
        raise RuntimeError("No matching timestamps between trajectories (after association).")

    gt_pts = []
    est_pts = []

    for t_ref, t_est in matches:
        T_gt = gt_traj[t_ref]
        T_est = est_traj[t_est]
        gt_pts.append(T_gt[:3, 3])
        est_pts.append(T_est[:3, 3])

    gt_pts = np.array(gt_pts, dtype=float)   # (N,3)
    est_pts = np.array(est_pts, dtype=float) # (N,3)

    # Align estimated trajectory to ground truth
    rot, trans, trans_error = align(est_pts.T, gt_pts.T)  # pass as (3,N)
    est_aligned = (rot @ est_pts.T + trans).T            # (N,3)

    # TUM-style ATE RMSE
    ate_rmse = np.sqrt(np.mean(trans_error ** 2))

    stats = {
        "num_pairs": len(matches),
        "rmse": ate_rmse,
        "mean": float(np.mean(trans_error)),
        "median": float(np.median(trans_error)),
        "std": float(np.std(trans_error)),
        "min": float(np.min(trans_error)),
        "max": float(np.max(trans_error)),
    }

    return stats, gt_pts, est_aligned

def full_evaluate_ate(gt_traj_file_name, est_traj_file_name, max_association_diff=0.02):
    """
    Compute ATE RMSE between two TUM-format trajectories.

    gt_traj, est_traj: dict[timestamp] -> 4x4 pose

    Returns:
        stats: dict with rmse, mean, median, etc.
        gt_pts: (N,3) ground truth positions (matched order)
        est_aligned_pts: (N,3) estimated positions after SE(3) alignment
    """
    print(f"Loading ground truth from: {gt_traj_file_name}")
    gt_traj = read_trajectory_tum(gt_traj_file_name)
    print(f"  Loaded {len(gt_traj)} poses.")

    print(f"Loading estimate from:     {est_traj_file_name}")
    est_traj = read_trajectory_tum(est_traj_file_name)
    print(f"  Loaded {len(est_traj)} poses.")

    matches = associate_trajectories(gt_traj, est_traj, max_diff=max_association_diff)
    if len(matches) == 0:
        raise RuntimeError("No matching timestamps between trajectories (after association).")

    gt_pts = []
    est_pts = []

    for t_ref, t_est in matches:
        T_gt = gt_traj[t_ref]
        T_est = est_traj[t_est]
        gt_pts.append(T_gt[:3, 3])
        est_pts.append(T_est[:3, 3])

    gt_pts = np.array(gt_pts, dtype=float)   # (N,3)
    est_pts = np.array(est_pts, dtype=float) # (N,3)

    # Align estimated trajectory to ground truth
    rot, trans, trans_error = align(est_pts.T, gt_pts.T)  # pass as (3,N)
    est_aligned = (rot @ est_pts.T + trans).T            # (N,3)

    # TUM-style ATE RMSE
    ate_rmse = np.sqrt(np.mean(trans_error ** 2))

    stats = {
        "num_pairs": len(matches),
        "rmse": ate_rmse,
        "mean": float(np.mean(trans_error)),
        "median": float(np.median(trans_error)),
        "std": float(np.std(trans_error)),
        "min": float(np.min(trans_error)),
        "max": float(np.max(trans_error)),
    }

    return stats, gt_pts, est_aligned


def evaluate_ate(gt_traj, est_traj, max_association_diff=0.02):
    """
    Compute ATE RMSE between two TUM-format trajectories.

    gt_traj, est_traj: dict[timestamp] -> 4x4 pose

    Returns:
        stats: dict with rmse, mean, median, etc.
        gt_pts: (N,3) ground truth positions (matched order)
        est_aligned_pts: (N,3) estimated positions after SE(3) alignment
    """
    matches = associate_trajectories(gt_traj, est_traj, max_diff=max_association_diff)
    if len(matches) == 0:
        raise RuntimeError("No matching timestamps between trajectories (after association).")

    gt_pts = []
    est_pts = []

    for t_ref, t_est in matches:
        T_gt = gt_traj[t_ref]
        T_est = est_traj[t_est]
        gt_pts.append(T_gt[:3, 3])
        est_pts.append(T_est[:3, 3])

    gt_pts = np.array(gt_pts, dtype=float)   # (N,3)
    est_pts = np.array(est_pts, dtype=float) # (N,3)

    # Align estimated trajectory to ground truth
    rot, trans, trans_error = align(est_pts.T, gt_pts.T)  # pass as (3,N)
    est_aligned = (rot @ est_pts.T + trans).T            # (N,3)

    # TUM-style ATE RMSE
    ate_rmse = np.sqrt(np.mean(trans_error ** 2))

    stats = {
        "num_pairs": len(matches),
        "rmse": ate_rmse,
        "mean": float(np.mean(trans_error)),
        "median": float(np.median(trans_error)),
        "std": float(np.std(trans_error)),
        "min": float(np.min(trans_error)),
        "max": float(np.max(trans_error)),
    }

    return stats, gt_pts, est_aligned


def plot_trajectories(plot_path, gt_pts, est_aligned_pts):
    """
    Save a 2D plot of ground truth and aligned estimated trajectories.
    Uses x and y coordinates (columns 0 and 1).
    """
    import matplotlib.pyplot as plt

    gt_xy = gt_pts[:, :2]
    est_xy = est_aligned_pts[:, :2]

    plt.figure()
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", label="ground truth")
    plt.plot(est_xy[:, 0], est_xy[:, 1], "b-", label="estimated")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved trajectory plot to: {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ATE (TUM-style) between two trajectories."
    )
    parser.add_argument("gt_traj", type=str, help="Ground truth trajectory (TUM format)")
    parser.add_argument("est_traj", type=str, help="Estimated trajectory (TUM format)")
    parser.add_argument(
        "--max_diff",
        type=float,
        default=0.02,
        help="Max time difference for association (seconds, default: 0.02)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="If given, save a plot of GT vs aligned estimate to this path (e.g. res.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading ground truth from: {args.gt_traj}")
    gt_traj = read_trajectory_tum(args.gt_traj)
    print(f"  Loaded {len(gt_traj)} poses.")

    print(f"Loading estimate from:     {args.est_traj}")
    est_traj = read_trajectory_tum(args.est_traj)
    print(f"  Loaded {len(est_traj)} poses.")

    stats, gt_pts, est_aligned_pts = evaluate_ate(
        gt_traj, est_traj, max_association_diff=args.max_diff
    )

    print("\n=== ATE (TUM-style, after SE(3) alignment) ===")
    print(f"Number of matched pose pairs: {stats['num_pairs']}")
    print(f"RMSE:   {stats['rmse']:.6f}")
    print(f"Mean:   {stats['mean']:.6f}")
    print(f"Median: {stats['median']:.6f}")
    print(f"Std:    {stats['std']:.6f}")
    print(f"Min:    {stats['min']:.6f}")
    print(f"Max:    {stats['max']:.6f}")

    if args.plot is not None:
        plot_trajectories(args.plot, gt_pts, est_aligned_pts)


if __name__ == "__main__":
    main()
