import os
import argparse
import numpy as np

from data.dataloader import TumRGBDDataset
from rgbd2poses import run_rgbd2poses
from poses2splats import poses2splats
from splats2floorplan import splats2floorplan
from eval.eval_ate import full_evaluate_ate


def main():
    parser = argparse.ArgumentParser(description="RGB-D → poses → splats → floorplan pipeline")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/irtiza/datasets/tum-rgbd/rgbd_dataset_freiburg1_room/",
        help="Path to TUM RGB-D dataset (default: %(default)s)",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory to store all results (default: %(default)s)",
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    dataset = TumRGBDDataset(dataset_path)
    rgb_list, depth_list, gt_tum_pose, dataset_params = dataset.load()

    # ---- Estimate poses ----
    poses = run_rgbd2poses(dataset_path, rgb_list, depth_list, gt_tum_pose, dataset_params)

    traj_path = os.path.join(results_dir, "trajectory.txt")
    np.savetxt(traj_path, poses)

    # If you already have GT trajectory saved as TUM txt in results_dir:
    gt_traj_path = os.path.join(results_dir, "gt_traj.txt")

    stats, gt_pts, est_aligned_pts = full_evaluate_ate(gt_traj_path, traj_path)
    print("\n=== ATE (TUM-style, after SE(3) alignment) ===")
    print(f"Number of matched pose pairs: {stats['num_pairs']}")
    print(f" RMSE:   {stats['rmse']:.6f}")

    # ---- Poses → splats ----
    poses2splats(
        poses[:, 1:],           
        dataset_params,
        dataset_path,
        depth_list,
        rgb_list,
        preview_idx=None,
    )

    # ---- Splats → floorplan ----
    splats_path = os.path.join(results_dir, "splats.json")
    floorplan_path = os.path.join(results_dir, "floorplan.png")

    splats2floorplan(
        splats_path,
        resolution=0.05,
        floor_percentile=2.0,
        out_floorplan=floorplan_path,
    )


if __name__ == "__main__":
    main()
