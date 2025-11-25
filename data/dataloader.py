import os
import numpy as np
from utils.utils import pvec2mat
    

class TumRGBDDataset:
    #Dataloader for the TUM RGB-D dataset. Loads RGB, depth, and pose streams, associates timestamps, and returns aligned (img, depth, pose) triplets.
    
    def __init__(self, dataset_path, max_dt_depth=0.08, max_dt_pose=0.08):
        self.dataset_path = dataset_path if dataset_path.endswith("/") else dataset_path + "/"
        self.max_dt_depth = max_dt_depth
        self.max_dt_pose = max_dt_pose

        # Will be filled by load()
        self.images = []
        self.depths = []
        self.poses = []
        self.info = {
            "camera_matrix": np.array([
                [517.3,   0.0, 318.6],
                [  0.0, 516.5, 255.3],
                [  0.0,   0.0,   1.0]
            ], dtype=np.float32),
            "edge": 60,
            "max_depth": 7.0,
            "min_depth" : 0.2,
                    }
        

    def load(self):
        img_data, depth_data, pose_data = self._read_raw_files()
        associations = self._associate_frames(
            img_data[:, 0].astype(np.float64),
            depth_data[:, 0].astype(np.float64),
            pose_data[:, 0].astype(np.float64)
        )

        self.images, self.depths, self.poses = self._filter_associated_data(
            img_data, depth_data, pose_data, associations
        )

        return self.images, self.depths, self.poses, self.info


    def _read_raw_files(self):
        img_data = np.loadtxt(self.dataset_path + "rgb.txt", delimiter=' ', dtype=np.str_)
        depth_data = np.loadtxt(self.dataset_path + "depth.txt", delimiter=' ', dtype=np.str_)
        pose_data = np.loadtxt(self.dataset_path + "groundtruth.txt", delimiter=' ', dtype=np.str_)
        return img_data, depth_data, pose_data

    def _associate_frames(self, t_img, t_depth, t_pose):
        associations = []
        t_depth = np.array(t_depth)
        t_pose = np.array(t_pose)

        for i, t in enumerate(t_img):

            # Depth association
            idx_d = np.searchsorted(t_depth, t)
            candidates_d = [c for c in (idx_d, idx_d-1) if 0 <= c < len(t_depth)]
            best_d, dt_d = min(((c, abs(t_depth[c] - t)) for c in candidates_d), key=lambda x: x[1])

            if dt_d > self.max_dt_depth:
                continue

            # Pose association
            idx_p = np.searchsorted(t_pose, t)
            candidates_p = [c for c in (idx_p, idx_p-1) if 0 <= c < len(t_pose)]
            best_p, dt_p = min(((c, abs(t_pose[c] - t)) for c in candidates_p), key=lambda x: x[1])

            if dt_p > self.max_dt_pose:
                continue

            associations.append((i, best_d, best_p))

        return associations

    def _filter_associated_data(self, img_data, depth_data, pose_data, associations):
        images, depths, poses = [], [], []

        for (i, j, k) in associations:
            images.append(img_data[i, 1])
            depths.append(depth_data[j, 1])

            # Convert pose vector (qw, qx, qy, qz, tx, ty, tz) → 4x4 matrix
            pose_mat = pvec2mat(pose_data[k, 1:].astype(np.float64))
            poses.append(pose_mat)

        return images, depths, poses

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.depths[idx], self.poses[idx]
