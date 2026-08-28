from __future__ import anootations

from torch.utils.data import Dataset
import numpy as np
import h5py
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)



class PickPlaceDataset(Dataset):
    def __init__(self, hdf5_path: str, obs_horizon: int = 2, action_horizon: int = 8):

        self.file = h5py.File(hdf5_path, "r")
        self.demo_keys = list(self.file["data"].keys())
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        self.index = []
        for key in self.demo_keys:
            T = self.file[f"data/{key}/actions"].shape[0]
            for t in range(T):
                self.index.append((key, t))

        all_actions = np.concatenate([self.file[f"data/{k}/actions"][:] for k in self.demo_keys], axis=0)
        self.action_min = all_actions.min(axis=0)
        self.action_max = all_actions.max(axis=0)

    def __len__(self):

        return len(self.index)

    def __getitem__(self, idx):

        key, t = self.index[idx]
        T = self.file[f"data/{key}/actions"].shape[0]

        obs_idxs = [max(0, t - self.obs_horizon + 1 + i) for i in range(self.obs_horizon)]
        image = self.file[f"data/{key}/obs/image"][obs_idxs].astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (0, 3, 1, 2))

        joint_pos = self.file[f"data/{key}/obs/joint_pos"][obs_idxs]
        joint_vel = self.file[f"data/{key}/obs/joint_vel"][obs_idxs]
        ee_pos = self.file[f"data/{key}/obs/ee_pos"][obs_idxs]

        command = self.file[f"data/{key}/obs/command"][t]

        act_idxs = [min(T - 1, t + i) for i in range(self.action_horizon)]
        actions = self.file[f"data/{key}/actions"][act_idxs]
        actions_norm = 2 * (actions - self.action_min) / (self.action_max - self.action_min + 1e-8) - 1

        return {
            "image": torch.from_numpy(image).float(),
            "joint_pos": torch.from_numpy(joint_pos).float(),
            "joint_vel": torch.from_numpy(joint_vel).float(),
            "ee_pos": torch.from_numpy(ee_pos).float(),
            "command": torch.from_numpy(command).float(),
            "action": torch.from_numpy(actions_norm).float(),
        }