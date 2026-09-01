from __future__ import annotations

from torch.utils.data import Dataset
import numpy as np
import h5py
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)



class PickPlaceDataset(Dataset):

    def __init__(self, hdf5_path: str, obs_horizon: int = 2, action_horizon: int = 8):

        self.hdf5_path = hdf5_path
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        self.demos = {}
        with h5py.File(hdf5_path, "r") as f:
            self.demo_keys = list(f["data"].keys())
            for key in self.demo_keys:
                g = f[f"data/{key}"]
                self.demos[key] = {
                    "image": g["obs/image"][:],
                    "joint_pos": g["obs/joint_pos"][:].astype(np.float32),
                    "joint_vel": g["obs/joint_vel"][:].astype(np.float32),
                    "ee_pos": g["obs/ee_pos"][:].astype(np.float32),
                    "command": g["obs/command"][:].astype(np.float32),
                    "actions": g["actions"][:].astype(np.float32),
                }

        self.index = []
        for key in self.demo_keys:
            for t in range(self.demos[key]["actions"].shape[0]):
                self.index.append((key, t))

        all_actions = np.concatenate([self.demos[k]["actions"] for k in self.demo_keys], axis=0)
        self.action_min = all_actions.min(axis=0)
        self.action_max = all_actions.max(axis=0)

    def __len__(self):

        return len(self.index)

    def __getitem__(self, idx):

        key, t = self.index[idx]
        demo = self.demos[key]
        T = demo["actions"].shape[0]

        obs_idxs = [max(0, t - self.obs_horizon + 1 + i) for i in range(self.obs_horizon)]
        image = demo["image"][obs_idxs].astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (0, 3, 1, 2))

        joint_pos = demo["joint_pos"][obs_idxs]
        joint_vel = demo["joint_vel"][obs_idxs]
        ee_pos = demo["ee_pos"][obs_idxs]

        command = demo["command"][t]

        act_idxs = [min(T - 1, t + i) for i in range(self.action_horizon)]
        actions = demo["actions"][act_idxs]
        actions_norm = 2 * (actions - self.action_min) / (self.action_max - self.action_min + 1e-8) - 1

        return {
            "image": torch.from_numpy(image).float(),
            "joint_pos": torch.from_numpy(joint_pos).float(),
            "joint_vel": torch.from_numpy(joint_vel).float(),
            "ee_pos": torch.from_numpy(ee_pos).float(),
            "command": torch.from_numpy(command).float(),
            "action": torch.from_numpy(actions_norm).float(),
        }