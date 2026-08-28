import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--object_name", type=str, default="object_cube_red")
parser.add_argument("--text", type=str, default="red cube")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--exec_horizon", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from Simulator.env import PickPlaceEnvCfg
from Model.policy import DiffusionPolicy
from Dataset.dataset import IMAGENET_MEAN, IMAGENET_STD


def preprocess_image(image_uint8: torch.Tensor) -> torch.Tensor:

    mean = torch.tensor(IMAGENET_MEAN, device=image_uint8.device)
    std = torch.tensor(IMAGENET_STD, device=image_uint8.device)
    image = image_uint8.float() / 255.0
    image = (image - mean) / std
    return image.permute(0, 3, 1, 2)


def main():

    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=cfg)
    device = env.device

    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    model = DiffusionPolicy(action_dim=8, obs_horizon=2, action_horizon=8).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    action_min = torch.tensor(ckpt["action_min"], device=device)
    action_max = torch.tensor(ckpt["action_max"], device=device)

    obs, _ = env.reset()
    env.command_manager.get_term("pick_place_command").set_object_by_language(
        args_cli.text, args_cli.object_name
    )

    obs_history = []

    while simulation_app.is_running():
        policy_obs = obs["policy"]
        obs_history.append(policy_obs)
        if len(obs_history) > model.obs_horizon:
            obs_history.pop(0)
        while len(obs_history) < model.obs_horizon:
            obs_history.insert(0, obs_history[0])

        batch = {
            "image": torch.stack([preprocess_image(o["image"]) for o in obs_history], dim=1),
            "joint_pos": torch.stack([o["joint_pos"] for o in obs_history], dim=1),
            "joint_vel": torch.stack([o["joint_vel"] for o in obs_history], dim=1),
            "ee_pos": torch.stack([o["ee_pos"] for o in obs_history], dim=1),
            "command": policy_obs["command"],
        }

        normalized_chunk = model.sample_actions(batch)
        actions = (normalized_chunk + 1) / 2 * (action_max - action_min) + action_min

        for i in range(args_cli.exec_horizon):
            obs, reward, terminated, truncated, info = env.step(actions[:, i])
            if terminated.any() or truncated.any():
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    
    main()