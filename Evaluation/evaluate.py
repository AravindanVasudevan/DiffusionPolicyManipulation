import argparse
from isaaclab.app import AppLauncher

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--object_name", type=str, default="object_cube_red")
parser.add_argument("--text", type=str, default="red cube")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--exec_horizon", type=int, default=4)
parser.add_argument("--max_episodes", type=int, default=3,
                    help="Stop after this many episodes finish (keeps the recording bounded).")
parser.add_argument("--no_record", action="store_true", help="Disable the demo video recording.")
parser.add_argument("--video_dir", type=str, default="Evaluation/videos",
                    help="Directory to write the recorded demo video into.")
parser.add_argument("--video_fps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from Simulator.env import PickPlaceEnvCfg
from Simulator.scene import make_record_camera
from Model.policy import DiffusionPolicy
from Dataset.dataset import IMAGENET_MEAN, IMAGENET_STD


def save_video(frames, path, fps):
    """Write a list of HxWx3 uint8 frames to an mp4 (falls back to a gif)."""

    if not frames:
        print("[evaluate] no frames captured, skipping video.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[evaluate] install `imageio[ffmpeg]` to save the demo video.")
        return
    try:
        imageio.mimsave(path, frames, fps=fps)
    except Exception as exc:
        gif_path = os.path.splitext(path)[0] + ".gif"
        print(f"[evaluate] mp4 write failed ({exc}); wrote {gif_path} instead.")
        imageio.mimsave(gif_path, frames, fps=fps)
        return
    print(f"[evaluate] saved demo video to {path}")


def preprocess_image(image_uint8: torch.Tensor) -> torch.Tensor:

    mean = torch.tensor(IMAGENET_MEAN, device=image_uint8.device)
    std = torch.tensor(IMAGENET_STD, device=image_uint8.device)
    image = image_uint8.float() / 255.0
    image = (image - mean) / std
    return image.permute(0, 3, 1, 2)


def main():

    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    record = not args_cli.no_record
    if record:
        cfg.scene.record_cam = make_record_camera(fps=args_cli.video_fps)

    env = ManagerBasedRLEnv(cfg=cfg)
    device = env.device
    frames = []

    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    model = DiffusionPolicy(action_dim=8, obs_horizon=2, action_horizon=8).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    action_min = torch.tensor(ckpt["action_min"], device=device)
    action_max = torch.tensor(ckpt["action_max"], device=device)

    obs, _ = env.reset()
    env.command_manager.get_term("pick_place_command").set_object_by_language(
        args_cli.text, args_cli.object_name
    )

    if record:
        origin = env.scene.env_origins[0]
        eye = origin + torch.tensor([1.6, 1.1, 1.1], device=device)
        target = origin + torch.tensor([0.5, 0.0, 0.05], device=device)
        env.scene["record_cam"].set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))

    obs_history = []
    episodes_done = 0

    def grab_frame():
        if not record:
            return
        rgb = env.scene["record_cam"].data.output["rgb"]
        if rgb is not None and rgb.shape[0] > 0:
            frames.append(rgb[0, ..., :3].to(torch.uint8).cpu().numpy())

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

        done = False
        for i in range(args_cli.exec_horizon):
            obs, reward, terminated, truncated, info = env.step(actions[:, i])
            grab_frame()
            if terminated.any() or truncated.any():
                done = True
                break

        if done:
            episodes_done += 1
            print(f"[evaluate] episode {episodes_done}/{args_cli.max_episodes} finished.")
            obs_history.clear()
            if episodes_done >= args_cli.max_episodes:
                break

    if record:
        save_video(frames, os.path.join(args_cli.video_dir, "eval_demo.mp4"), args_cli.video_fps)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    
    main()