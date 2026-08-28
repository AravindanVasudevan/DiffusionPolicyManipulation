from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_position(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:

    """End-effector position in the robot's base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    return ee_pos_w - robot.data.root_pos_w