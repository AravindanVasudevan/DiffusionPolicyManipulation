from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_object_pos_on_table(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_names: list[str],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    min_separation: float = 0.09,
    max_tries: int = 50,
):
    """Reseting the objects pos without overlapping."""

    assets = [env.scene[name] for name in object_names]
    device = assets[0].device
    num_envs = len(env_ids)
    num_objects = len(assets)

    sampled_xy = torch.zeros(num_envs, num_objects, 2, device=device)

    for i in range(num_envs):
        placed = []
        for k in range(num_objects):
            for _ in range(max_tries):
                xy = torch.empty(2, device=device)
                xy[0].uniform_(*x_range)
                xy[1].uniform_(*y_range)
                if all(torch.norm(xy - p) >= min_separation for p in placed):
                    break
            placed.append(xy)
            sampled_xy[i, k] = xy

    origins_xy = env.scene.env_origins[env_ids, 0:2]

    for k, asset in enumerate(assets):
        root_states = asset.data.default_root_state[env_ids].clone()
        root_states[:, 0:2] = sampled_xy[:, k] + origins_xy
        root_states[:, 7:13] = 0.0
        asset.write_root_pose_to_sim(root_states[:, 0:7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)