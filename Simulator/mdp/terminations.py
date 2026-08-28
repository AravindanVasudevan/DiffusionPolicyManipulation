from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _commanded_object_pos_w(env: ManagerBasedRLEnv, object_names: list[str], command_name: str) -> torch.Tensor:

    """World position of whichever object the per-episode command selects."""
    positions = torch.stack(
        [env.scene[name].data.root_pos_w for name in object_names], dim=1
    )
    object_idx = env.command_manager.get_term(command_name).object_idx

    return positions[torch.arange(env.num_envs, device=positions.device), object_idx]

def object_in_target_bin(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    command_name: str = "pick_place_command",
    bin_pos: tuple[float, float, float] = (0.6, 0.3, 0.0),
    xy_threshold: float = 0.08,
    z_threshold: float = 0.05,
) -> torch.Tensor:
    """True when the commanded object is within the target bin's footprint"""

    pos = _commanded_object_pos_w(env, object_names, command_name)

    bin_pos_t = torch.tensor(bin_pos, device=pos.device)
    xy_dist = torch.norm(pos[:, :2] - bin_pos_t[:2], dim=-1)
    z_dist = torch.abs(pos[:, 2] - bin_pos_t[2])

    return (xy_dist < xy_threshold) & (z_dist < z_threshold)

def object_dropped(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    command_name: str = "pick_place_command",
    minimum_height: float = -0.05,
) -> torch.Tensor:
    """True when the commanded object has fallen below the minimum height"""

    pos = _commanded_object_pos_w(env, object_names, command_name)
    return pos[:, 2] < minimum_height