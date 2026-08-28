from __future__ import annotations

import random
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .text_encoder import embed_text


TEXT_TEMPLATES = {
    "object_cube_red": ["red cube"],
    "object_cube_green": ["green cube"],
    "object_cube_blue": ["blue cube"],
    "object_sphere_red": ["red sphere"],
    "object_sphere_green": ["green sphere"],
    "object_sphere_blue": ["blue sphere"],
    "object_cylinder_red": ["red cylinder"],
    "object_cylinder_green": ["green cylinder"],
    "object_cylinder_blue": ["blue cylinder"],
}


class PickPlaceCommand(CommandTerm):

    cfg: PickPlaceCommandCfg

    def __init__(self, cfg: PickPlaceCommandCfg, env: ManagerBasedRLEnv):

        super().__init__(cfg, env)
        self.object_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.text_embedding = torch.zeros(self.num_envs, 512, device=self.device)

    @property
    def command(self):

        return self.text_embedding

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):

        idx = torch.randint(len(self.cfg.object_names), (len(env_ids),), device=self.device)
        self.object_idx[env_ids] = idx
        texts = [random.choice(TEXT_TEMPLATES[self.cfg.object_names[i]]) for i in idx.tolist()]
        self.text_embedding[env_ids] = embed_text(texts, self.device)

    def _update_command(self):
        pass

    def set_object_by_language(self, text: str, object_name: str, env_ids=None):

        env_ids = env_ids if env_ids is not None else slice(None)
        self.object_idx[env_ids] = self.cfg.object_names.index(object_name)
        self.text_embedding[env_ids] = embed_text([text], self.device)

@configclass
class PickPlaceCommandCfg(CommandTermCfg):

    class_type: type = PickPlaceCommand
    object_names: list[str] = None
    resampling_time_range: tuple[float, float] = (1e9, 1e9)