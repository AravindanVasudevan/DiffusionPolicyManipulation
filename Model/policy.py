import torch
import torch.nn as nn

from .vision_encoder import build_resnet18_encoder
from .noise_pred_net import NoisePredNet
from .scheduler import DDPMScheduler


class DiffusionPolicy(nn.Module):

    def __init__(self, action_dim: int = 8, obs_horizon: int = 2, action_horizon: int = 8, num_train_timesteps: int = 100):

        super().__init__()
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        self.vision_encoder = build_resnet18_encoder(pretrained=True)
        vision_dim, proprio_dim, command_dim = 512, 9 + 9 + 3, 512
        cond_dim = (vision_dim + proprio_dim) * obs_horizon + command_dim

        self.noise_pred_net = NoisePredNet(action_dim=action_dim, cond_dim=cond_dim)
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    def to(self, *args, **kwargs):

        super().to(*args, **kwargs)
        self.scheduler.to(*args, **kwargs)

        return self

    def encode_obs(self, batch: dict) -> torch.Tensor:

        B, To = batch["image"].shape[:2]
        images = batch["image"].reshape(B * To, *batch["image"].shape[2:])
        vision_feat = self.vision_encoder(images).reshape(B, To, -1)

        proprio = torch.cat([batch["joint_pos"], batch["joint_vel"], batch["ee_pos"]], dim=-1)
        obs_feat = torch.cat([vision_feat, proprio], dim=-1).reshape(B, -1)

        return torch.cat([obs_feat, batch["command"]], dim=-1)

    def compute_loss(self, batch: dict) -> torch.Tensor:

        cond = self.encode_obs(batch)
        actions = batch["action"]
        noise = torch.randn_like(actions)
        t = torch.randint(0, self.scheduler.num_train_timesteps, (actions.shape[0],), device=actions.device)
        noisy_actions = self.scheduler.add_noise(actions, noise, t)
        pred_noise = self.noise_pred_net(noisy_actions, t, cond)

        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample_actions(self, batch: dict) -> torch.Tensor:

        cond = self.encode_obs(batch)
        B, device = cond.shape[0], cond.device
        sample = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        for t in reversed(range(self.scheduler.num_train_timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.noise_pred_net(sample, t_batch, cond)
            sample = self.scheduler.step(pred_noise, t, sample)

        return sample

    def forward(self, batch: dict) -> torch.Tensor:
        return self.compute_loss(batch)