import torch.nn as nn
import torch



class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim: int):

        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:

        half = self.dim // 2
        freq = torch.exp(-torch.log(torch.tensor(10000.0, device=t.device)) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freq[None, :]

        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class FiLM(nn.Module):
    """Feature-wise linear modulation: cond -> per-channel scale + shift."""

    def __init__(self, cond_dim: int, channels: int):

        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:

        scale, shift = self.proj(cond).chunk(2, dim=-1)

        return x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

class ConditionalResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 5):

        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.film = FiLM(cond_dim, out_ch)
        self.act = nn.Mish()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:

        h = self.act(self.norm1(self.conv1(x)))
        h = self.film(h, cond)
        h = self.act(self.norm2(self.conv2(h)))

        return h + self.residual(x)

class NoisePredNet(nn.Module):
    """Simplified 1D conditional U-Net over the action-chunk sequence.
    Two down/up levels -- action_horizon must be divisible by 4."""

    def __init__(self, action_dim: int, cond_dim: int, time_dim: int = 128, base_channels: int = 64):

        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        total_cond_dim = cond_dim + time_dim
        ch1, ch2, ch3 = base_channels, base_channels * 2, base_channels * 4

        self.down1 = ConditionalResBlock1D(action_dim, ch1, total_cond_dim)
        self.down2 = ConditionalResBlock1D(ch1, ch2, total_cond_dim)
        self.pool = nn.AvgPool1d(2)
        self.mid = ConditionalResBlock1D(ch2, ch3, total_cond_dim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = ConditionalResBlock1D(ch3 + ch2, ch2, total_cond_dim)
        self.up2 = ConditionalResBlock1D(ch2 + ch1, ch1, total_cond_dim)
        self.out = nn.Conv1d(ch1, action_dim, 1)

    def forward(self, noisy_actions: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:

        x = noisy_actions.transpose(1, 2)
        c = torch.cat([cond, self.time_mlp(t)], dim=-1)

        h1 = self.down1(x, c)
        h2 = self.down2(self.pool(h1), c)
        h_mid = self.mid(self.pool(h2), c)

        u1 = self.up1(torch.cat([self.up(h_mid), h2], dim=1), c)
        u2 = self.up2(torch.cat([self.up(u1), h1], dim=1), c)

        return self.out(u2).transpose(1, 2)