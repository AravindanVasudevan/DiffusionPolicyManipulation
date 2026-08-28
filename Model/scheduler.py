import torch


class DDPMScheduler:

    def __init__(self, num_train_timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):

        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device):

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        shape = [-1] + [1] * (x0.dim() - 1)
        sqrt_alpha = self.alphas_cumprod[t].sqrt().view(*shape)
        sqrt_one_minus = (1 - self.alphas_cumprod[t]).sqrt().view(*shape)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def step(self, model_output: torch.Tensor, t: int, sample: torch.Tensor) -> torch.Tensor:
        
        alpha, alpha_cumprod, beta = self.alphas[t], self.alphas_cumprod[t], self.betas[t]
        mean = (1 / alpha.sqrt()) * (sample - (beta / (1 - alpha_cumprod).sqrt()) * model_output)
        if t == 0:
            return mean
        noise = torch.randn_like(sample)
        return mean + beta.sqrt() * noise