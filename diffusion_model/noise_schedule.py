import torch
import numpy as np

class PredefinedNoiseScheduleDiscrete:
    """
    A predefined noise schedule for discrete diffusion models.
    """
    def __init__(self, noise_schedule, timesteps):
        self.timesteps = timesteps
        
        if noise_schedule == 'linear':
            self.beta = torch.linspace(1e-4, 0.02, timesteps)
        elif noise_schedule == 'cosine':
            steps = torch.linspace(0, timesteps, timesteps + 1)
            alpha_bar = torch.cos(((steps / timesteps + 0.008) / 1.008) * torch.pi * 0.5) ** 2
            self.beta = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
            
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
    def get_alpha_bar(self, t_normalized):
        """Get alpha_bar for a given normalized timestep."""
        # Move t to CPU for indexing
        t = torch.clamp(t_normalized * self.timesteps, 0, self.timesteps - 1).long().cpu()
        # Get alpha_bar and move to same device as input
        alpha_bar_t = self.alpha_bar[t].to(t_normalized.device)
        return alpha_bar_t 