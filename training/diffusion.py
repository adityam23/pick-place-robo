"""DDPM / DDIM diffusion utilities for Diffusion Policy.

Reference: Ho et al. (2020), Song et al. (2021) (DDIM).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    """Linear β schedule (DDPM)."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine β schedule (Improved DDPM / Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, max=0.999)


def make_beta_schedule(name: str, timesteps: int) -> torch.Tensor:
    if name == 'linear':
        return linear_beta_schedule(timesteps)
    elif name == 'cosine':
        return cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown noise schedule: {name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extract coefficients at timestep t and reshape to broadcast with x_shape."""
    b = t.shape[0]
    out = a.to(t.device).gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# ---------------------------------------------------------------------------
# Forward diffusion (training)
# ---------------------------------------------------------------------------

def q_sample(x_start: torch.Tensor, t: torch.Tensor,
             noise: torch.Tensor,
             sqrt_alphas_cumprod: torch.Tensor,
             sqrt_one_minus_alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Forward process: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε."""
    return (extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


# ---------------------------------------------------------------------------
# Reverse diffusion (sampling)
# ---------------------------------------------------------------------------

def _predict_start_from_noise(x_t: torch.Tensor, t: torch.Tensor,
                               noise_pred: torch.Tensor,
                               sqrt_alphas_cumprod: torch.Tensor,
                               sqrt_one_minus_alphas_cumprod: torch.Tensor
                               ) -> torch.Tensor:
    """Given x_t and predicted noise ε, recover x_0 estimate."""
    return (extract(sqrt_alphas_cumprod ** (-1), t, x_t.shape) * x_t -
            extract(sqrt_alphas_cumprod ** (-1) * sqrt_one_minus_alphas_cumprod,
                    t, x_t.shape) * noise_pred)


@torch.no_grad()
def ddim_sample(
        model: nn.Module,
        obs_cond: torch.Tensor,
        chunk_size: int,
        action_dim: int,
        betas: torch.Tensor,
        timesteps: int = 100,
        ddim_steps: int = 10,
        eta: float = 0.0,  # 0 = DDIM, 1 = DDPM
        device: str = 'cpu',
    ) -> torch.Tensor:
    """DDIM sampling loop.

    Args:
        model: noise prediction network ε_θ(x_t, t, obs)
        obs_cond: (B, obs_dim) observation conditioning
        chunk_size: number of action steps to generate
        action_dim: action dimension per step
        betas: full β schedule of `timesteps`
        timesteps: total diffusion steps (training)
        ddim_steps: number of DDIM inference steps
        eta: DDIM stochasticity (0 = deterministic)

    Returns:
        action chunk (B, chunk_size, action_dim)
    """
    batch_size = obs_cond.shape[0]
    shape = (batch_size, chunk_size, action_dim)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    # Evenly spaced sub-sequence ending at the last training step, e.g. for
    # T=100, S=10: [9, 19, ..., 99]. Sampling walks it in reverse.
    step_ratio = timesteps // ddim_steps
    ddim_timesteps = torch.arange(1, ddim_steps + 1, device=device) * step_ratio - 1

    # Start from pure noise
    x_t = torch.randn(shape, device=device)

    for i in reversed(range(ddim_steps)):
        t = ddim_timesteps[i]
        t_batch = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
        noise_pred = model(x_t, t_batch, obs_cond)

        alpha_cumprod_t = alphas_cumprod[t]
        # The final step lands on the clean sample (alpha_cumprod = 1).
        alpha_cumprod_prev = (alphas_cumprod[ddim_timesteps[i - 1]] if i > 0
                              else torch.ones((), device=device))

        x0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        sigma_t = eta * torch.sqrt(
            (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_prev)
        )
        pred_dir = torch.sqrt(1 - alpha_cumprod_prev - sigma_t ** 2) * noise_pred
        x_t = torch.sqrt(alpha_cumprod_prev) * x0_pred + pred_dir
        if eta > 0:
            x_t = x_t + sigma_t * torch.randn(shape, device=device)

    return x_t


# ---------------------------------------------------------------------------
# Conditioned 1D Conv U-Net (noise prediction network)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResidualBlock1D(nn.Module):
    """1D Conv residual block with optional FiLM conditioning."""
    def __init__(self, in_channels: int, out_channels: int,
                 cond_dim: int = 0, kernel_size: int = 5,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)

        self.norm1 = nn.GroupNorm(min(32, in_channels // 8), in_channels) if in_channels >= 8 else nn.Identity()
        self.norm2 = nn.GroupNorm(min(32, out_channels // 8), out_channels) if out_channels >= 8 else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Mish()

        # Channel adapter for skip connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # FiLM conditioning (shift + scale)
        self.film = None
        if cond_dim > 0:
            self.film = nn.Sequential(
                nn.Linear(cond_dim, out_channels * 2),
                nn.Mish(),
                nn.Linear(out_channels * 2, out_channels * 2),
            )

        # Zero-init the last conv for stable training
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        residual = self.residual_conv(x)

        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # FiLM modulation
        if self.film is not None and cond is not None:
            film_out = self.film(cond)  # (B, out_channels * 2)
            scale, shift = film_out.chunk(2, dim=-1)
            scale = scale.unsqueeze(-1)  # (B, C, 1)
            shift = shift.unsqueeze(-1)  # (B, C, 1)
            h = h * (1 + scale) + shift

        return h + residual


class ConditionalUnet1D(nn.Module):
    """1D Conditional U-Net for action denoising.

    Input:  noisy action chunk (B, T, action_dim)
    Output: predicted noise (B, T, action_dim)

    Architecture:
    - Timestep embedding: sinusoidal → MLP
    - Observation conditioning: MLP projection
    - Combined condition: t_emb + obs_emb
    - 1D Conv encoder-decoder with residual FiLM blocks
    """
    def __init__(self,
                 action_dim: int = 8,
                 obs_dim: int = 512,
                 hidden_dim: int = 512,
                 num_res_blocks: int = 4,
                 kernel_size: int = 5,
                 dropout: float = 0.1,
                 time_dim: int = 256,
                 ):
        super().__init__()
        self.action_dim = action_dim

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Observation projection
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, time_dim),
        )

        # Combined condition dimension
        cond_dim = time_dim * 2  # time + obs

        # Input projection: (B, action_dim, T) after permute
        self.input_proj = nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1)

        # Down-sampling blocks
        self.down_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, hidden_dim, cond_dim=cond_dim,
                           kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_res_blocks)
        ])

        # Bottleneck (no downsampling since action chunks are small)
        self.mid_block = ResidualBlock1D(hidden_dim, hidden_dim, cond_dim=cond_dim,
                                         kernel_size=kernel_size, dropout=dropout)

        # Up-sampling blocks
        self.up_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim * 2, hidden_dim, cond_dim=cond_dim,
                           kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_res_blocks)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(32, hidden_dim // 8), hidden_dim) if hidden_dim >= 8 else nn.Identity(),
            nn.Mish(),
            nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1),
        )

        # Zero-init output
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                obs_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: noisy action chunk (B, T, action_dim)
            t: timestep indices (B,)
            obs_cond: observation features (B, obs_dim)

        Returns:
            predicted noise (B, T, action_dim)
        """
        # Time embedding
        t_emb = self.time_mlp(t)  # (B, time_dim)

        # Observation embedding
        obs_emb = self.obs_proj(obs_cond)  # (B, time_dim)

        # Combined condition
        cond = torch.cat([t_emb, obs_emb], dim=-1)  # (B, time_dim * 2)

        # Convert to (B, C, T) for 1D convolutions
        x = x.permute(0, 2, 1)  # (B, action_dim, T)
        x = self.input_proj(x)  # (B, hidden_dim, T)

        # Store skip connections
        skips = []

        # Down blocks
        for block in self.down_blocks:
            skips.append(x)
            x = block(x, cond)

        # Mid block
        x = self.mid_block(x, cond)

        # Up blocks with skip connections
        for block in self.up_blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)  # skip connection
            x = block(x, cond)

        # Output projection
        x = self.output_proj(x)  # (B, action_dim, T)
        x = x.permute(0, 2, 1)  # (B, T, action_dim)

        return x
