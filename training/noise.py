"""Observation noise for sim-to-real robustness.

A single perturbation is shared by training-time augmentation (EpisodicDataset)
and evaluation-time corruption (scripts/eval_noise.py), so "noise level 0.3"
means the same thing in both places.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def perturb_observation(
        images: torch.Tensor,        # (num_cams, C, H, W) in [0, 1]
        qpos: torch.Tensor,          # (state_dim,) raw joint positions
        noise_config: dict,
        level: float,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt camera images and joint readings, scaled by `level` in [0, 1].

    - Gaussian pixel noise with std `image_noise_std * level`
    - 3x3 box blur, applied with probability `level` (out-of-focus / motion blur)
    - Per-camera brightness and contrast jitter up to `image_brightness * level`
      and `image_contrast * level`
    - Gaussian joint-encoder jitter with std `joint_noise_std * level` (rad)
    """
    if level <= 0.0:
        return images, qpos

    def rand(*shape):
        return torch.rand(*shape, generator=generator)

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    num_cams = images.shape[0]

    images = images + randn(*images.shape) * noise_config['image_noise_std'] * level

    if rand(1).item() < level:
        images = F.avg_pool2d(images, kernel_size=3, stride=1, padding=1,
                              count_include_pad=False)

    brightness = (rand(num_cams, 1, 1, 1) * 2 - 1) * noise_config['image_brightness'] * level
    contrast = 1 + (rand(num_cams, 1, 1, 1) * 2 - 1) * noise_config['image_contrast'] * level
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    images = (images - mean) * contrast + mean + brightness

    qpos = qpos + randn(*qpos.shape) * noise_config['joint_noise_std'] * level
    return images.clamp(0.0, 1.0), qpos


class ObservationNoise:
    """Apply `perturb_observation` to raw env observations during evaluation.

    Seeded so a noise sweep is reproducible.
    """

    def __init__(self, noise_config: dict, level: float, camera_names: list[str], seed: int = 0):
        self.noise_config = noise_config
        self.level = level
        self.camera_names = camera_names
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, obs: dict) -> dict:
        images = np.stack([obs['images'][cam] for cam in self.camera_names])  # (K, H, W, C)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        qpos = torch.from_numpy(obs['qpos']).float()

        images, qpos = perturb_observation(images, qpos, self.noise_config, self.level, self.generator)

        images = (images * 255.0).round().byte().permute(0, 2, 3, 1).numpy()
        return {
            **obs,
            'qpos': qpos.double().numpy(),
            'images': {cam: images[i] for i, cam in enumerate(self.camera_names)},
        }
