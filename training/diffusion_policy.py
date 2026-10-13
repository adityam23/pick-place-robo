"""Diffusion Policy for visuomotor policy learning.

Architecture: ResNet18 vision backbone + 1D Conv U-Net noise predictor.
Reference: Chi et al. (2023) — Diffusion Policy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from training.diffusion import (
    ConditionalUnet1D,
    make_beta_schedule,
    q_sample,
    ddim_sample,
)


class DiffusionPolicy(nn.Module):
    """Diffusion-based visuomotor policy.

    Training: forward() adds noise to action chunk, predicts noise, returns MSE loss.
    Inference: sample() runs DDIM denoising to generate an action chunk.
    """

    def __init__(self, args_override: dict):
        super().__init__()

        # --- config ---
        self.action_dim = args_override.get('action_dim', 8)
        self.state_dim = args_override.get('state_dim', 8)
        self.num_queries = args_override.get('num_queries', 16)       # action chunk size
        self.hidden_dim = args_override.get('hidden_dim', 512)
        self.diffusion_steps = args_override.get('diffusion_steps', 100)
        self.ddim_steps = args_override.get('ddim_steps', 10)
        self.beta_schedule = args_override.get('beta_schedule', 'cosine')
        self.lr = args_override.get('lr', 1e-4)
        self.lr_backbone = args_override.get('lr_backbone', 1e-5)
        self.camera_names = args_override.get('camera_names', ['front'])
        self.backbone_name = args_override.get('backbone', 'resnet18')

        # --- vision backbone (ResNet18) ---
        from detr_vendored.models.backbone import build_backbone
        # Build a minimal args namespace for backbone construction
        class AttrDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name)
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        backbone_args = AttrDict(args_override)
        backbone_args['masks'] = False
        backbone_args['position_embedding'] = 'sine'

        self.vision_encoder = build_backbone(backbone_args)
        self.obs_dim = self.vision_encoder.num_channels  # 512 for ResNet18

        # Projector from backbone features to fixed-dim observation vector
        self.obs_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.obs_fc = nn.Sequential(
            nn.Linear(self.obs_dim, self.obs_dim),
            nn.Mish(),
            nn.Linear(self.obs_dim, self.obs_dim),
        )

        # Project joint state
        self.state_proj = nn.Linear(self.state_dim, self.hidden_dim)
        self.state_fc = nn.Mish()

        # Combined observation dimension (vision + proprio)
        combined_obs_dim = self.obs_dim + self.hidden_dim

        # --- noise prediction network ---
        self.noise_pred_net = ConditionalUnet1D(
            action_dim=self.action_dim,
            obs_dim=combined_obs_dim,
            hidden_dim=self.hidden_dim,
            num_res_blocks=4,
            kernel_size=5,
            dropout=0.1,
            time_dim=256,
        )

        # --- diffusion schedule ---
        self.betas = make_beta_schedule(self.beta_schedule, self.diffusion_steps)
        alphas = 1.0 - self.betas
        self.sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(alphas, dim=0))
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - torch.cumprod(alphas, dim=0))

        # --- optimizer ---
        self._optimizer = None

        print(f'DiffusionPolicy: backbone={self.backbone_name}, '
              f'chunk={self.num_queries}, '
              f'diffusion_steps={self.diffusion_steps}, '
              f'ddim_steps={self.ddim_steps}, '
              f'obs_dim={self.obs_dim}, '
              f'noise_params={sum(p.numel() for p in self.noise_pred_net.parameters()):,}')

    def _encode_obs(self, image: torch.Tensor, qpos: torch.Tensor
                    ) -> torch.Tensor:
        """Encode image + joint state into a single observation vector.

        Args:
            image: (B, num_cam, C, H, W)
            qpos: (B, state_dim)

        Returns:
            obs_vec: (B, obs_dim + hidden_dim)
        """
        B = image.shape[0]

        # Vision features
        # Use first camera (HARDCODED like original ACT)
        img_features, _ = self.vision_encoder(image[:, 0])
        # img_features is a list; take last layer
        feat = img_features[0]  # (B, C, H', W')
        feat = self.obs_pool(feat).flatten(1)  # (B, C)
        feat = self.obs_fc(feat)  # (B, obs_dim)

        # Joint state features
        state_feat = self.state_fc(self.state_proj(qpos))  # (B, hidden_dim)

        return torch.cat([feat, state_feat], dim=-1)

    def forward(self, qpos: torch.Tensor, image: torch.Tensor,
                actions: torch.Tensor = None, is_pad: torch.Tensor = None
                ) -> dict:
        """Training forward pass: compute denoising MSE loss.

        Args:
            qpos: (B, state_dim)
            image: (B, num_cam, C, H, W)
            actions: (B, chunk_size, action_dim) — target action chunk
            is_pad: (B, chunk_size) — padding mask

        Returns:
            loss_dict with 'loss' key
        """
        # Normalize image
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)

        if actions is not None:
            # Training: denoising objective
            B = actions.shape[0]
            actions = actions[:, :self.num_queries]

            # Encode observation
            device = qpos.device
            obs_cond = self._encode_obs(image, qpos)

            # Sample timesteps
            t = torch.randint(0, self.diffusion_steps, (B,), device=device).long()

            # Move schedule to device
            sqrt_alpha_cumprod = self.sqrt_alphas_cumprod.to(device)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)

            # Add noise
            noise = torch.randn_like(actions)
            x_noisy = q_sample(actions, t, noise,
                              sqrt_alpha_cumprod, sqrt_one_minus_alpha)

            # Predict noise
            noise_pred = self.noise_pred_net(x_noisy, t, obs_cond)

            # Loss (mask out padded regions)
            if is_pad is not None:
                is_pad = is_pad[:, :self.num_queries]
                mask = (~is_pad).unsqueeze(-1).expand_as(noise)
                loss = F.mse_loss(noise_pred[mask], noise[mask])
            else:
                loss = F.mse_loss(noise_pred, noise)

            loss_dict = {'loss': loss}
            return loss_dict
        else:
            # Inference: run DDIM sampling
            return self.sample(qpos, image)

    @torch.no_grad()
    def sample(self, qpos: torch.Tensor, image: torch.Tensor
               ) -> torch.Tensor:
        """Sample action chunk via DDIM.

        Args:
            qpos: (B, state_dim)
            image: (B, num_cam, C, H, W)

        Returns:
            action chunk (B, num_queries, action_dim)
        """
        # Normalize image
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        device = qpos.device

        obs_cond = self._encode_obs(image, qpos)
        betas = self.betas.to(device)

        action = ddim_sample(
            model=self.noise_pred_net,
            obs_cond=obs_cond,
            chunk_size=self.num_queries,
            action_dim=self.action_dim,
            betas=betas,
            timesteps=self.diffusion_steps,
            ddim_steps=self.ddim_steps,
            eta=0.0,  # deterministic DDIM
            device=device,
        )

        return action

    def configure_optimizers(self):
        """Return optimizer (lazy init)."""
        if self._optimizer is None:
            param_dicts = [
                {"params": [p for n, p in self.named_parameters()
                           if "vision_encoder" not in n and p.requires_grad]},
                {"params": [p for n, p in self.named_parameters()
                           if "vision_encoder" in n and p.requires_grad],
                 "lr": self.lr_backbone},
            ]
            self._optimizer = torch.optim.AdamW(
                param_dicts, lr=self.lr, weight_decay=1e-4)

        return self._optimizer
