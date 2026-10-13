import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = 'data/'

# checkpoint directory
CHECKPOINT_DIR = 'checkpoints/'

# device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
#if torch.backends.mps.is_available(): device = 'mps'
os.environ['DEVICE'] = device

# task config
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 300,
    'state_dim': 8,
    'action_dim': 8,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front'],
}


# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'state_dim': 8,
    'action_dim': 8,
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['front'],
    'policy_class': 'ACT',
    'temporal_agg': False,
    # Diffusion Policy specific
    'diffusion_steps': 100,
    'ddim_steps': 10,
    'beta_schedule': 'cosine',
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 2000,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}

# observation noise for sim-to-real robustness (see training/noise.py).
# Each value is the magnitude at noise level 1.0; runs pick a level in [0, 1].
NOISE_CONFIG = {
    'image_noise_std': 0.1,        # std of Gaussian pixel noise, image in [0, 1]
    'image_brightness': 0.1,       # max brightness offset
    'image_contrast': 0.1,         # max contrast change (fraction)
    'joint_noise_std': 0.01,       # rad, Gaussian jitter on joint positions
}
