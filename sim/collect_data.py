"""Collect demonstration episodes using the scripted expert and save to HDF5."""

import os
import sys
import h5py
import argparse
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import TASK_CONFIG
from sim.env import PandaPickPlaceEnv
from sim.scripted_expert import run_expert_episode


def collect_episodes(num_episodes, task_name, max_steps=None, verbose=False,
                     render=False):
    """Collect demonstration episodes and save to HDF5 files.

    Args:
        num_episodes: number of episodes to collect
        task_name: task name (used for data directory)
        max_steps: max timesteps per episode (default from config)
        verbose: print per-episode info
        render: open MuJoCo GUI viewer to watch collection live
    """
    cfg = TASK_CONFIG
    if max_steps is None:
        max_steps = cfg['episode_len']

    data_dir = os.path.join(cfg['dataset_dir'], task_name)
    os.makedirs(data_dir, exist_ok=True)

    # Count existing episodes
    existing = len([f for f in os.listdir(data_dir) if f.endswith('.hdf5')])
    print(f"Data directory: {data_dir}")
    print(f"Existing episodes: {existing}")
    print(f"Collecting {num_episodes} new episodes (starting at index {existing})")

    env = PandaPickPlaceEnv(
        cam_width=cfg['cam_width'],
        cam_height=cfg['cam_height'],
        render_gui=render,
    )

    n_success = 0
    episode_idx = existing

    pbar = tqdm(total=num_episodes, desc="Collecting episodes")
    attempts = 0

    while n_success < num_episodes:
        attempts += 1
        seed = existing + attempts + 1000  # offset to avoid collisions

        obs_list, action_list, success = run_expert_episode(
            env, max_steps=max_steps, seed=seed, verbose=False
        )

        if not success:
            if verbose:
                print(f"  Attempt {attempts}: FAILED (discarding)")
            continue

        # Save successful episode
        save_episode(
            data_dir, episode_idx, obs_list, action_list, cfg
        )

        n_success += 1
        episode_idx += 1
        pbar.update(1)

        if verbose:
            cube_pos = env.get_cube_pos()
            target_pos = env.get_target_pos()
            dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
            print(f"  Episode {episode_idx-1}: SUCCESS (dist={dist:.3f})")

    pbar.close()
    env.close()

    print(f"\nCollection complete!")
    print(f"  Episodes collected: {n_success}")
    print(f"  Total attempts: {attempts}")
    print(f"  Success rate: {n_success/attempts:.0%}")
    print(f"  Saved to: {data_dir}")


def save_episode(data_dir, episode_idx, obs_list, action_list, cfg):
    """Save a single episode to HDF5 in ACT format.

    HDF5 structure (matching training/utils.py EpisodicDataset):
        attrs['sim'] = True
        /observations/qpos          (T, state_dim)      float64
        /observations/qvel          (T, state_dim)      float64
        /observations/images/front  (T, H, W, 3)        uint8
        /action                     (T, action_dim)      float64
    """
    T = len(obs_list)
    state_dim = cfg['state_dim']
    action_dim = cfg['action_dim']
    cam_h = cfg['cam_height']
    cam_w = cfg['cam_width']

    dataset_path = os.path.join(data_dir, f'episode_{episode_idx}.hdf5')

    with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2 * 2) as root:
        root.attrs['sim'] = True

        obs_grp = root.create_group('observations')
        img_grp = obs_grp.create_group('images')

        # Create datasets
        qpos_ds = obs_grp.create_dataset(
            'qpos', (T, state_dim), dtype='float64'
        )
        qvel_ds = obs_grp.create_dataset(
            'qvel', (T, state_dim), dtype='float64'
        )
        action_ds = root.create_dataset(
            'action', (T, action_dim), dtype='float64'
        )

        # Image datasets with chunking for compression
        for cam_name in cfg['camera_names']:
            img_grp.create_dataset(
                cam_name, (T, cam_h, cam_w, 3), dtype='uint8',
                chunks=(1, cam_h, cam_w, 3),
            )

        # Write data
        for t, (obs, action) in enumerate(zip(obs_list, action_list)):
            qpos_ds[t] = obs['qpos']
            qvel_ds[t] = obs['qvel']
            action_ds[t] = action
            for cam_name in cfg['camera_names']:
                img_grp[cam_name][t] = obs['images'][cam_name]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect demonstration episodes')
    parser.add_argument('--task', type=str, default='pick_place',
                       help='Task name for data directory')
    parser.add_argument('--num_episodes', type=int, default=50,
                       help='Number of successful episodes to collect')
    parser.add_argument('--verbose', action='store_true',
                       help='Print per-episode info')
    parser.add_argument('--render', action='store_true',
                       help='Open MuJoCo GUI viewer to watch collection live')
    args = parser.parse_args()

    collect_episodes(
        num_episodes=args.num_episodes,
        task_name=args.task,
        verbose=args.verbose,
        render=args.render,
    )
