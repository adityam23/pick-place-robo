"""Evaluate a trained ACT policy in the MuJoCo simulation."""

import os
import sys
import cv2
import torch
import pickle
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from sim.env import PandaPickPlaceEnv
from training.utils import make_policy, get_image


def load_policy(ckpt_path, stats_path, policy_config, device):
    """Load a trained policy and normalization stats."""
    policy = make_policy(policy_config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(
        torch.load(ckpt_path, map_location=torch.device(device), weights_only=True)
    )
    print(f"Loaded checkpoint: {ckpt_path} ({loading_status})")
    policy.to(device)
    policy.eval()

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    return policy, stats


def run_policy_episode(env, policy, stats, policy_config, device,
                       max_steps=300, render_frames=False):
    """Run one episode with the trained policy.

    Args:
        env: PandaPickPlaceEnv
        policy: trained ACTPolicy
        stats: normalization stats dict
        policy_config: policy configuration dict
        device: torch device string
        max_steps: max timesteps
        render_frames: if True, collect rendered frames for video

    Returns:
        success: whether cube reached target
        frames: list of BGR frames (empty if render_frames=False)
        final_dist: final cube-target distance
    """
    obs = env.reset()

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if policy_config.get('temporal_agg', False):
        query_frequency = 1
        num_queries = policy_config['num_queries']
        all_time_actions = torch.zeros(
            [max_steps, max_steps + num_queries, policy_config['action_dim']]
        ).to(device)

    frames = []
    all_actions = None

    with torch.inference_mode():
        for t in range(max_steps):
            # Pre-process observation
            qpos_numpy = obs['qpos'].astype(np.float32)
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
            curr_image = get_image(obs['images'], policy_config['camera_names'], device)

            # Query policy
            if t % query_frequency == 0:
                all_actions = policy(qpos, curr_image)

            if policy_config.get('temporal_agg', False):
                all_time_actions[[t], t:t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = all_actions[:, t % query_frequency]

            # Post-process action
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)

            # Render frame before stepping
            if render_frames:
                frame = env.render_camera('front')
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Add overlay text
                cube_pos = env.get_cube_pos()
                target_pos = env.get_target_pos()
                dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
                cv2.putText(frame_bgr, f't={t}',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_bgr, f'dist={dist:.3f}',
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                frames.append(frame_bgr)

            # Step environment
            obs, success = env.step(action)

    # Final success check
    success = env.check_success()
    cube_pos = env.get_cube_pos()
    target_pos = env.get_target_pos()
    final_dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])

    return success, frames, final_dist


def render_eval_video(env, policy, stats, policy_config, device,
                      output_path, num_episodes=3, max_steps=300, fps=25,
                      seed_offset=0):
    """Run evaluation episodes and save a compiled video.

    Args:
        env: PandaPickPlaceEnv
        policy: trained ACTPolicy
        stats: normalization stats
        policy_config: policy configuration
        device: torch device string
        output_path: path to save MP4 video
        num_episodes: number of episodes to render
        max_steps: max steps per episode
        fps: video frame rate
        seed_offset: base seed for reproducibility

    Returns:
        successes: list of bool per episode
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_frames = []
    successes = []

    for ep in range(num_episodes):
        np.random.seed(seed_offset + ep + 5000)  # deterministic seeds
        env.reset(seed=seed_offset + ep + 5000)

        success, frames, final_dist = run_policy_episode(
            env, policy, stats, policy_config, device,
            max_steps=max_steps, render_frames=True
        )
        successes.append(success)

        # Add episode header to first few frames
        for i, f in enumerate(frames[:5]):
            result_text = "SUCCESS" if success else f"FAIL (d={final_dist:.3f})"
            cv2.putText(f, f'Episode {ep} - {result_text}',
                       (10, f.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0) if success else (0, 0, 255), 2)

        # Add result to last frames
        for f in frames[-10:]:
            result_text = "SUCCESS" if success else f"FAIL (d={final_dist:.3f})"
            cv2.putText(f, result_text,
                       (10, f.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0) if success else (0, 0, 255), 2)

        all_frames.extend(frames)

        # Add blank separator frames between episodes
        if ep < num_episodes - 1 and len(frames) > 0:
            blank = np.zeros_like(frames[0])
            cv2.putText(blank, f'Episode {ep}: {"SUCCESS" if success else "FAIL"}',
                       (blank.shape[1] // 4, blank.shape[0] // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            all_frames.extend([blank] * int(fps * 0.5))  # 0.5s pause

    # Write video
    if all_frames:
        h, w = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for f in all_frames:
            writer.write(f)
        writer.release()

    n_success = sum(successes)
    print(f"Eval video: {output_path} | {n_success}/{num_episodes} success")
    return successes


def evaluate_policy(ckpt_path, stats_path, num_episodes=50, max_steps=300,
                    render_video=False, video_path=None,
                    chunk_size=None, temporal_agg=False):
    """Full evaluation: run N episodes, report success rate, optionally save video."""
    cfg = TASK_CONFIG
    policy_config = dict(POLICY_CONFIG)
    if chunk_size is not None:
        policy_config['num_queries'] = chunk_size
    if temporal_agg:
        policy_config['temporal_agg'] = True
    device = os.environ.get('DEVICE', 'cpu')

    policy, stats = load_policy(ckpt_path, stats_path, policy_config, device)

    env = PandaPickPlaceEnv(
        cam_width=cfg['cam_width'],
        cam_height=cfg['cam_height'],
    )

    if render_video and video_path:
        successes = render_eval_video(
            env, policy, stats, policy_config, device,
            output_path=video_path,
            num_episodes=num_episodes,
            max_steps=max_steps,
        )
    else:
        successes = []
        for ep in range(num_episodes):
            env.reset(seed=ep + 5000)
            success, _, final_dist = run_policy_episode(
                env, policy, stats, policy_config, device,
                max_steps=max_steps, render_frames=False
            )
            successes.append(success)
            print(f"  Episode {ep}: {'SUCCESS' if success else 'FAIL'} (dist={final_dist:.3f})")

    env.close()

    n_success = sum(successes)
    print(f"\nEvaluation: {n_success}/{num_episodes} = {n_success/num_episodes:.0%} success")
    return successes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained policy in simulation')
    parser.add_argument('--task', type=str, default='pick_place')
    parser.add_argument('--ckpt', type=str, default=None,
                       help='Checkpoint path (default: policy_last.ckpt)')
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--video', action='store_true', help='Save evaluation video')
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--chunk_size', type=int, default=None,
                       help='Action chunk size (must match training)')
    parser.add_argument('--temporal_agg', action='store_true',
                       help='Enable temporal aggregation')
    args = parser.parse_args()

    train_cfg = TRAIN_CONFIG
    ckpt_dir = os.path.join(train_cfg['checkpoint_dir'], args.task)

    if args.ckpt:
        ckpt_path = args.ckpt
        # Derive stats path from checkpoint directory
        stats_path = os.path.join(os.path.dirname(ckpt_path), 'dataset_stats.pkl')
    else:
        ckpt_path = os.path.join(ckpt_dir, train_cfg['eval_ckpt_name'])
        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')

    video_path = args.video_path
    if args.video and not video_path:
        video_path = os.path.join(ckpt_dir, 'eval_video.mp4')

    evaluate_policy(
        ckpt_path=ckpt_path,
        stats_path=stats_path,
        num_episodes=args.num_episodes,
        render_video=args.video,
        video_path=video_path,
        chunk_size=args.chunk_size,
        temporal_agg=args.temporal_agg,
    )
