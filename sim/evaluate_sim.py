"""Evaluate a trained policy in the MuJoCo simulation.

This module is the single rollout/evaluation path: train.py (periodic evals),
this CLI (headline evals) and scripts/eval_noise.py (noise sweeps) all go
through `evaluate()`.
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import TASK_CONFIG  # noqa: F401 — sets os.environ['DEVICE']
from sim.env import PandaPickPlaceEnv
from training.utils import get_image, load_run

SUCCESS_THRESHOLD_M = 0.05
EVAL_SEED_START = 5000


def make_env(task_config):
    """Env whose cameras and resolution match what the policy was trained on."""
    return PandaPickPlaceEnv(
        cam_width=task_config['cam_width'],
        cam_height=task_config['cam_height'],
        camera_names=task_config['camera_names'],
    )


def run_policy_episode(env, policy, stats, policy_config, device, seed,
                       max_steps=300, render_frames=False, obs_transform=None):
    """Run one seeded episode with the trained policy.

    Args:
        obs_transform: optional callable applied to every observation before
            the policy sees it (e.g. ObservationNoise). The simulator itself
            stays clean.

    Returns:
        success, frames (front-camera BGR frames if render_frames), final_dist
    """
    obs = env.reset(seed=seed)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    num_queries = policy_config['num_queries']
    # Diffusion executes each sampled chunk open-loop; ACT can instead
    # re-query every step and average overlapping chunks.
    temporal_agg = (policy_config.get('temporal_agg', False)
                    and policy_config['policy_class'] != 'Diffusion')
    query_frequency = 1 if temporal_agg else num_queries
    if temporal_agg:
        all_time_actions = torch.zeros(
            [max_steps, max_steps + num_queries, policy_config['action_dim']]
        ).to(device)

    frames = []
    all_actions = None

    with torch.inference_mode():
        for t in range(max_steps):
            policy_obs = obs_transform(obs) if obs_transform else obs
            qpos = pre_process(policy_obs['qpos'].astype(np.float32))
            qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
            curr_image = get_image(policy_obs['images'], policy_config['camera_names'], device)

            if t % query_frequency == 0:
                all_actions = policy(qpos, curr_image)

            if temporal_agg:
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

            action = post_process(raw_action.squeeze(0).cpu().numpy())

            if render_frames:
                frames.append(_annotated_frame(env, t))

            obs, _ = env.step(action)

    success = env.check_success(SUCCESS_THRESHOLD_M)
    final_dist = float(np.linalg.norm(env.get_cube_pos()[:2] - env.get_target_pos()[:2]))
    return success, frames, final_dist


def _annotated_frame(env, t):
    frame = cv2.cvtColor(env.render_camera('front'), cv2.COLOR_RGB2BGR)
    dist = np.linalg.norm(env.get_cube_pos()[:2] - env.get_target_pos()[:2])
    cv2.putText(frame, f't={t}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'dist={dist:.3f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame


def _label_episode(frames, ep, success, final_dist):
    result_text = "SUCCESS" if success else f"FAIL (d={final_dist:.3f})"
    colour = (0, 255, 0) if success else (0, 0, 255)
    for f in frames[:5]:
        cv2.putText(f, f'Episode {ep} - {result_text}', (10, f.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    for f in frames[-10:]:
        cv2.putText(f, result_text, (10, f.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)


def _write_video(path, frames, fps=25):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def evaluate(policy, stats, run_config, device, num_episodes, seed_start=EVAL_SEED_START,
             video_path=None, video_episodes=3, obs_transform=None, verbose=True):
    """Roll out `num_episodes` seeded episodes and summarise them.

    The first `video_episodes` episodes are recorded to `video_path` if given.

    Returns:
        dict with success counts, distance statistics and per-episode results.
    """
    policy_config = run_config['policy']
    task_config = run_config['task']
    env = make_env(task_config)
    policy.eval()

    episodes, video_frames = [], []
    for ep in range(num_episodes):
        seed = seed_start + ep
        record = video_path is not None and ep < video_episodes
        success, frames, final_dist = run_policy_episode(
            env, policy, stats, policy_config, device, seed,
            max_steps=task_config['episode_len'], render_frames=record,
            obs_transform=obs_transform,
        )
        episodes.append({'seed': seed, 'success': bool(success), 'final_dist_m': round(final_dist, 4)})
        if record:
            _label_episode(frames, ep, success, final_dist)
            video_frames.extend(frames)
        if verbose:
            print(f"  Episode {ep} (seed {seed}): {'SUCCESS' if success else 'FAIL'} (dist={final_dist:.3f})")
    env.close()

    if video_frames:
        _write_video(video_path, video_frames)

    dists = np.array([e['final_dist_m'] for e in episodes])
    ok = np.array([e['success'] for e in episodes])
    summary = {
        'num_episodes': num_episodes,
        'seed_range': [seed_start, seed_start + num_episodes - 1],
        'successes': int(ok.sum()),
        'success_rate': round(float(ok.mean()), 4),
        'success_threshold_m': SUCCESS_THRESHOLD_M,
        'distance_all': {'mean_m': round(float(dists.mean()), 4),
                         'median_m': round(float(np.median(dists)), 4)},
        'distance_successes_mean_m': round(float(dists[ok].mean()), 4) if ok.any() else None,
        'distance_failures_mean_m': round(float(dists[~ok].mean()), 4) if (~ok).any() else None,
        'episodes': episodes,
    }
    print(f"Evaluation: {summary['successes']}/{num_episodes} = {summary['success_rate']:.0%} success")
    return summary


def run_summary(run_config):
    """The parts of a run config worth reporting next to its results."""
    p, t = run_config['policy'], run_config['task']
    return {
        'policy': p['policy_class'],
        'backbone': p['backbone'],
        'chunk_size': p['num_queries'],
        'temporal_agg': p.get('temporal_agg', False),
        'cameras': p['camera_names'],
        'image_size': [t['cam_width'], t['cam_height']],
        'train_noise_level': run_config.get('noise_level', 0.0),
        **run_config.get('training', {}),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained policy in simulation')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Checkpoint path; run_config.json and dataset_stats.pkl must sit next to it')
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--seed_start', type=int, default=EVAL_SEED_START)
    parser.add_argument('--video', action='store_true', help='Record the first 3 episodes to eval_video.mp4')
    parser.add_argument('--output', type=str, default=None, help='Write results JSON here')
    args = parser.parse_args()

    device = os.environ['DEVICE']
    policy, stats, run_config = load_run(args.ckpt, device)
    video_path = os.path.join(os.path.dirname(args.ckpt), 'eval_video.mp4') if args.video else None

    summary = evaluate(policy, stats, run_config, device, args.num_episodes,
                       seed_start=args.seed_start, video_path=video_path)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({'checkpoint': args.ckpt, 'config': run_summary(run_config), 'eval': summary}, f, indent=2)
        print(f"Results written to {args.output}")
