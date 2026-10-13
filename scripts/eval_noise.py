"""Evaluate a trained policy under increasing observation noise.

Produces a degradation curve (success rate vs noise level). Noise corrupts what
the policy sees (camera images and joint readings); the simulator stays clean.

Usage:
    uv run python scripts/eval_noise.py \
        --ckpt checkpoints/pick_place_act/policy_best.ckpt \
        --noise_levels 0.0,0.3,0.5,0.7,1.0 --num_episodes 20 \
        --output results/noise_act.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import NOISE_CONFIG
from sim.evaluate_sim import evaluate, run_summary
from training.noise import ObservationNoise
from training.utils import load_run

NOISE_SEED_START = 9000


def main():
    parser = argparse.ArgumentParser(description='Noise-robustness sweep')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--noise_levels', type=str, default='0.0,0.3,0.5,0.7,1.0')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--output', type=str, required=True, help='Results JSON path')
    args = parser.parse_args()

    device = os.environ['DEVICE']
    policy, stats, run_config = load_run(args.ckpt, device)
    camera_names = run_config['policy']['camera_names']

    sweep = []
    for level in [float(x) for x in args.noise_levels.split(',')]:
        print(f"\n--- Observation noise level {level:.1f} ---")
        noise = ObservationNoise(NOISE_CONFIG, level, camera_names, seed=0)
        summary = evaluate(policy, stats, run_config, device, args.num_episodes,
                           seed_start=NOISE_SEED_START, obs_transform=noise, verbose=False)
        sweep.append({'noise_level': level,
                      'success_rate': summary['success_rate'],
                      'mean_dist_m': summary['distance_all']['mean_m']})

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'checkpoint': args.ckpt, 'config': run_summary(run_config),
                   'noise_config': NOISE_CONFIG, 'num_episodes': args.num_episodes,
                   'sweep': sweep}, f, indent=2)
    for row in sweep:
        print(f"  noise {row['noise_level']:.1f}: {row['success_rate']:.0%}")
    print(f"Results written to {args.output}")


if __name__ == '__main__':
    main()
