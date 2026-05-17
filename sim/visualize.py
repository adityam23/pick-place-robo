"""Visualize expert episodes - render video and/or display images."""

import os
import sys
import numpy as np
import cv2
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.env import PandaPickPlaceEnv
from sim.scripted_expert import run_expert_episode


def render_episode_video(output_path='videos/expert_demo.mp4', seed=42,
                        max_steps=300, fps=25):
    """Render a single expert episode to a video file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    env = PandaPickPlaceEnv(cam_width=640, cam_height=480)
    obs = env.reset(seed=seed)

    from sim.scripted_expert import PickPlaceExpert
    expert = PickPlaceExpert(env)

    frames = []
    for t in range(max_steps):
        # Render frame
        frame = env.render_camera('front')
        # Add text overlay
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, f't={t} {expert.phase.name}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cube_pos = env.get_cube_pos()
        target_pos = env.get_target_pos()
        dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        cv2.putText(frame_bgr, f'dist={dist:.3f}',
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        frames.append(frame_bgr)

        # Step
        action = expert.get_action()
        obs, success = env.step(action)

    # Write video
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    env.close()

    print(f"Video saved to: {output_path}")
    print(f"Success: {success}, final dist: {dist:.3f}")
    return success


def render_episode_frames(output_dir='videos/frames', seed=42, max_steps=300):
    """Render key frames from an episode as images."""
    os.makedirs(output_dir, exist_ok=True)

    env = PandaPickPlaceEnv(cam_width=640, cam_height=480)
    obs = env.reset(seed=seed)

    from sim.scripted_expert import PickPlaceExpert
    expert = PickPlaceExpert(env)

    # Save initial frame
    frame = env.render_camera('front')
    cv2.imwrite(os.path.join(output_dir, f'frame_000_INIT.png'),
               cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    prev_phase = expert.phase
    for t in range(max_steps):
        action = expert.get_action()

        # Save frame on phase transitions
        if expert.phase != prev_phase:
            frame = env.render_camera('front')
            fname = f'frame_{t:03d}_{expert.phase.name}.png'
            cv2.imwrite(os.path.join(output_dir, fname),
                       cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            prev_phase = expert.phase

        obs, success = env.step(action)

    # Save final frame
    frame = env.render_camera('front')
    cv2.imwrite(os.path.join(output_dir, f'frame_{max_steps:03d}_FINAL.png'),
               cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    env.close()
    print(f"Key frames saved to: {output_dir}")
    print(f"Success: {success}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize expert episodes')
    parser.add_argument('--mode', choices=['video', 'frames', 'both'], default='both')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='videos/expert_demo.mp4')
    args = parser.parse_args()

    if args.mode in ('video', 'both'):
        render_episode_video(output_path=args.output, seed=args.seed)
    if args.mode in ('frames', 'both'):
        render_episode_frames(seed=args.seed)
