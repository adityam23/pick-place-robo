"""Run 50-episode expert baseline and save video of 5 successful runs."""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.env import PandaPickPlaceEnv
from sim.scripted_expert import run_expert_episode, PickPlaceExpert


def run_expert_eval(num_episodes=50, video_episodes=5):
    """Run expert for 50 episodes, save stats and a video of successful runs."""
    env = PandaPickPlaceEnv(cam_width=640, cam_height=480)

    successes = []
    distances = []
    video_frames = []
    video_successes_collected = 0

    for ep in range(num_episodes):
        seed = ep + 7000
        obs = env.reset(seed=seed)
        expert = PickPlaceExpert(env)

        ep_frames = []
        for t in range(300):
            # Render frame for video candidates
            if video_successes_collected < video_episodes:
                frame = env.render_camera('front')
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f'Expert ep={ep} t={t} {expert.phase.name}',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                ep_frames.append(frame_bgr)

            action = expert.get_action()
            obs, _ = env.step(action)

        success = env.check_success()
        cube_pos = env.get_cube_pos()
        target_pos = env.get_target_pos()
        dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])

        successes.append(success)
        distances.append(dist)

        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep}: {status} (dist={dist:.3f})")

        # Keep frames from successful episodes for the video
        if success and video_successes_collected < video_episodes and ep_frames:
            # Add result overlay to last frames
            for f in ep_frames[-15:]:
                cv2.putText(f, "SUCCESS", (10, f.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            video_frames.extend(ep_frames)
            video_successes_collected += 1

            # Add separator
            if video_successes_collected < video_episodes:
                blank = np.zeros_like(ep_frames[0])
                cv2.putText(blank, f'Expert: {video_successes_collected}/{video_episodes} collected',
                           (blank.shape[1] // 4, blank.shape[0] // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                video_frames.extend([blank] * 12)

    env.close()

    # Stats
    n_success = sum(successes)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    print(f"\n=== Expert Baseline (50 episodes) ===")
    print(f"Success rate: {n_success}/{num_episodes} = {n_success/num_episodes:.0%}")
    print(f"Distance: mean={mean_dist:.4f}, std={std_dist:.4f}")

    # Save video
    os.makedirs('checkpoints/expert_baseline', exist_ok=True)
    video_path = 'checkpoints/expert_baseline/expert_50ep.mp4'
    if video_frames:
        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, 25, (w, h))
        for f in video_frames:
            writer.write(f)
        writer.release()
        print(f"Video saved: {video_path} ({video_successes_collected} successful episodes)")

    return successes, distances


if __name__ == '__main__':
    run_expert_eval()
