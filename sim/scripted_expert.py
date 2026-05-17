"""Scripted IK-based expert for Franka Panda pick-and-place."""

import os
import sys
import numpy as np
import mujoco
from enum import Enum, auto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.env import PandaPickPlaceEnv, FINGER_OPEN, FINGER_CLOSED, TABLE_HEIGHT, CUBE_SIZE


class Phase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    TRANSPORT = auto()
    LOWER = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()


# Phase transition thresholds
POS_THRESHOLD = 0.01         # position error for phase transition (m)
GRASP_STEPS = 30             # steps to hold during grasp
RELEASE_STEPS = 20           # steps to hold during release
APPROACH_HEIGHT = 0.12       # height above table for approach/transport
PLACE_HEIGHT_OFFSET = 0.02   # height above table surface for placing


def compute_ik_delta(model, data, target_pos, body_id, ee_point_world,
                     desired_down=None, step_size=0.05, ori_weight=0.3):
    """Compute joint delta using damped least-squares IK with optional orientation.

    Args:
        model: MjModel
        data: MjData
        target_pos: desired fingertip position (3,) in world frame
        body_id: body the end-effector is attached to
        ee_point_world: current end-effector point in world frame (3,)
        desired_down: desired z-axis direction for gripper (3,), or None
        step_size: max joint angle change per step (rad)
        ori_weight: weight for orientation error relative to position

    Returns:
        dq: joint position delta (7,)
    """
    # Position error
    pos_err = target_pos - ee_point_world

    # Compute Jacobian for the point attached to the body
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, ee_point_world, body_id)

    # Use only arm joints (first 7 columns)
    Jp = jacp[:, :7]

    if desired_down is not None:
        # Orientation control: align the hand's z-axis with desired direction
        body_rot = data.xmat[body_id].reshape(3, 3)
        current_z = body_rot[:, 2]  # z-axis of hand frame

        # Orientation error as cross product (proportional to sin(angle))
        ori_err = np.cross(current_z, desired_down) * ori_weight

        Jr = jacr[:, :7]

        # Stack position and orientation
        J = np.vstack([Jp, Jr])
        err = np.concatenate([pos_err, ori_err])
    else:
        J = Jp
        err = pos_err

    # Damped least-squares
    damping = 1e-4
    JJT = J @ J.T + damping * np.eye(J.shape[0])
    dq = J.T @ np.linalg.solve(JJT, err)

    # Clip to maximum step size
    dq_norm = np.linalg.norm(dq)
    if dq_norm > step_size:
        dq = dq * step_size / dq_norm

    return dq


class PickPlaceExpert:
    """State-machine based scripted expert for pick-and-place."""

    # Offset from hand body frame to fingertip center
    EE_OFFSET = np.array([0, 0, 0.1034])

    # Desired gripper orientation: pointing straight down
    DESIRED_DOWN = np.array([0, 0, -1.0])

    # Home joint configuration (from mujoco_menagerie panda keyframe)
    HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

    def __init__(self, env: PandaPickPlaceEnv):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.hand_body_id = env._hand_body_id

        self.phase = Phase.APPROACH
        self.phase_step = 0

    def reset(self):
        """Reset expert state for a new episode."""
        self.phase = Phase.APPROACH
        self.phase_step = 0

    def _get_ee_pos(self):
        """Get current end-effector (fingertip) position in world frame."""
        body_pos = self.data.xpos[self.hand_body_id]
        body_rot = self.data.xmat[self.hand_body_id].reshape(3, 3)
        return body_pos + body_rot @ self.EE_OFFSET

    def _get_target_for_phase(self):
        """Get the target end-effector position for the current phase."""
        cube_pos = self.env.get_cube_pos()
        target_pos = self.env.get_target_pos()

        if self.phase == Phase.APPROACH:
            return np.array([cube_pos[0], cube_pos[1],
                           TABLE_HEIGHT + CUBE_SIZE + APPROACH_HEIGHT])
        elif self.phase == Phase.DESCEND:
            # Fingertips at cube center height
            return np.array([cube_pos[0], cube_pos[1],
                           TABLE_HEIGHT + CUBE_SIZE])
        elif self.phase == Phase.GRASP:
            return self._get_ee_pos()
        elif self.phase == Phase.LIFT:
            return np.array([cube_pos[0], cube_pos[1],
                           TABLE_HEIGHT + APPROACH_HEIGHT + CUBE_SIZE])
        elif self.phase == Phase.TRANSPORT:
            return np.array([target_pos[0], target_pos[1],
                           TABLE_HEIGHT + APPROACH_HEIGHT + CUBE_SIZE])
        elif self.phase == Phase.LOWER:
            return np.array([target_pos[0], target_pos[1],
                           TABLE_HEIGHT + CUBE_SIZE + PLACE_HEIGHT_OFFSET])
        elif self.phase == Phase.RELEASE:
            return self._get_ee_pos()
        elif self.phase == Phase.RETREAT:
            return np.array([target_pos[0], target_pos[1],
                           TABLE_HEIGHT + APPROACH_HEIGHT + CUBE_SIZE])
        else:
            return self._get_ee_pos()

    def _should_transition(self):
        """Check if the current phase should transition to the next."""
        if self.phase == Phase.GRASP:
            return self.phase_step >= GRASP_STEPS
        if self.phase == Phase.RELEASE:
            return self.phase_step >= RELEASE_STEPS
        if self.phase == Phase.DONE:
            return False

        target = self._get_target_for_phase()
        ee_pos = self._get_ee_pos()
        dist = np.linalg.norm(target - ee_pos)
        return dist < POS_THRESHOLD

    def _next_phase(self):
        """Advance to the next phase."""
        transitions = {
            Phase.APPROACH: Phase.DESCEND,
            Phase.DESCEND: Phase.GRASP,
            Phase.GRASP: Phase.LIFT,
            Phase.LIFT: Phase.TRANSPORT,
            Phase.TRANSPORT: Phase.LOWER,
            Phase.LOWER: Phase.RELEASE,
            Phase.RELEASE: Phase.RETREAT,
            Phase.RETREAT: Phase.DONE,
        }
        self.phase = transitions.get(self.phase, Phase.DONE)
        self.phase_step = 0

    def get_action(self):
        """Compute the next action.

        Returns:
            action: (8,) array — 7 joint position targets + 1 gripper target
        """
        if self._should_transition():
            self._next_phase()

        self.phase_step += 1

        current_qpos = self.data.qpos[:7].copy()

        # Gripper: open during approach/descend/release/retreat, closed otherwise
        if self.phase in (Phase.APPROACH, Phase.DESCEND,
                         Phase.RELEASE, Phase.RETREAT, Phase.DONE):
            gripper = FINGER_OPEN
        else:
            gripper = FINGER_CLOSED

        # Arm IK
        if self.phase in (Phase.GRASP, Phase.RELEASE):
            target_qpos = current_qpos
        elif self.phase == Phase.DONE:
            # Move back toward home position to stay clear of the cube
            alpha = 0.15  # blend speed toward home
            target_qpos = current_qpos + alpha * (self.HOME_QPOS - current_qpos)
        else:
            target_pos = self._get_target_for_phase()
            ee_pos = self._get_ee_pos()

            # Use orientation control to keep gripper pointing down
            dq = compute_ik_delta(
                self.model, self.data, target_pos,
                self.hand_body_id, ee_pos,
                desired_down=self.DESIRED_DOWN,
                step_size=0.1,
                ori_weight=0.5
            )
            target_qpos = current_qpos + dq

            # Clip to joint limits
            for i in range(7):
                lo = self.model.jnt_range[i, 0]
                hi = self.model.jnt_range[i, 1]
                if lo < hi:
                    target_qpos[i] = np.clip(target_qpos[i], lo + 0.05, hi - 0.05)

        return np.concatenate([target_qpos, [gripper]])

    @property
    def is_done(self):
        return self.phase == Phase.DONE


def run_expert_episode(env, max_steps=300, seed=None, verbose=False):
    """Run one expert episode and return observations and actions.

    Returns:
        obs_list: list of observation dicts
        action_list: list of (8,) action arrays
        success: whether cube was placed successfully
    """
    obs = env.reset(seed=seed)
    expert = PickPlaceExpert(env)

    obs_list = []
    action_list = []

    for t in range(max_steps):
        action = expert.get_action()
        obs_list.append(obs)
        action_list.append(action)

        obs, success = env.step(action)

        if verbose and t % 30 == 0:
            ee_pos = expert._get_ee_pos()
            cube_pos = env.get_cube_pos()
            print(f"  t={t:3d} phase={expert.phase.name:12s} "
                  f"ee_z={ee_pos[2]:.3f} cube_xy=[{cube_pos[0]:.3f},{cube_pos[1]:.3f}]")

    success = env.check_success()
    return obs_list, action_list, success


if __name__ == '__main__':
    env = PandaPickPlaceEnv()

    n_success = 0
    n_episodes = 10
    for ep in range(n_episodes):
        print(f"\n--- Episode {ep} ---")
        obs_list, action_list, success = run_expert_episode(
            env, max_steps=300, seed=ep, verbose=True
        )
        cube_pos = env.get_cube_pos()
        target_pos = env.get_target_pos()
        dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        print(f"  Success: {success}  dist={dist:.3f}")
        if success:
            n_success += 1

    print(f"\nSuccess rate: {n_success}/{n_episodes} = {n_success/n_episodes:.0%}")
    env.close()
