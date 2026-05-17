"""MuJoCo environment for Franka Panda pick-and-place task."""

import os
import numpy as np
import mujoco

# Path to mujoco_menagerie Franka Panda model (bundled at repo root)
_MENAGERIE_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'mujoco_menagerie', 'franka_emika_panda'
)
_MENAGERIE_DIR = os.path.abspath(_MENAGERIE_DIR)

# Gripper constants
GRIPPER_OPEN = 255    # ctrl value for fully open gripper
GRIPPER_CLOSED = 0    # ctrl value for fully closed gripper
FINGER_OPEN = 0.04    # finger joint position when fully open (meters)
FINGER_CLOSED = 0.0   # finger joint position when fully closed

# Workspace bounds for randomizing cube and target
CUBE_X_RANGE = (0.35, 0.65)
CUBE_Y_RANGE = (-0.2, 0.2)
TARGET_X_RANGE = (0.35, 0.65)
TARGET_Y_RANGE = (-0.2, 0.2)
TABLE_HEIGHT = 0.4
CUBE_SIZE = 0.02
MIN_CUBE_TARGET_DIST = 0.10  # minimum distance between cube and target


def _build_scene_xml():
    """Build MJCF XML string for the pick-and-place scene."""
    assets_dir = os.path.join(_MENAGERIE_DIR, 'assets')
    panda_xml_path = os.path.join(_MENAGERIE_DIR, 'panda.xml')

    return f"""
<mujoco model="panda_pick_place">
  <include file="{panda_xml_path}"/>

  <compiler meshdir="{assets_dir}"/>
  <option cone="elliptic" impratio="10"/>

  <statistic center="0.4 0 0.4" extent="1"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
    <material name="table_mat" rgba="0.45 0.35 0.25 1" specular="0.3" shininess="0.5"/>
    <material name="cube_mat" rgba="0.9 0.2 0.2 1" specular="0.5" shininess="0.8"/>
    <material name="target_mat" rgba="0.2 0.9 0.2 0.4"/>
  </asset>

  <worldbody>
    <light pos="0.5 0 1.5" dir="0 0 -1" directional="true"/>
    <light pos="0.5 0.5 1.0" dir="-0.2 -0.2 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- Table -->
    <body name="table" pos="0.5 0 {TABLE_HEIGHT / 2}">
      <geom type="box" size="0.35 0.35 {TABLE_HEIGHT / 2}" material="table_mat"
            mass="100"/>
    </body>

    <!-- Cube (free body) -->
    <body name="cube" pos="0.5 0 {TABLE_HEIGHT + CUBE_SIZE}">
      <freejoint name="cube_joint"/>
      <geom type="box" size="{CUBE_SIZE} {CUBE_SIZE} {CUBE_SIZE}"
            material="cube_mat" mass="0.05" friction="1.0 0.005 0.0001"
            condim="4" priority="1"/>
    </body>

    <!-- Target marker (visual only) -->
    <site name="target_site" pos="0.5 0.15 {TABLE_HEIGHT + 0.001}"
          size="0.03 0.001" type="cylinder" material="target_mat"/>

    <!-- Front camera -->
    <camera name="front" pos="1.3 0 0.9" xyaxes="0 1 0 -0.5 0 0.87"
            fovy="45"/>
  </worldbody>
</mujoco>
"""


class PandaPickPlaceEnv:
    """MuJoCo environment for Franka Panda pick-and-place."""

    # Joint indices in qpos
    ARM_JOINTS = 7       # joints 0-6
    FINGER_JOINT1 = 7    # finger_joint1
    FINGER_JOINT2 = 8    # finger_joint2

    # Number of sub-steps per action step
    N_SUBSTEPS = 20

    def __init__(self, cam_width=640, cam_height=480, render_gui=False):
        self.cam_width = cam_width
        self.cam_height = cam_height

        xml = _build_scene_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Cache body/joint/actuator IDs
        self._cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        self._hand_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        self._target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, 'target_site')

        # Cube joint address in qpos (freejoint = 7 DoF: 3 pos + 4 quat)
        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
        self._cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]

        # Offscreen renderer
        self.renderer = mujoco.Renderer(self.model, self.cam_height, self.cam_width)

        # GUI viewer (optional)
        self._viewer = None
        if render_gui:
            from mujoco.viewer import launch_passive
            self._viewer = launch_passive(self.model, self.data)

        # Set home position
        home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key_id)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None):
        """Reset environment with randomized cube and target positions."""
        if seed is not None:
            np.random.seed(seed)

        # Reset to home keyframe
        home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key_id)

        # Randomize cube position on table
        cube_x = np.random.uniform(*CUBE_X_RANGE)
        cube_y = np.random.uniform(*CUBE_Y_RANGE)
        cube_z = TABLE_HEIGHT + CUBE_SIZE + 0.001  # slightly above table

        # Randomize target position on table (ensuring minimum distance from cube)
        for _ in range(100):
            target_x = np.random.uniform(*TARGET_X_RANGE)
            target_y = np.random.uniform(*TARGET_Y_RANGE)
            dist = np.sqrt((target_x - cube_x)**2 + (target_y - cube_y)**2)
            if dist >= MIN_CUBE_TARGET_DIST:
                break

        # Set cube position (freejoint: 3 pos + 4 quat)
        addr = self._cube_qpos_addr
        self.data.qpos[addr:addr+3] = [cube_x, cube_y, cube_z]
        self.data.qpos[addr+3:addr+7] = [1, 0, 0, 0]  # identity quaternion

        # Set target position
        self.model.site_pos[self._target_site_id] = [
            target_x, target_y, TABLE_HEIGHT + 0.001
        ]

        # Zero velocities
        self.data.qvel[:] = 0

        # Forward to compute derived quantities
        mujoco.mj_forward(self.model, self.data)

        # Let the simulation settle (cube on table)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        return self._get_obs()

    def step(self, action):
        """Apply action and step simulation.

        Args:
            action: (8,) array — 7 joint position targets + 1 gripper finger target (0-0.04)

        Returns:
            obs: observation dict
            success: whether cube is within 5cm of target
        """
        # Set arm joint position targets (actuators 0-6)
        self.data.ctrl[:7] = action[:7]

        # Convert finger target (0-0.04) to gripper actuator command (0-255)
        gripper_target = np.clip(action[7], FINGER_CLOSED, FINGER_OPEN)
        self.data.ctrl[7] = gripper_target / FINGER_OPEN * GRIPPER_OPEN

        # Step simulation
        for _ in range(self.N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        if self._viewer is not None:
            self._viewer.sync()

        obs = self._get_obs()
        success = self.check_success()
        return obs, success

    def _get_obs(self):
        """Get current observation."""
        # Arm joint positions (7) + left finger position (1)
        qpos = np.concatenate([
            self.data.qpos[:self.ARM_JOINTS].copy(),
            [self.data.qpos[self.FINGER_JOINT1]]
        ])
        # Arm joint velocities (7) + left finger velocity (1)
        qvel = np.concatenate([
            self.data.qvel[:self.ARM_JOINTS].copy(),
            [self.data.qvel[self.FINGER_JOINT1]]
        ])
        # Camera image
        image = self.render_camera()

        return {
            'qpos': qpos.astype(np.float64),
            'qvel': qvel.astype(np.float64),
            'images': {'front': image},
        }

    def render_camera(self, camera_name='front'):
        """Render RGB image from named camera."""
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render().copy()

    def get_cube_pos(self):
        """Get current cube position (3,)."""
        addr = self._cube_qpos_addr
        return self.data.qpos[addr:addr+3].copy()

    def get_target_pos(self):
        """Get target position (3,)."""
        return self.model.site_pos[self._target_site_id].copy()

    def get_ee_pos(self):
        """Get end-effector (hand) position (3,)."""
        return self.data.xpos[self._hand_body_id].copy()

    def get_ee_pose(self):
        """Get end-effector position and rotation matrix."""
        pos = self.data.xpos[self._hand_body_id].copy()
        rot = self.data.xmat[self._hand_body_id].reshape(3, 3).copy()
        return pos, rot

    def get_gripper_tip_pos(self):
        """Get position of the gripper fingertip center."""
        hand_pos = self.data.xpos[self._hand_body_id]
        hand_rot = self.data.xmat[self._hand_body_id].reshape(3, 3)
        # Offset from hand frame origin to fingertip center
        tip_offset = np.array([0, 0, 0.1034])
        return hand_pos + hand_rot @ tip_offset

    def check_success(self, threshold=0.05):
        """Check if cube is within threshold distance of target."""
        cube_pos = self.get_cube_pos()
        target_pos = self.get_target_pos()
        target_pos_3d = np.array([target_pos[0], target_pos[1], cube_pos[2]])
        dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        return dist < threshold

    def close(self):
        """Clean up resources."""
        if self._viewer is not None:
            self._viewer.close()
        self.renderer.close()


if __name__ == '__main__':
    # Quick test: create env, reset, render, and show info
    env = PandaPickPlaceEnv()
    obs = env.reset(seed=42)
    print(f"qpos shape: {obs['qpos'].shape}")
    print(f"qvel shape: {obs['qvel'].shape}")
    print(f"image shape: {obs['images']['front'].shape}")
    print(f"image dtype: {obs['images']['front'].dtype}")
    print(f"Cube position: {env.get_cube_pos()}")
    print(f"Target position: {env.get_target_pos()}")
    print(f"EE position: {env.get_ee_pos()}")
    print(f"Gripper tip: {env.get_gripper_tip_pos()}")

    # Step with home position action
    action = np.zeros(8)
    action[:7] = obs['qpos'][:7]
    action[7] = FINGER_OPEN
    obs2, success = env.step(action)
    print(f"\nAfter step - success: {success}")
    print(f"qpos: {obs2['qpos']}")

    env.close()
    print("\nEnvironment test passed!")
