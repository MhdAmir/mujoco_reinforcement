# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick locomotion task for the EROS humanoid robot.

Design notes
------------
* The EROS robot has **only IMU sensors** (gyro, accelerometer, velocimeter,
  framezaxis, framelinvel, frameangvel). There are no dedicated foot-position
  or foot-velocity sensors.
* The **policy controls only the 12 leg joints** (r/l hip yaw/roll/pitch,
  r/l knee pitch, r/l ankle pitch/roll). The arm and head actuators are held
  at their home positions throughout training.
* Observations that required foot sensors (feet_vel, feet_clearance,
  feet_height, feet_phase) are replaced by foot positions derived directly
  from site_xpos (available in simulation) and by contact / air-time signals
  that rely only on collision geometry.
* The reward structure mirrors the Wolfgang Joystick task as closely as
  possible, but `feet_clearance`, `feet_height`, and `feet_phase` are
  disabled by default (scale = 0.0) because precise foot-height control
  without an accurate foot-velocity estimate is ill-conditioned.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.eros import base as eros_base
from mujoco_playground._src.locomotion.eros import eros_constants as consts


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              # Leg joint position noise (rad).
              hip_pos=0.03,
              knee_pos=0.05,
              ankle_pos=0.08,
              # Joint velocity noise (rad/s).
              joint_vel=0.2,
              # IMU noise.
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking rewards.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base-state rewards.
              lin_vel_z=-2.0,
              ang_vel_xy=-0.15,
              orientation=-1.0,
              base_height=0.0,
              # Energy rewards.
              torques=-2.5e-5,
              action_rate=-0.01,
              energy=-1.0e-3,
              # Foot rewards (contact/air-time only; no foot-vel sensors).
              feet_air_time=2.0,
              feet_slip=-0.25,
              # Phase-based rewards – disabled by default without foot sensors.
              feet_clearance=0.0,
              feet_height=0.0,
              feet_phase=1.0,
              # Other rewards.
              stand_still=0.0,
              alive=0.0,
              termination=-1.0,
              # Pose rewards.
              joint_deviation_knee=-0.1,
              joint_deviation_hip=-0.25,
              dof_pos_limits=-1.0,
              pose=-1.0,
          ),
          tracking_sigma=0.5,
          max_foot_height=0.1,
          base_height_target=-0.038,  # EROS base origin z when standing (~feet at ground)
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.05, 0.8],
      ),
      lin_vel_x=[-0.5, 0.5],
      lin_vel_y=[-0.5, 0.5],
      ang_vel_yaw=[-1.5, 1.5],
  )


# ---------------------------------------------------------------------------
# Joystick task
# ---------------------------------------------------------------------------

class Joystick(eros_base.ErosEnv):
  """Track a joystick (vx, vy, wz) command using IMU-only observations.

  The policy outputs residual joint angles for the **12 leg joints** only.
  Arm/head actuators are commanded to their home position at every step.
  """

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  # -------------------------------------------------------------------------
  # Action size override: policy controls only the 12 leg joints.
  # The base class returns self._mjx_model.nu = 20 (all actuators), but
  # the training framework uses this property to size the action vector,
  # so we must return 12 here to avoid shape mismatches.
  # -------------------------------------------------------------------------

  @property
  def action_size(self) -> int:
    return consts.NUM_LEG_DOFS  # 12

  # -------------------------------------------------------------------------
  # Initialisation helpers
  # -------------------------------------------------------------------------

  def _post_init(self) -> None:
    # -----------------------------------------------------------------------
    # Joint address lookup by NAME (robust against body-tree ordering).
    #
    # MuJoCo orders joints in qpos/qvel by depth-first body-tree traversal,
    # NOT by the actuator order in <actuator>.  For the EROS robot the tree
    # is:  base(free) → neck → head → r_arm(×3) → l_arm(×3) → r_leg(×6)
    #                                                           → l_leg(×6)
    # That means leg joints start at qpos[15], NOT qpos[7].
    # Using named lookups makes this code immune to XML reordering.
    # -----------------------------------------------------------------------
    self._leg_qpos_adr = np.array([
        self._mj_model.joint(name).qposadr for name in consts.LEG_JOINT_NAMES
    ]).squeeze()
    self._leg_dof_adr = np.array([
        self._mj_model.joint(name).dofadr for name in consts.LEG_JOINT_NAMES
    ]).squeeze()


    # Home pose from the keyframe.
    home_qpos = self._mj_model.keyframe("home").qpos
    self._init_q = jp.array(home_qpos)

    # Default leg joint targets — read at the CORRECT qpos addresses.
    self._default_pose = jp.array(home_qpos[self._leg_qpos_adr])

    # Build default ctrl in ACTUATOR order (legs first in the XML, then arms,
    # then neck/head).  For each actuator we look up its joint's home qpos.
    home_ctrl = np.zeros(self._mj_model.nu)
    for i in range(self._mj_model.nu):
      joint_id = self._mj_model.actuator_trnid[i, 0]
      qpos_adr = self._mj_model.jnt_qposadr[joint_id]
      home_ctrl[i] = home_qpos[qpos_adr]
    self._default_ctrl = jp.array(home_ctrl)

    # Soft joint limits — use the actual leg joint ranges by name.
    leg_ranges = np.array([
        self._mj_model.joint(name).range for name in consts.LEG_JOINT_NAMES
    ])
    self._lowers, self._uppers = leg_ranges.T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    # Hip joint indices within the 12-DOF leg vector (actuator order).
    # r_hip_yaw=0, r_hip_roll=1, l_hip_yaw=6, l_hip_roll=7
    self._hip_indices = jp.array([0, 1, 6, 7])

    # Knee joint indices within the 12-DOF leg vector.
    # r_knee_pitch=3, l_knee_pitch=9
    self._knee_indices = jp.array([3, 9])

    # Pose deviation  weights for each of the 12 leg joints.
    # fmt: off
    self._weights = jp.array([
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,   # right leg
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,   # left leg
    ])
    # fmt: on

    # Body / site IDs.
    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._imu_site_id = self._mj_model.site("imu").id

    # Foot site IDs (for site_xpos lookups → foot height).
    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )

    # Floor and foot collision geom IDs.
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    # Per-joint position noise scale (in leg-actuator order, matching policy).
    qpos_noise_scale = np.zeros(consts.NUM_LEG_DOFS)
    hip_ids    = [0, 1, 2, 6, 7, 8]   # yaw, roll, pitch (both sides)
    knee_ids   = [3, 9]
    ankle_ids  = [4, 5, 10, 11]
    qpos_noise_scale[hip_ids]   = self._config.noise_config.scales.hip_pos
    qpos_noise_scale[knee_ids]  = self._config.noise_config.scales.knee_pos
    qpos_noise_scale[ankle_ids] = self._config.noise_config.scales.ankle_pos
    self._qpos_noise_scale = jp.array(qpos_noise_scale)

  # -------------------------------------------------------------------------
  # Reset
  # -------------------------------------------------------------------------

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # Randomise initial XY position.
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)

    # Randomise initial yaw.
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # Randomise initial leg joint angles (*U(0.5, 1.5)).
    # Use _leg_qpos_adr (named address lookup) to reach the correct qpos slots.
    rng, key = jax.random.split(rng)
    scale = jax.random.uniform(
        key, (consts.NUM_LEG_DOFS,), minval=0.5, maxval=1.5
    )
    qpos = qpos.at[self._leg_qpos_adr].set(
        qpos[self._leg_qpos_adr] * scale
    )

    # Randomise initial base velocity.
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    # Initialise full actuator control at home position (leg + arm/head).
    data = mjx_env.init(
        self.mjx_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=self._default_ctrl,
    )

    # Gait phase initialisation.
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    rng, key = jax.random.split(rng)
    phase_offset = jax.random.uniform(key, minval=-jp.pi, maxval=jp.pi)
    phase = jp.array([phase_offset, phase_offset + jp.pi])
    phase = jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    # Push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(consts.NUM_LEG_DOFS),
        "last_last_act": jp.zeros(consts.NUM_LEG_DOFS),
        "motor_targets": jp.zeros(consts.NUM_LEG_DOFS),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        # Phase.
        "phase_dt": phase_dt,
        "phase": phase,
        # Push.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  # -------------------------------------------------------------------------
  # Step
  # -------------------------------------------------------------------------

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Advance the simulation by one control step.

    Args:
      state: Current environment state.
      action: 12-D residual action for the leg joints.

    Returns:
      Updated environment state.
    """
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)

    # --- Leg motor targets (residual from home pose) -------------------------
    leg_targets = self._default_pose + action * self._config.action_scale

    # Build full actuator command: leg joints first, then arm/head at home.
    full_ctrl = self._default_ctrl.at[:consts.NUM_LEG_DOFS].set(leg_targets)

    data = mjx_env.step(self.mjx_model, state.data, full_ctrl, self.n_substeps)
    state.info["motor_targets"] = leg_targets

    # Contact detection.
    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt

    # Track swing peak height from foot sites.
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  # -------------------------------------------------------------------------
  # Termination
  # -------------------------------------------------------------------------

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    """Terminate if the robot falls (up-vector points down) or qpos/qvel NaN."""
    fall_termination = self.get_gravity(data)[-1] < 0.0
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  # -------------------------------------------------------------------------
  # Observation (IMU-only)
  # -------------------------------------------------------------------------

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    """Build the observation dictionary.

    ``state`` (policy input):
      gyro (3) + gravity (3) + command (3) + joint_angles (12) +
      joint_vel (12) + last_action (12) + gait_phase (4) = 49-D

    ``privileged_state`` (teacher / critic input):
      state (49) + gyro_clean (3) + accel (3) + gravity_clean (3) +
      linvel_clean (3) + global_angvel (3) + joint_angles_clean (12) +
      joint_vel_clean (12) + root_height (1) + actuator_force (12) +
      contact (2) + feet_air_time (2) + foot_height (2) = 109-D
    """
    # --- Noisy gyro ---
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + jax.random.normal(noise_rng, shape=gyro.shape)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    # --- Noisy gravity (up-vector projected through IMU rotation matrix) ---
    gravity = data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + jax.random.normal(noise_rng, shape=gravity.shape)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    # --- Noisy leg joint angles (12 DOF, in actuator order) ---
    # Use _leg_qpos_adr so we read the CORRECT slots regardless of body-tree
    # ordering (leg joints start at qpos[15], not qpos[7], in the EROS model).
    joint_angles = data.qpos[self._leg_qpos_adr]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._qpos_noise_scale
    )

    # --- Noisy leg joint velocities (12 DOF, in actuator order) ---
    joint_vel = data.qvel[self._leg_dof_adr]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + jax.random.normal(noise_rng, shape=joint_vel.shape)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    # --- Noisy local linear velocity (from velocimeter) ---
    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    # --- Gait phase (cos/sin of each leg phase) ---
    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])  # shape (4,)

    # ---- Policy-visible state (no privileged info) -------------------------
    state = jp.hstack([
        noisy_gyro,                             # 3
        noisy_gravity,                          # 3
        info["command"],                        # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,                        # 12
        info["last_act"],                       # 12
        phase,                                  # 4
    ])  # total: 49

    # ---- Privileged / teacher state ----------------------------------------
    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
    root_height = data.qpos[2]
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_height = foot_pos[..., -1]  # z-coordinate of each foot site

    privileged_state = jp.hstack([
        state,
        gyro,                                   # 3  (clean)
        accelerometer,                          # 3
        gravity,                                # 3  (clean)
        linvel,                                 # 3  (clean)
        global_angvel,                          # 3
        joint_angles - self._default_pose,      # 12 (clean)
        joint_vel,                              # 12 (clean)
        root_height,                            # 1
        data.actuator_force[:consts.NUM_LEG_DOFS],  # 12
        contact,                                # 2
        info["feet_air_time"],                  # 2
        foot_height,                            # 2
    ])  # total: 49 + 3+3+3+3+3+12+12+1+12+2+2+2 = 109

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  # -------------------------------------------------------------------------
  # Reward
  # -------------------------------------------------------------------------

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        # Base-state rewards.
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_gravity(data)),
        "base_height": self._cost_base_height(data.qpos[2]),
        # Energy rewards.
        "torques": self._cost_torques(
            data.actuator_force[:consts.NUM_LEG_DOFS]
        ),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[self._leg_dof_adr],
                                    data.actuator_force[:consts.NUM_LEG_DOFS]),
        # Foot rewards (geometry/contact-based; no foot-velocity sensors).
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_slip": self._cost_feet_slip(data, contact),
        # Phase / clearance rewards (disabled by default; sensors absent).
        "feet_clearance": jp.zeros(()),
        "feet_height": jp.zeros(()),
        "feet_phase": self._reward_feet_phase(
            data, info["phase"], self._config.reward_config.max_foot_height, info["command"]
        ),
        # Other.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(
            info["command"], data.qpos[self._leg_qpos_adr]
        ),
        # Pose rewards.
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[self._leg_qpos_adr], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(
            data.qpos[self._leg_qpos_adr]
        ),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[self._leg_qpos_adr]),
        "pose": self._cost_pose(data.qpos[self._leg_qpos_adr]),
    }

  # -------------------------------------------------------------------------
  # Tracking rewards
  # -------------------------------------------------------------------------

  def _reward_tracking_lin_vel(
      self, commands: jax.Array, local_vel: jax.Array
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self, commands: jax.Array, ang_vel: jax.Array
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # -------------------------------------------------------------------------
  # Base-state costs
  # -------------------------------------------------------------------------

  def _reward_feet_phase(
      self,
      data: mjx.Data,
      phase: jax.Array,
      foot_height: jax.Array,
      commands: jax.Array,
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    del commands  # Unused.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    return reward


  def _cost_lin_vel_z(self, global_linvel: jax.Array) -> jax.Array:
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.square(
        base_height - self._config.reward_config.base_height_target
    )

  # -------------------------------------------------------------------------
  # Energy costs
  # -------------------------------------------------------------------------

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  # -------------------------------------------------------------------------
  # Other costs / rewards
  # -------------------------------------------------------------------------

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_stand_still(
      self, commands: jax.Array, qpos: jax.Array
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.1)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  # -------------------------------------------------------------------------
  # Pose costs
  # -------------------------------------------------------------------------

  def _cost_joint_deviation_hip(
      self, qpos: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    cost = jp.sum(
        jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices])
    )
    cost *= jp.abs(cmd[1]) > 0.1
    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(
        jp.abs(
            qpos[self._knee_indices] - self._default_pose[self._knee_indices]
        )
    )

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

  # -------------------------------------------------------------------------
  # Foot rewards (geometry / contact-based; no foot-velocity sensors needed)
  # -------------------------------------------------------------------------

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array
  ) -> jax.Array:
    """Penalise horizontal base velocity when a foot is in contact."""
    body_vel = self.get_global_linvel(data)[:2]
    return jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward

  # -------------------------------------------------------------------------
  # Command sampling
  # -------------------------------------------------------------------------

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1,
        minval=self._config.lin_vel_x[0],
        maxval=self._config.lin_vel_x[1],
    )
    lin_vel_y = jax.random.uniform(
        rng2,
        minval=self._config.lin_vel_y[0],
        maxval=self._config.lin_vel_y[1],
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    # With 10% chance, send a zero command (stand still).
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )
