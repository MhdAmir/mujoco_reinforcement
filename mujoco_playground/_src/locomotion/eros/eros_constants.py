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
"""Constants for the EROS humanoid robot."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "eros"

FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
  }[task_name]


# ---------------------------------------------------------------------------
# Foot geometry names (for contact detection with the floor).
# The EROS ankle-roll geoms are the only collision-enabled geoms on the feet.
# ---------------------------------------------------------------------------
LEFT_FEET_GEOMS = [
    "l_ankle_roll_geom",  # box geom on l_ankle_roll_link (contype=1)
]

RIGHT_FEET_GEOMS = [
    "r_ankle_roll_geom",  # box geom on r_ankle_roll_link (contype=1)
]

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

# ---------------------------------------------------------------------------
# Sites used for foot position tracking.
# These must exist in the XML (we reference them for site_xpos queries).
# ---------------------------------------------------------------------------
FEET_SITES = [
    "l_foot",
    "r_foot",
]

# ---------------------------------------------------------------------------
# Root body name.
# ---------------------------------------------------------------------------
ROOT_BODY = "base_link"

# ---------------------------------------------------------------------------
# IMU sensor names (all defined on the "imu" site in the XML).
# EROS only exposes IMU sensors — no dedicated foot sensors.
# ---------------------------------------------------------------------------
GRAVITY_SENSOR = "upvector"           # framezaxis → up-vector in world frame
GLOBAL_LINVEL_SENSOR = "global_linvel"  # framelinvel → linear vel in world frame
GLOBAL_ANGVEL_SENSOR = "global_angvel"  # frameangvel → angular vel in world frame
LOCAL_LINVEL_SENSOR = "local_linvel"   # velocimeter → linear vel in body frame
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# ---------------------------------------------------------------------------
# Leg joint names (right leg first, then left leg).
# Used to query self._leg_qpos_adr and self._leg_dof_adr by joint names.
# ---------------------------------------------------------------------------
LEG_JOINT_NAMES = [
    "r_hip_yaw",
    "r_hip_roll",
    "r_hip_pitch",
    "r_knee_pitch",
    "r_ankle_pitch",
    "r_ankle_roll",
    "l_hip_yaw",
    "l_hip_roll",
    "l_hip_pitch",
    "l_knee_pitch",
    "l_ankle_pitch",
    "l_ankle_roll",
]

# Number of leg DOFs controlled by the policy (arm/head joints are excluded).
NUM_LEG_DOFS = 12
