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
"""Base class for EROS humanoid robot environments.

The EROS robot exposes only IMU sensors (gyro, accelerometer, velocimeter,
framezaxis, framelinvel, frameangvel).  There are no dedicated foot-position
or foot-velocity sensors, so the observation and reward design is adapted
accordingly.
"""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.eros import eros_constants as consts


def get_assets() -> Dict[str, bytes]:
  """Return all XML/mesh assets needed to load the EROS model."""
  assets = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
  return assets


class ErosEnv(mjx_env.MjxEnv):
  """Base class for EROS locomotion environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=get_assets()
    )
    self._mj_model.opt.timestep = self.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model)
    self._xml_path = xml_path

  # ---------------------------------------------------------------------------
  # IMU sensor readings (all EROS sensors originate from the "imu" site).
  # ---------------------------------------------------------------------------

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    """Return the up-vector (z-axis) of the IMU site in world frame.

    A value of [0, 0, 1] means the robot is perfectly upright.
    """
    return mjx_env.get_sensor_data(self.mj_model, data, consts.GRAVITY_SENSOR)

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the IMU site in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR
    )

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    """Return the angular velocity of the IMU site in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
    )

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the IMU site in the body (local) frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.LOCAL_LINVEL_SENSOR
    )

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    """Return raw accelerometer readings from the IMU (body frame)."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.ACCELEROMETER_SENSOR
    )

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    """Return gyroscope (angular rate) readings from the IMU (body frame)."""
    return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

  # ---------------------------------------------------------------------------
  # Accessors.
  # ---------------------------------------------------------------------------

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
