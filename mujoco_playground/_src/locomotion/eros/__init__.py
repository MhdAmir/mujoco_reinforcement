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
"""EROS humanoid locomotion environments (IMU-only sensor suite)."""

from mujoco_playground._src.locomotion.eros import eros_constants as constants
from mujoco_playground._src.locomotion.eros.base import ErosEnv, get_assets
from mujoco_playground._src.locomotion.eros.joystick import Joystick, default_config
from mujoco_playground._src.locomotion.eros.randomize import domain_randomize

__all__ = [
    "constants",
    "ErosEnv",
    "get_assets",
    "Joystick",
    "default_config",
    "domain_randomize",
]
