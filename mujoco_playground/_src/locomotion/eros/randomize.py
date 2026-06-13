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
"""Domain randomization for the EROS humanoid environment.

Randomises physical parameters that are uncertain on the real robot:
* Floor friction
* Joint frictionloss / armature / damping
* Actuator gains (leg joints only)
* Body masses and centres of mass
* Initial joint positions (qpos0)

The randomization is applied per-environment in a vectorised (vmap) fashion,
exactly as in the Wolfgang implementation.
"""

import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1   # "base_link" is body index 1 (index 0 is the world body)

# Number of leg joints / actuators that the policy controls.
NUM_LEG_DOFS = 12


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Return a batched mjx.Model with randomised physics parameters.

  Args:
    model: The base mjx.Model (not yet batched).
    rng:   A JAX random key array of shape (N, 2), one key per environment.

  Returns:
    (randomised_model, in_axes): randomised_model is a batched mjx.Model;
    in_axes marks which leaves are batched (axis=0) versus shared (None).
  """

  def rand_dynamics(rng, model):
    # Floor friction: uniform in [0.4, 1.0].
    rng, key = jax.random.split(rng)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        jax.random.uniform(key, minval=0.4, maxval=1.0)
    )

    # Static friction loss on leg joints: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[14:14 + NUM_LEG_DOFS] * (
        jax.random.uniform(key, shape=(NUM_LEG_DOFS,), minval=0.9, maxval=1.1)
    )
    dof_frictionloss = model.dof_frictionloss.at[14:14 + NUM_LEG_DOFS].set(
        frictionloss
    )

    # Armature on leg joints: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[14:14 + NUM_LEG_DOFS] * (
        jax.random.uniform(key, shape=(NUM_LEG_DOFS,), minval=1.0, maxval=1.05)
    )
    dof_armature = model.dof_armature.at[14:14 + NUM_LEG_DOFS].set(armature)

    # Damping on leg joints: *U(0.95, 1.05).
    rng, key = jax.random.split(rng)
    damping = model.dof_damping[14:14 + NUM_LEG_DOFS] * (
        jax.random.uniform(
            key, shape=(NUM_LEG_DOFS,), minval=0.95, maxval=1.05
        )
    )
    dof_damping = model.dof_damping.at[14:14 + NUM_LEG_DOFS].set(damping)

    # Actuator gains on leg actuators: *U(0.5, 5.0).
    rng, key = jax.random.split(rng)
    noise = jax.random.uniform(
        key, shape=(NUM_LEG_DOFS,), minval=0.5, maxval=5.0
    )
    actuator_gain = model.actuator_gainprm.at[:NUM_LEG_DOFS, 0].set(
        model.actuator_gainprm[:NUM_LEG_DOFS, 0] * noise
    )
    actuator_bias = model.actuator_biasprm.at[:NUM_LEG_DOFS, 1].set(
        model.actuator_biasprm[:NUM_LEG_DOFS, 1] * noise
    )

    # Scale all body masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add extra mass to torso: +U(-1.0, 1.0) kg.
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Randomise torso CoM: +U(-0.07, 0.07) m.
    rng, key = jax.random.split(rng)
    com_offset = jax.random.uniform(
        key, shape=(3,), minval=-0.07, maxval=0.07
    )
    body_com = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + com_offset
    )

    # Randomise all body CoMs: +U(-0.01, 0.01) m.
    rng, key = jax.random.split(rng)
    com_offset = jax.random.uniform(
        key, shape=(model.nbody, 3), minval=-0.01, maxval=0.01
    )
    body_com = body_com.at[:].set(body_com + com_offset)

    # Jitter leg joint home positions: +U(-0.05, 0.05) rad.
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[15:15 + NUM_LEG_DOFS].set(
        qpos0[15:15 + NUM_LEG_DOFS]
        + jax.random.uniform(
            key, shape=(NUM_LEG_DOFS,), minval=-0.05, maxval=0.05
        )
    )

    return (
        geom_friction,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gain,
        actuator_bias,
        body_mass,
        body_com,
        qpos0,
    )

  rand_dynamics_vmap = jax.vmap(rand_dynamics, in_axes=(0, None))
  (
      friction,
      frictionloss,
      armature,
      damping,
      actuator_gain,
      actuator_bias,
      body_mass,
      body_com,
      qpos0,
  ) = rand_dynamics_vmap(rng, model)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
      "body_mass": 0,
      "body_ipos": 0,
      "qpos0": 0,
  })

  model = model.tree_replace({
      "geom_friction": friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "dof_damping": damping,
      "actuator_gainprm": actuator_gain,
      "actuator_biasprm": actuator_bias,
      "body_mass": body_mass,
      "body_ipos": body_com,
      "qpos0": qpos0,
  })

  return model, in_axes
