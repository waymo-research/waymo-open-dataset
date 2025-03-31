# Copyright 2023 The Waymo Open Dataset Authors.
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
# =============================================================================
"""Conversion utilities for the Sim Agents challenge submission."""

import tensorflow as tf

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.utils.sim_agents import submission_specs

_ChallengeType = submission_specs.ChallengeType


def _all_tensor_length_equal(
    trajectories: trajectory_utils.ObjectTrajectories,
) -> bool:
  """Checks if all the contained tensors have the same length (i.e. shape[0]).

  Args:
    trajectories: Object trajectory objects containing tensor attributes to be
      verified.

  Returns:
    A bool value indicating if all the tensors have the same length.
  """
  tensors = [
      trajectories.x,  # shape: (n_objects, n_steps)
      trajectories.y,  # shape: (n_objects, n_steps)
      trajectories.z,  # shape: (n_objects, n_steps)
      trajectories.heading,  # shape: (n_objects, n_steps)
      trajectories.length,  # shape: (n_objects, n_steps)
      trajectories.width,  # shape: (n_objects, n_steps)
      trajectories.height,  # shape: (n_objects, n_steps)
      trajectories.valid,  # shape: (n_objects, n_steps)
      trajectories.object_id,  # shape (n_objects,)
      trajectories.object_type,  # shape (n_objects,)
  ]
  # Extract the first dimension of each tensor.
  first_dims = [tf.shape(t)[0] for t in tensors]
  # Check if all first dimensions are the same.
  return tf.reduce_all([first_dims[0] == dim for dim in first_dims])


def scenario_to_joint_scene(
    scenario: scenario_pb2.Scenario,
    challenge_type: _ChallengeType = _ChallengeType.SIM_AGENTS,
) -> sim_agents_submission_pb2.JointScene:
  """Converts a WOMD Scenario proto into the submission format (`JointScene`).

  This function populates a JointScene with log-following agents. This can be
  used to standardise the interface of metrics and to provide a baseline
  benchmark.

  A `JointScene` requires objects to be fully valid over the whole simulation.
  This is not the case for the original Scenario, which can contain invalid
  states. This function will copy states irrespectively of their validity in
  the original Scenario.

  This function will not work for the public test set data, as it doesn't
  provide information about the future, which will lead to an invalid
  `JointScene` proto. This is only to be used for local validation on either
  the training or validation set.

  Args:
    scenario: The original Scenario proto from the Waymo Open Dataset.
    challenge_type: The challenge type to use.

  Returns:
    A `JointScene` with trajectories copied from data.
  """
  trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
  # Slice by the required sim agents.
  sim_agent_ids = submission_specs.get_sim_agent_ids(scenario, challenge_type)
  trajectories = trajectories.gather_objects_by_id(
      tf.convert_to_tensor(sim_agent_ids)
  )
  config = submission_specs.get_submission_config(challenge_type)
  # For sim agents, we only need the future trajectories.
  # For scenario gen, we need the full trajectories.
  if challenge_type == _ChallengeType.SIM_AGENTS:
    # Slice in time to only include steps after `current_time_index`.
    trajectories = trajectories.slice_time(config.current_time_index + 1)

  if trajectories.valid.shape[-1] != config.n_simulation_steps:
    raise ValueError(
        'The Scenario used does not include the right number of time steps. '
        f'Expected: {config.n_simulation_steps}, '
        f'Actual: {trajectories.valid.shape[-1]}.'
    )
  if not _all_tensor_length_equal(trajectories):
    raise ValueError(
        'The trajectories used do not have all the contained tensors with the'
        ' same length.'
    )
  # Iterate over objects and create `SimulatedTrajectory`s.
  simulated_trajectories = []
  for idx in range(trajectories.object_id.shape[0]):
    simulated_trajectories.append(
        sim_agents_submission_pb2.SimulatedTrajectory(
            center_x=trajectories.x[idx],
            center_y=trajectories.y[idx],
            center_z=trajectories.z[idx],
            heading=trajectories.heading[idx],
            object_id=trajectories.object_id[idx],
            length=trajectories.length[idx],
            width=trajectories.width[idx],
            height=trajectories.height[idx],
            object_type=trajectories.object_type[idx],
            valid=trajectories.valid[idx],
        )
    )
  return sim_agents_submission_pb2.JointScene(
      simulated_trajectories=simulated_trajectories
  )


def joint_scene_to_trajectories(
    joint_scene: sim_agents_submission_pb2.JointScene,
    scenario: scenario_pb2.Scenario,
    use_log_validity: bool = False,
) -> trajectory_utils.ObjectTrajectories:
  """Converts a JointScene and the relative Scenario into `ObjectTrajectories`.

  Note: This function is only used for the SIM_AGENTS challenge.

  Args:
    joint_scene: A JointScene representing either logged or simulated data,
      corresponding to the same scenario ID as the provided logged `scenario`.
    scenario: The original scenario proto, used to infer static attributes (i.e.
      object types and box dimensions) from the last step of history
      (`CURRENT_TIME_INDEX` when 0-indexed). The history steps from this
      scenario are also prepended to the returned `Trajectories`.
    use_log_validity: If True, copies the validity mask from the original
      scenario instead of assuming all the steps are valid. This is used to
      compute features for the logged scenario.

  Returns:
    An `ObjectTrajectories` containing the trajectories of all simulated objects
    in a scenario, with prepended history steps and inferred static dimensions
    and object types.
  """
  logged_trajectories_full = trajectory_utils.ObjectTrajectories.from_scenario(
      scenario
  )
  config = submission_specs.get_submission_config(_ChallengeType.SIM_AGENTS)
  logged_trajectories_history = logged_trajectories_full.slice_time(
      start_index=0, end_index=config.current_time_index + 1
  )
  # Extract states and object IDs from the simulated scene.
  sim_ids, sim_x, sim_y, sim_z, sim_heading = [], [], [], [], []
  for simulated_trajectory in joint_scene.simulated_trajectories:
    sim_x.append(simulated_trajectory.center_x)
    sim_y.append(simulated_trajectory.center_y)
    sim_z.append(simulated_trajectory.center_z)
    sim_heading.append(simulated_trajectory.heading)
    sim_ids.append(simulated_trajectory.object_id)
  # Convert to tensors.
  sim_x, sim_y, sim_z, sim_heading, sim_ids = map(
      tf.convert_to_tensor, [sim_x, sim_y, sim_z, sim_heading, sim_ids]
  )
  # Align objects from the logged scenario to the simulated one.
  logged_trajectories_history = (
      logged_trajectories_history.gather_objects_by_id(sim_ids)
  )
  # Prepare the missing tensors: validity and box sizes.
  if use_log_validity:
    # When copying validity from the log, make sure objects are in the same
    # order as `sim_ids`.
    logged_trajectories_full = logged_trajectories_full.gather_objects_by_id(
        sim_ids
    )
    logged_trajectories_future = logged_trajectories_full.slice_time(
        start_index=config.current_time_index + 1
    )
    sim_valid = logged_trajectories_future.valid
  else:
    sim_valid = tf.fill(sim_x.shape, True)

  sim_length = tf.repeat(
      logged_trajectories_history.length[:, -1, tf.newaxis],
      sim_x.shape[-1],
      axis=-1,
  )
  sim_width = tf.repeat(
      logged_trajectories_history.width[:, -1, tf.newaxis],
      sim_x.shape[-1],
      axis=-1,
  )
  sim_height = tf.repeat(
      logged_trajectories_history.height[:, -1, tf.newaxis],
      sim_x.shape[-1],
      axis=-1,
  )
  # Concatenate history and logged/simulated future.
  return trajectory_utils.ObjectTrajectories(
      x=tf.concat([logged_trajectories_history.x, sim_x], axis=-1),
      y=tf.concat([logged_trajectories_history.y, sim_y], axis=-1),
      z=tf.concat([logged_trajectories_history.z, sim_z], axis=-1),
      heading=tf.concat(
          [logged_trajectories_history.heading, sim_heading], axis=-1
      ),
      length=tf.concat(
          [logged_trajectories_history.length, sim_length], axis=-1
      ),
      width=tf.concat([logged_trajectories_history.width, sim_width], axis=-1),
      height=tf.concat(
          [logged_trajectories_history.height, sim_height], axis=-1
      ),
      valid=tf.concat([logged_trajectories_history.valid, sim_valid], axis=-1),
      object_id=sim_ids,
      object_type=logged_trajectories_history.object_type,
  )


def simulated_scenariogen_to_trajectories(
    joint_scene: sim_agents_submission_pb2.JointScene,
    scenario: scenario_pb2.Scenario,
    challenge_type: _ChallengeType,
) -> trajectory_utils.ObjectTrajectories:
  """Converts a JointScene into `ObjectTrajectories`.

  Note: This function is only used for the SCENARIO_GEN challenge.

  Args:
    joint_scene: A JointScene representing a simulated scenario for Scenario
      Generation. The input joint_scene is assumed to be obtained from
      `scenario_to_joint_scene()` above.
    scenario: The original scenario proto, only used for validity mask.
    challenge_type: The challenge type to use to read the submission config.

  Returns:
    An `ObjectTrajectories` containing the trajectories of all simulated objects
    in a scenario.
  """
  # Extract states, object IDs, dimensions, and object types from the simulated
  # scene.
  sim_ids = []
  sim_x = []
  sim_y = []
  sim_z = []
  sim_heading = []
  sim_length = []
  sim_width = []
  sim_height = []
  sim_object_type = []

  for simulated_trajectory in joint_scene.simulated_trajectories:
    sim_x.append(simulated_trajectory.center_x)
    sim_y.append(simulated_trajectory.center_y)
    sim_z.append(simulated_trajectory.center_z)
    sim_heading.append(simulated_trajectory.heading)
    sim_ids.append(simulated_trajectory.object_id)
    # For SCENARIO_GEN, we assume the length, width, height, valid, and
    # object_type must be provided, instead of optional.
    sim_length.append(simulated_trajectory.length)
    sim_width.append(simulated_trajectory.width)
    sim_height.append(simulated_trajectory.height)
    sim_object_type.append(simulated_trajectory.object_type)

  # Convert to tensors.
  to_tensor_func = tf.convert_to_tensor

  # shape: (n_objects, n_steps)
  sim_x, sim_y, sim_z, sim_heading, sim_length, sim_width, sim_height = map(
      to_tensor_func,
      [sim_x, sim_y, sim_z, sim_heading, sim_length, sim_width, sim_height],
  )

  # shape: (n_objects, steps=1)
  sim_ids, sim_object_type = map(to_tensor_func, [sim_ids, sim_object_type])

  logged_trajectories_full = trajectory_utils.ObjectTrajectories.from_scenario(
      scenario
  )
  # When copying validity from the log, make sure objects are in the same
  # order as `sim_ids`.
  logged_trajectories_full = logged_trajectories_full.gather_objects_by_id(
      sim_ids
  )
  scenario_gen_config = submission_specs.get_submission_config(challenge_type)
  log_valid_at_current_step = logged_trajectories_full.valid[
      :, scenario_gen_config.current_time_index
  ]
  # Repeat the validity mask for all the past, current, and future steps.
  sim_valid = tf.repeat(
      log_valid_at_current_step[:, tf.newaxis],
      sim_x.shape[-1],
      axis=-1,
  )

  return trajectory_utils.ObjectTrajectories(
      x=sim_x,
      y=sim_y,
      z=sim_z,
      heading=sim_heading,
      length=sim_length,
      width=sim_width,
      height=sim_height,
      valid=sim_valid,
      object_id=sim_ids,
      object_type=sim_object_type,
  )
