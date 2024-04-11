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


def scenario_to_joint_scene(
    scenario: scenario_pb2.Scenario) -> sim_agents_submission_pb2.JointScene:
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

  Returns:
    A `JointScene` with trajectories copied from data.
  """
  trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
  # Slice by the required sim agents.
  sim_agent_ids = submission_specs.get_sim_agent_ids(scenario)
  trajectories = trajectories.gather_objects_by_id(
      tf.convert_to_tensor(sim_agent_ids))
  # Slice in time to only include steps after `current_time_index`.
  trajectories = trajectories.slice_time(submission_specs.CURRENT_TIME_INDEX+1)
  if trajectories.valid.shape[-1] != submission_specs.N_SIMULATION_STEPS:
    raise ValueError(
        'The Scenario used does not include the right number of time steps. '
        f'Expected: {submission_specs.N_SIMULATION_STEPS}, '
        f'Actual: {trajectories.valid.shape[-1]}.')
  # Iterate over objects and create `SimulatedTrajectory`s.
  simulated_trajectories = []
  for idx in range(trajectories.object_id.shape[0]):
    simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
        center_x=trajectories.x[idx],
        center_y=trajectories.y[idx],
        center_z=trajectories.z[idx],
        heading=trajectories.heading[idx],
        object_id=trajectories.object_id[idx],
    ))
  return sim_agents_submission_pb2.JointScene(
      simulated_trajectories=simulated_trajectories)


def joint_scene_to_trajectories(
    joint_scene: sim_agents_submission_pb2.JointScene,
    scenario: scenario_pb2.Scenario,
    use_log_validity: bool = False
) -> trajectory_utils.ObjectTrajectories:
  """Converts a JointScene and the relative Scenario into `ObjectTrajectories`.

  Args:
    joint_scene: A JointScene representing either logged or simulated data,
      corresponding to the same scenario ID as the provided logged `scenario`.
    scenario: The original scenario proto, used to infer static attributes (i.e.
      object types and box dimensions) from the last step of history
      (`CURRENT_TIME_INDEX` when 0-indexed). The history steps from this
      scenario are also prepended to the returned `Trajectories`, resulting in a
      total length of `submission_specs.N_FULL_SCENARIO_STEPS`.
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
  logged_trajectories_history = logged_trajectories_full.slice_time(
      start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1
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
      tf.convert_to_tensor, [sim_x, sim_y, sim_z, sim_heading, sim_ids])
  # Align objects from the logged scenario to the simulated one.
  logged_trajectories_history = (
      logged_trajectories_history.gather_objects_by_id(sim_ids))
  # Prepare the missing tensors: validity and box sizes.
  if use_log_validity:
    # When copying validity from the log, make sure objects are in the same
    # order as `sim_ids`.
    logged_trajectories_full = logged_trajectories_full.gather_objects_by_id(
        sim_ids)
    logged_trajectories_future = logged_trajectories_full.slice_time(
        start_index=submission_specs.CURRENT_TIME_INDEX + 1
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
