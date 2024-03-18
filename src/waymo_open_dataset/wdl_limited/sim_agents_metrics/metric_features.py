# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================
"""Simulation features used for sim agent metrics."""

from __future__ import annotations

import collections
import dataclasses

import tensorflow as tf

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.utils.sim_agents import converters
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import interaction_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import map_metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import trajectory_features


@dataclasses.dataclass(frozen=True)
class MetricFeatures:
  """Collection of features used to compute sim-agent metrics.

  These features may be a function of simulated data (e.g. dynamics and
  collisions), logged data (e.g. displacement) and map features (e.g. offroad).

  This class can be used to represent both features coming from the original
  Scenario and features from simulation. The samples dimension is set
  accordingly depending on the source (n_samples=1 for log and n_samples=32 for
  simulation).

  Some of the features are computed in 3D (x/y/z) to have better consistency
  with the original data and making these metrics more suitable for future
  updates.

  Attributes:
    object_id: A tensor of shape (n_objects,), containing the integer IDs of all
      the evaluated objects. The object_id tensor is not batched because all the
      objects need to be consistent over samples for proper evaluation.
    valid: Boolean tensor of shape (n_samples, n_objects, n_steps), identifying
      which objects are valid over time. This is used to filter the features
      when computing metrics.
    average_displacement_error: Per-object average (over time) displacement
      error compared to the logged trajectory. Shape: (n_samples, n_objects).
    linear_speed: Linear speed in 3D computed as the 1-step difference between
      trajectory points. Shape: (n_samples, n_objects, n_steps).
    linear_acceleration: Linear acceleration in 3D computed as the 1-step
      difference between speeds of objects.
      Shape: (n_samples, n_objects, n_steps).
    angular_speed: Angular speed computed as the 1-step difference in heading.
      Shape: (n_samples, n_objects, n_steps).
    angular_acceleration: Angular acceleration computed as the 1-step difference
      in angular_speed. Shape: (n_samples, n_objects, n_steps).
    distance_to_nearest_object: Signed distance (in meters) to the nearest
      object in the scene. Shape: (n_samples, n_objects, n_steps).
    collision_per_step: Boolean tensor indicating whether the object collided,
      with any other object. Shape: (n_samples, n_objects, n_steps).
    time_to_collision: Time (in seconds) before the object collides with the
      object it is following (if it exists), assuming constant speeds.
      Shape: (n_samples, n_objects, n_steps).
    distance_to_road_edge: Signed distance (in meters) to the nearest road edge
      in the scene. Shape: (n_samples, n_objects, n_steps).
    offroad_per_step: Boolean tensor indicating whether the object went
      off-road. Shape: (n_samples, n_objects, n_steps).
  """
  object_id: tf.Tensor
  valid: tf.Tensor
  average_displacement_error: tf.Tensor
  linear_speed: tf.Tensor
  linear_acceleration: tf.Tensor
  angular_speed: tf.Tensor
  angular_acceleration: tf.Tensor
  distance_to_nearest_object: tf.Tensor
  collision_per_step: tf.Tensor
  time_to_collision: tf.Tensor
  distance_to_road_edge: tf.Tensor
  offroad_per_step: tf.Tensor


def compute_metric_features(
    scenario: scenario_pb2.Scenario,
    joint_scene: sim_agents_submission_pb2.JointScene,
    use_log_validity: bool = False
) -> MetricFeatures:
  """Computes features for a single scene.

  Args:
    scenario: The `Scenario` loaded from WOMD.
    joint_scene: Single sample of the predicted scene starting from scenario's
      initial conditions.
    use_log_validity: If True, copies the validity mask from the original
      scenario instead of assuming all the steps are valid. This is used to
      compute features for the logged scenario.

  Returns:
    A `MetricFeatures` containing all the features.
  """
  # Extract `ObjectTrajectories` object from the joint scene, prepending the
  # history from the original scenario. These composite trajectories are used to
  # compute dynamics features, which require a few steps of context.
  simulated_trajectories = converters.joint_scene_to_trajectories(
      joint_scene, scenario, use_log_validity=use_log_validity)
  # Extract `ObjectTrajectories` from the original scenario, used for
  # log-comparison metrics (i.e. displacement error). These also need to be
  # aligned to the simulated trajectories.
  logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(
      scenario)
  logged_trajectories = logged_trajectories.gather_objects_by_id(
      simulated_trajectories.object_id)
  # From the simulated trajectories, just select the subset of objects that
  # needs evaluation.
  evaluated_sim_agent_ids = tf.convert_to_tensor(
      submission_specs.get_evaluation_sim_agent_ids(scenario)
  )
  evaluated_trajectories = simulated_trajectories.gather_objects_by_id(
      evaluated_sim_agent_ids
  )
  # Re-order simulated trajectories, so that evaluated objects appear first,
  # in a particular order.
  non_evaluated_sim_agent_ids = set(
      simulated_trajectories.object_id.numpy()
  ) - set(evaluated_sim_agent_ids.numpy())
  reordered_all_simulated_agent_ids = tf.constant(
      list(evaluated_sim_agent_ids.numpy()) + list(non_evaluated_sim_agent_ids)
  )
  simulated_trajectories = simulated_trajectories.gather_objects_by_id(
      reordered_all_simulated_agent_ids
  )
  # Prune logged trajectories to those that will be evaluated.
  evaluated_logged_trajectories = logged_trajectories.gather_objects_by_id(
      evaluated_sim_agent_ids
  )

  # Validity bit mask.
  if use_log_validity:
    validity_mask = evaluated_logged_trajectories.valid
  else:
    validity_mask = evaluated_trajectories.valid
  # Slice in time to reduce to `submission_specs.N_SIMULATION_STEPS` steps.
  validity_mask = validity_mask[:, submission_specs.CURRENT_TIME_INDEX+1:]

  # Average displacement error (ADE) in 3D. Before averaging over time, the
  # invalid states in the log need to be properly handled.
  displacement_error = trajectory_features.compute_displacement_error(
      evaluated_trajectories.x, evaluated_trajectories.y,
      evaluated_trajectories.z, evaluated_logged_trajectories.x,
      evaluated_logged_trajectories.y, evaluated_logged_trajectories.z)
  object_valid_steps = tf.reduce_sum(
      tf.cast(evaluated_logged_trajectories.valid, tf.float32), axis=1)
  ade = tf.reduce_sum(
      tf.where(evaluated_logged_trajectories.valid, displacement_error, 0.0),
      axis=1) / object_valid_steps

  # Kinematics-related features, i.e. speed and acceleration, both linear and
  # angular. These feature are computed as finite differences of the objects
  # position, which makes the first step invalid. We prepend the history steps
  # so that this first simulation step has a valid difference too.
  linear_speed, linear_accel, angular_speed, angular_accel = (
      trajectory_features.compute_kinematic_features(
          evaluated_trajectories.x,
          evaluated_trajectories.y,
          evaluated_trajectories.z,
          evaluated_trajectories.heading,
          seconds_per_step=submission_specs.STEP_DURATION_SECONDS))
  # Removes the data corresponding to the history time interval.
  linear_speed, linear_accel, angular_speed, angular_accel = (
      map(lambda t: t[:, submission_specs.CURRENT_TIME_INDEX+1:],
          [linear_speed, linear_accel, angular_speed, angular_accel])
  )

  # Collision and distances to objects.
  evaluated_object_mask = tf.reduce_any(
      # `evaluated_sim_agents` shape: (n_evaluated_objects,).
      # `simulated_trajectories.object_id` shape: (n_objects,).
      evaluated_sim_agent_ids[:, tf.newaxis]
      == simulated_trajectories.object_id,
      axis=0,
  )
  # Interactive features are computed between all simulated objects, but only
  # scored for evaluated objects.
  distances_to_objects = (
      interaction_features.compute_distance_to_nearest_object(
          center_x=simulated_trajectories.x,
          center_y=simulated_trajectories.y,
          center_z=simulated_trajectories.z,
          length=simulated_trajectories.length,
          width=simulated_trajectories.width,
          height=simulated_trajectories.height,
          heading=simulated_trajectories.heading,
          valid=simulated_trajectories.valid,
          evaluated_object_mask=evaluated_object_mask
          ))
  # Slice in time, as `simulated_trajectories` also include the history steps.
  distances_to_objects = (
      distances_to_objects[:, submission_specs.CURRENT_TIME_INDEX+1:])
  # Shape: (n_evaluated_objects, n_steps).
  is_colliding_per_step = tf.less(
      distances_to_objects, interaction_features.COLLISION_DISTANCE_THRESHOLD)

  times_to_collision = (
      interaction_features.compute_time_to_collision_with_object_in_front(
          center_x=simulated_trajectories.x,
          center_y=simulated_trajectories.y,
          length=simulated_trajectories.length,
          width=simulated_trajectories.width,
          heading=simulated_trajectories.heading,
          valid=simulated_trajectories.valid,
          evaluated_object_mask=evaluated_object_mask,
          seconds_per_step=submission_specs.STEP_DURATION_SECONDS,
      )
  )
  times_to_collision = times_to_collision[
      :, submission_specs.CURRENT_TIME_INDEX + 1 :
  ]

  # Roadgraph features.
  road_edges = []
  for map_feature in scenario.map_features:
    if map_feature.HasField('road_edge'):
      road_edges.append(map_feature.road_edge.polyline)
  distances_to_road_edge = map_metric_features.compute_distance_to_road_edge(
      center_x=simulated_trajectories.x,
      center_y=simulated_trajectories.y,
      center_z=simulated_trajectories.z,
      length=simulated_trajectories.length,
      width=simulated_trajectories.width,
      height=simulated_trajectories.height,
      heading=simulated_trajectories.heading,
      valid=simulated_trajectories.valid,
      evaluated_object_mask=evaluated_object_mask,
      road_edge_polylines=road_edges,
  )
  distances_to_road_edge = distances_to_road_edge[
      :, submission_specs.CURRENT_TIME_INDEX + 1 :
  ]
  is_offroad_per_step = tf.greater(
      distances_to_road_edge, map_metric_features.OFFROAD_DISTANCE_THRESHOLD
  )

  # Pack into `MetricFeatures`, also adding a batch dimension of 1 (except for
  # `object_id`).
  return MetricFeatures(
      object_id=evaluated_trajectories.object_id,
      valid=validity_mask[tf.newaxis],
      average_displacement_error=ade[tf.newaxis],
      linear_speed=linear_speed[tf.newaxis],
      linear_acceleration=linear_accel[tf.newaxis],
      angular_speed=angular_speed[tf.newaxis],
      angular_acceleration=angular_accel[tf.newaxis],
      distance_to_nearest_object=distances_to_objects[tf.newaxis],
      collision_per_step=is_colliding_per_step[tf.newaxis],
      time_to_collision=times_to_collision[tf.newaxis],
      distance_to_road_edge=distances_to_road_edge[tf.newaxis],
      offroad_per_step=is_offroad_per_step[tf.newaxis],
  )


def compute_scenario_rollouts_features(
    scenario: scenario_pb2.Scenario,
    scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts
) -> tuple[MetricFeatures, MetricFeatures]:
  """Computes the metrics features for both logged and simulated scenarios.

  Args:
    scenario: The `Scenario` loaded from WOMD.
    scenario_rollouts: The collection of joint scenes from simulation.

  Returns:
    Two `MetricFeatures`, the first one from logged data with n_samples=1 and
    the second from simulation with n_samples=`submission_specs.N_ROLLOUTS`.
  """
  log_joint_scene = converters.scenario_to_joint_scene(scenario)
  log_features = compute_metric_features(
      scenario, log_joint_scene, use_log_validity=True)

  # Aggregate the different parallel simulations.
  features_fields = [field.name for field in dataclasses.fields(MetricFeatures)]
  features_fields.remove('object_id')
  sim_features = collections.defaultdict(list)
  for joint_scene in scenario_rollouts.joint_scenes:
    rollout_features = compute_metric_features(scenario, joint_scene)
    if tf.reduce_any(log_features.object_id != rollout_features.object_id):
      raise ValueError('Misaligned object IDs for evaluation.')
    for field in features_fields:
      sim_features[field].append(getattr(rollout_features, field))

  # Concatenate and generate `MetricFeatures`.
  for field in features_fields:
    sim_features[field] = tf.concat(sim_features[field], axis=0)

  sim_features = MetricFeatures(
      **sim_features, object_id=log_features.object_id)
  return log_features, sim_features
