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
"""Collection of metrics used to evaluate Sim Agents Challenge submissions."""

from typing import List

from google.protobuf import text_format
import numpy as np
import tensorflow as tf

# copybara removed file resource import

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.wdl_limited.sim_agents_metrics import estimators
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features


def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
  """Loads the `SimAgentMetricsConfig` used for the challenge."""
  # pylint: disable=line-too-long
  # pyformat: disable
  config_path = '{pyglib_resource}waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_config.textproto'.format(pyglib_resource='')
  with open(config_path, 'r') as f:
    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    text_format.Parse(f.read(), config)
  return config


def compute_scenario_metrics_for_bundle(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    scenario: scenario_pb2.Scenario,
    scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts
) -> sim_agents_metrics_pb2.SimAgentMetrics:
  """Computes the scenario-level metrics for the given bundle."""
  # Computes the metric features for log and sim.
  log_features, sim_features = (
      metric_features.compute_scenario_rollouts_features(
          scenario, scenario_rollouts)
    )

  # ==== Average Displacement Error ====
  # This metric is not included in the scoring meta-metric, but we report it
  # to have a baseline comparison with existing Behaviour Prediction challenges.
  # We report both ADE (averaged over simulations and objects) and minADE
  # (averaged over objects, minimum over simulations).
  average_displacement_error = tf.reduce_mean(
      sim_features.average_displacement_error)
  min_average_displacement_error = tf.reduce_min(
      tf.reduce_mean(sim_features.average_displacement_error, axis=1))

  # ==== Dynamics features ====
  # Compute the log-likelihoods of speed features (first derivatives).
  # Note: For log_values we take only index=0 of the batch dimension to have
  # shape (n_objects, n_steps), as specified by
  # `log_likelihood_estimate_timeseries()`.
  linear_speed_log_likelihood = estimators.log_likelihood_estimate_timeseries(
      feature_config=config.linear_speed,
      log_values=log_features.linear_speed[0],
      sim_values=sim_features.linear_speed)
  angular_speed_log_likelihood = estimators.log_likelihood_estimate_timeseries(
      feature_config=config.angular_speed,
      log_values=log_features.angular_speed[0],
      sim_values=sim_features.angular_speed)
  # Get the log speed (linear and angular) validity. Since this is computed by
  # a delta between steps `i` and `i+1`, we verify that both of these are valid
  # (logical and).
  speed_validity = _logical_and_diff(log_features.valid[0], prepend_value=False)
  # The score is computed as the sum of the log-likelihoods, filtered by
  # validity. We exponentiate the result to get a score in the range [0,1].
  linear_speed_likelihood = tf.exp(_reduce_average_with_validity(
      linear_speed_log_likelihood, speed_validity))
  angular_speed_likelihood = tf.exp(_reduce_average_with_validity(
      angular_speed_log_likelihood, speed_validity))

  # Similarly, we compute likelihoods for acceleration features. This time,
  # we have to compute a double-step validity mask, because accelerations
  # involve the validity of `i`, `i+1`, `i+2`.
  linear_accel_log_likelihood = estimators.log_likelihood_estimate_timeseries(
      feature_config=config.linear_acceleration,
      log_values=log_features.linear_acceleration[0],
      sim_values=sim_features.linear_acceleration)
  angular_accel_log_likelihood = estimators.log_likelihood_estimate_timeseries(
      feature_config=config.angular_acceleration,
      log_values=log_features.angular_acceleration[0],
      sim_values=sim_features.angular_acceleration)
  acceleration_validity = _logical_and_diff(speed_validity, prepend_value=False)
  linear_accel_likelihood = tf.exp(_reduce_average_with_validity(
      linear_accel_log_likelihood, acceleration_validity))
  angular_accel_likelihood = tf.exp(_reduce_average_with_validity(
      angular_accel_log_likelihood, acceleration_validity))

  # Collision and distance to other objects. Again, aggregate over objects and
  # timesteps by summing the log-probabilities.
  collision_score = estimators.log_likelihood_estimate_scenario_level(
      feature_config=config.collision_indication,
      log_values=log_features.collision_indication[0],
      sim_values=sim_features.collision_indication
  )
  collision_likelihood = tf.exp(tf.reduce_mean(collision_score))

  distance_to_objects_log_likelihood = (
      estimators.log_likelihood_estimate_timeseries(
          feature_config=config.distance_to_nearest_object,
          log_values=log_features.distance_to_nearest_object[0],
          sim_values=sim_features.distance_to_nearest_object,
      )
  )
  distance_to_obj_likelihood = tf.exp(
      _reduce_average_with_validity(
          distance_to_objects_log_likelihood, log_features.valid[0]
      )
  )

  ttc_log_likelihood = estimators.log_likelihood_estimate_timeseries(
      feature_config=config.time_to_collision,
      log_values=log_features.time_to_collision[0],
      sim_values=sim_features.time_to_collision,
  )
  ttc_likelihood = tf.exp(
      _reduce_average_with_validity(ttc_log_likelihood, log_features.valid[0])
  )

  # Off-road and distance to road edge. Again, aggregate over objects and
  # timesteps by summing the log-probabilities.
  offroad_score = estimators.log_likelihood_estimate_scenario_level(
      feature_config=config.offroad_indication,
      log_values=log_features.offroad_indication[0],
      sim_values=sim_features.offroad_indication,
  )
  offroad_likelihood = tf.exp(tf.reduce_mean(offroad_score))

  # `distance_to_road_edge_log_likelihood` shape: (n_objects, n_steps).
  distance_to_road_edge_log_likelihood = (
      estimators.log_likelihood_estimate_timeseries(
          feature_config=config.distance_to_road_edge,
          log_values=log_features.distance_to_road_edge[0],
          sim_values=sim_features.distance_to_road_edge,
      )
  )
  distance_to_road_edge_likelihood = tf.exp(
      _reduce_average_with_validity(
          distance_to_road_edge_log_likelihood, log_features.valid[0]
      )
  )

  # ==== Meta-metric ====
  metametric = _compute_metametric(
      config,
      linear_speed_likelihood=linear_speed_likelihood,
      linear_accel_likelihood=linear_accel_likelihood,
      angular_speed_likelihood=angular_speed_likelihood,
      angular_accel_likelihood=angular_accel_likelihood,
      distance_to_nearest_object_likelihood=distance_to_obj_likelihood,
      collision_indication_likelihood=collision_likelihood,
      time_to_collision_likelihood=ttc_likelihood,
      distance_to_road_edge_likelihood=distance_to_road_edge_likelihood,
      offroad_indication_likelihood=offroad_likelihood,
  )

  return sim_agents_metrics_pb2.SimAgentMetrics(
      scenario_id=scenario.scenario_id,
      metametric=metametric,
      average_displacement_error=average_displacement_error,
      min_average_displacement_error=min_average_displacement_error,
      linear_speed_likelihood=linear_speed_likelihood.numpy(),
      linear_acceleration_likelihood=linear_accel_likelihood.numpy(),
      angular_speed_likelihood=angular_speed_likelihood.numpy(),
      angular_acceleration_likelihood=angular_accel_likelihood.numpy(),
      distance_to_nearest_object_likelihood=distance_to_obj_likelihood.numpy(),
      collision_indication_likelihood=collision_likelihood.numpy(),
      time_to_collision_likelihood=ttc_likelihood.numpy(),
      distance_to_road_edge_likelihood=distance_to_road_edge_likelihood.numpy(),
      offroad_indication_likelihood=offroad_likelihood.numpy(),
  )


def aggregate_scenario_metrics(
    all_scenario_metrics: List[sim_agents_metrics_pb2.SimAgentMetrics]
    ) -> sim_agents_metrics_pb2.SimAgentMetrics:
  """Aggregates the per-scenario metrics over the whole dataset."""
  msg_fields = [field[0].name for field in all_scenario_metrics[0].ListFields()]
  field_values = {field_name: [] for field_name in msg_fields}
  for scenario_metrics in all_scenario_metrics:
    for field_name in msg_fields:
      field_values[field_name].append(getattr(scenario_metrics, field_name))
  # Remove the scenario ID field.
  del field_values['scenario_id']
  # Average all the fields.
  field_values = {
      name: np.mean(values) for (name, values) in field_values.items()}
  return sim_agents_metrics_pb2.SimAgentMetrics(
      **field_values)


def _logical_and_diff(
    valid: tf.Tensor, prepend_value: bool = False) -> tf.Tensor:
  """Computes the 1-step logical and between the elements of a boolean tensor.

  To determine the validity of fields that are computed by deltas between steps
  (e.g. speed, acceleration), we determine that if any of the steps are invalid,
  the whole difference needs to be invalidated (i.e. logical_and).

  Args:
    valid: A tensor of shape (..., n_steps).
    prepend_value: The value to prepend to the result to maintain the original
      tensor shape.

  Returns:
    The 1-step logical and between elements of `valid. The resulting shape is
    still (..., n_steps), as the `prepend_value` is prepended to the whole
    result.
  """
  if valid.dtype != tf.bool:
    raise ValueError('The `valid` tensor must have boolean dtype. '
                     f'(Actual: {valid.dtype}).')
  prepend_shape = (*valid.shape[:-1], 1)
  prepend_tensor = tf.fill(prepend_shape, prepend_value)
  return tf.concat([
      prepend_tensor, tf.logical_and(valid[..., 1:], valid[..., :-1])
      ], axis=-1)


def _reduce_average_with_validity(
    tensor: tf.Tensor, validity: tf.Tensor) -> tf.Tensor:
  """Returns the tensor's average, only selecting valid items.

  Args:
    tensor: A float tensor of any shape.
    validity: A boolean tensor of the same shape as `tensor`.

  Returns:
    A float tensor of shape (1,), containing the average of the valid elements
    of `tensor`.
  """
  if tensor.shape != validity.shape:
    raise ValueError('Shapes of `tensor` and `validity` must be the same.'
                     f'(Actual: {tensor.shape}, {validity.shape}).')
  cond_sum = tf.reduce_sum(tf.where(validity, tensor, tf.zeros_like(tensor)))
  valid_sum = tf.reduce_sum(tf.cast(validity, tf.float32))
  return cond_sum / valid_sum


def _compute_metametric(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    linear_speed_likelihood: tf.Tensor,
    linear_accel_likelihood: tf.Tensor,
    angular_speed_likelihood: tf.Tensor,
    angular_accel_likelihood: tf.Tensor,
    distance_to_nearest_object_likelihood: tf.Tensor,
    collision_indication_likelihood: tf.Tensor,
    time_to_collision_likelihood: tf.Tensor,
    distance_to_road_edge_likelihood: tf.Tensor,
    offroad_indication_likelihood: tf.Tensor,
):
  """Computes the meta-metric aggregation."""
  return (
      # Dynamics features.
      config.linear_speed.metametric_weight * linear_speed_likelihood
      + config.linear_acceleration.metametric_weight * linear_accel_likelihood
      + config.angular_speed.metametric_weight * angular_speed_likelihood
      + config.angular_acceleration.metametric_weight * angular_accel_likelihood
      +
      # Distance to nearest object.
      config.distance_to_nearest_object.metametric_weight
      * distance_to_nearest_object_likelihood
      +
      # Collision indication.
      config.collision_indication.metametric_weight
      * collision_indication_likelihood
      +
      # Time-to-collision.
      config.time_to_collision.metametric_weight * time_to_collision_likelihood
      +
      # Distance to road edge.
      config.distance_to_road_edge.metametric_weight
      * distance_to_road_edge_likelihood
      +
      # Off-road indication.
      config.offroad_indication.metametric_weight
      * offroad_indication_likelihood
  )
