# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================
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
from waymo_open_dataset.wdl_limited.sim_agents_metrics import trajectory_features

_METRIC_FIELD_NAMES_BY_BUCKET = {
    'kinematic': [
        'linear_speed', 'linear_acceleration',
        'angular_speed', 'angular_acceleration',
    ],
    'interactive': [
        'distance_to_nearest_object', 'collision_indication',
        'time_to_collision',
    ],
    'map_based': [
        'distance_to_road_edge', 'offroad_indication'
    ]
}
_METRIC_FIELD_NAMES = (
    _METRIC_FIELD_NAMES_BY_BUCKET['kinematic'] +
    _METRIC_FIELD_NAMES_BY_BUCKET['interactive'] +
    _METRIC_FIELD_NAMES_BY_BUCKET['map_based']
)


def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
  """Loads the `SimAgentMetricsConfig` used for the challenge."""
  # pylint: disable=line-too-long
  # pyformat: disable
  config_path = '{pyglib_resource}waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2024_config.textproto'.format(pyglib_resource='')
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
  # a delta between steps `i-1` and `i+1`, we verify that both of these are
  # valid (logical and).
  speed_validity, acceleration_validity = (
      trajectory_features.compute_kinematic_validity(log_features.valid[0])
  )
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
  linear_accel_likelihood = tf.exp(_reduce_average_with_validity(
      linear_accel_log_likelihood, acceleration_validity))
  angular_accel_likelihood = tf.exp(_reduce_average_with_validity(
      angular_accel_log_likelihood, acceleration_validity))

  # Collision likelihood is computed by aggregating in time. For invalid objects
  # in the logged scenario, we need to filter possible collisions in simulation.
  # `sim_collision_indication` shape: (n_samples, n_objects).
  sim_collision_indication = tf.reduce_any(
      tf.where(log_features.valid, sim_features.collision_per_step, False),
      axis=2)
  log_collision_indication = tf.reduce_any(
      tf.where(log_features.valid, log_features.collision_per_step, False),
      axis=2)

  # Collision and distance to other objects. Again, aggregate over objects and
  # timesteps by summing the log-probabilities.
  collision_score = estimators.log_likelihood_estimate_scenario_level(
      feature_config=config.collision_indication,
      log_values=log_collision_indication[0],
      sim_values=sim_collision_indication
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
  # `sim_offroad_indication` shape: (n_samples, n_objects).
  sim_offroad_indication = tf.reduce_any(
      tf.where(log_features.valid, sim_features.offroad_per_step, False),
      axis=2)
  log_offroad_indication = tf.reduce_any(
      tf.where(log_features.valid, log_features.offroad_per_step, False),
      axis=2)
  offroad_score = estimators.log_likelihood_estimate_scenario_level(
      feature_config=config.offroad_indication,
      log_values=log_offroad_indication[0],
      sim_values=sim_offroad_indication,
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

  # ==== Simulated collision and offroad rates ====
  simulated_collision_rate = tf.reduce_sum(
      # `sim_collision_indication` shape: (n_samples, n_objects).
      tf.cast(sim_collision_indication, tf.int32)
  ) / tf.reduce_sum(tf.ones_like(sim_collision_indication, dtype=tf.int32))
  simulated_offroad_rate = tf.reduce_sum(
      # `sim_offroad_indication` shape: (n_samples, n_objects).
      tf.cast(sim_offroad_indication, tf.int32)
  ) / tf.reduce_sum(tf.ones_like(sim_offroad_indication, dtype=tf.int32))

  # ==== Meta-metric ====
  likelihood_metrics = {
      'linear_speed_likelihood': linear_speed_likelihood.numpy(),
      'linear_acceleration_likelihood': linear_accel_likelihood.numpy(),
      'angular_speed_likelihood': angular_speed_likelihood.numpy(),
      'angular_acceleration_likelihood': angular_accel_likelihood.numpy(),
      'distance_to_nearest_object_likelihood': (
          distance_to_obj_likelihood.numpy()
      ),
      'collision_indication_likelihood': collision_likelihood.numpy(),
      'time_to_collision_likelihood': ttc_likelihood.numpy(),
      'distance_to_road_edge_likelihood': (
          distance_to_road_edge_likelihood.numpy()
      ),
      'offroad_indication_likelihood': offroad_likelihood.numpy(),
  }

  metametric = _compute_metametric(
      config, sim_agents_metrics_pb2.SimAgentMetrics(**likelihood_metrics))

  return sim_agents_metrics_pb2.SimAgentMetrics(
      scenario_id=scenario.scenario_id,
      metametric=metametric,
      average_displacement_error=average_displacement_error,
      min_average_displacement_error=min_average_displacement_error,
      simulated_collision_rate=simulated_collision_rate.numpy(),
      simulated_offroad_rate=simulated_offroad_rate.numpy(),
      **likelihood_metrics,
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


def aggregate_metrics_to_buckets(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    metrics: sim_agents_metrics_pb2.SimAgentMetrics
) -> sim_agents_metrics_pb2.SimAgentsBucketedMetrics:
  """Aggregates metrics into buckets for better readability."""
  bucketed_metrics = {}
  for bucket_name, fields_in_bucket in _METRIC_FIELD_NAMES_BY_BUCKET.items():
    weighted_metric, weights_sum = 0.0, 0.0
    for field_name in fields_in_bucket:
      likelihood_field_name = field_name + '_likelihood'
      weight = getattr(config, field_name).metametric_weight
      metric_score = getattr(metrics, likelihood_field_name)
      weighted_metric += weight * metric_score
      weights_sum += weight
    if weights_sum == 0:
      raise ValueError('The bucket\'s weight sum is zero. Check your metrics'
                       ' config.')
    bucketed_metrics[bucket_name] = weighted_metric / weights_sum

  return sim_agents_metrics_pb2.SimAgentsBucketedMetrics(
      realism_meta_metric=metrics.metametric,
      kinematic_metrics=bucketed_metrics['kinematic'],
      interactive_metrics=bucketed_metrics['interactive'],
      map_based_metrics=bucketed_metrics['map_based'],
      min_ade=metrics.min_average_displacement_error,
      simulated_collision_rate=metrics.simulated_collision_rate,
      simulated_offroad_rate=metrics.simulated_offroad_rate,
  )


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
    metrics: sim_agents_metrics_pb2.SimAgentMetrics,
):
  """Computes the meta-metric aggregation."""
  metametric = 0.0
  for field_name in _METRIC_FIELD_NAMES:
    likelihood_field_name = field_name + '_likelihood'
    weight = getattr(config, field_name).metametric_weight
    metric_score = getattr(metrics, likelihood_field_name)
    metametric += weight * metric_score
  return metametric
