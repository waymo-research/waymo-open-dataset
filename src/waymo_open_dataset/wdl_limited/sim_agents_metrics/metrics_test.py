# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================

import dataclasses

from absl.testing import parameterized
import tensorflow as tf

from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics


_SimAgentMetricsConfig = sim_agents_metrics_pb2.SimAgentMetricsConfig
_FeatureConfig = _SimAgentMetricsConfig.FeatureConfig
_HistogramEstimate = _SimAgentMetricsConfig.HistogramEstimate
_BernoulliEstimate = _SimAgentMetricsConfig.BernoulliEstimate


class MetricsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (submission_specs.ChallengeType.SIM_AGENTS),
      (submission_specs.ChallengeType.SCENARIO_GEN),
  )
  def test_loads_challenge_config(
      self, challenge_type: submission_specs.ChallengeType
  ):
    config = metrics.load_metrics_config(challenge_type)
    self.assertIsInstance(config, _SimAgentMetricsConfig)

  @parameterized.parameters([
      (submission_specs.ChallengeType.SIM_AGENTS),
  ])
  def test_compute_scenario_metrics(
      self, challenge_type: submission_specs.ChallengeType
  ):
    test_config = sim_agents_test_utils.load_test_metrics_config()
    scenario = test_utils.get_womd_test_scenario()
    submission = sim_agents_test_utils.load_test_submission()
    log_features, sim_features = (
        metric_features.compute_scenario_rollouts_features(
            scenario, submission.scenario_rollouts[0], challenge_type
        )
    )

    bundle_metrics = metrics.compute_scenario_metrics_for_features_bundle(
        test_config,
        scenario.scenario_id,
        log_features,
        sim_features,
    )

    self.assertTrue(bundle_metrics.HasField('scenario_id'))
    self.assertNotAlmostEqual(bundle_metrics.average_displacement_error, 0.0)
    self.assertNotAlmostEqual(
        bundle_metrics.min_average_displacement_error, 0.0)
    # Likehoods should all be in the [0, 1] range.
    self.assertBetween(bundle_metrics.linear_speed_likelihood, 0.0, 1.0)
    self.assertBetween(bundle_metrics.linear_acceleration_likelihood, 0.0, 1.0)
    self.assertBetween(bundle_metrics.angular_speed_likelihood, 0.0, 1.0)
    self.assertBetween(bundle_metrics.angular_acceleration_likelihood, 0.0, 1.0)
    self.assertBetween(
        bundle_metrics.distance_to_nearest_object_likelihood, 0.0, 1.0)
    self.assertBetween(bundle_metrics.collision_indication_likelihood, 0.0, 1.0)
    self.assertBetween(bundle_metrics.time_to_collision_likelihood, 0.0, 1.0)
    self.assertBetween(
        bundle_metrics.distance_to_road_edge_likelihood, 0.0, 1.0
    )
    self.assertBetween(bundle_metrics.offroad_indication_likelihood, 0.0, 1.0)
    self.assertBetween(
        bundle_metrics.traffic_light_violation_likelihood, 0.0, 1.0
    )
    # The metametric should be higher than 0 (the maximum depends on the weights
    # inside the config).
    self.assertGreater(bundle_metrics.metametric, 0.0)
    self.assertBetween(bundle_metrics.simulated_collision_rate, 0.0, 1.0)
    self.assertBetween(bundle_metrics.simulated_offroad_rate, 0.0, 1.0)
    self.assertBetween(
        bundle_metrics.simulated_traffic_light_violation_rate, 0.0, 1.0
    )

  def test_filter_by_object_type_correctly_returns(self):
    config = sim_agents_test_utils.load_test_metrics_config()
    scenario = test_utils.get_womd_test_scenario()
    submission = sim_agents_test_utils.load_test_submission()
    log_features, sim_features = (
        metric_features.compute_scenario_rollouts_features(
            scenario, submission.scenario_rollouts[0],
            submission_specs.ChallengeType.SIM_AGENTS,
        )
    )
    # Test scenario contains 4 objects to evaluate, 1 of which is a pedestrian.
    # We want to modify the features that should be filtered by object type
    # (TTC) and check that the final value does not change.
    # Modify TTC for object idx=2 (pedestrian) to 0.0.
    ttc_mod = tf.where(
        tf.constant([False, False, True, False])[tf.newaxis, :, tf.newaxis],
        0.0, sim_features.time_to_collision)
    sim_features_mod = dataclasses.replace(
        sim_features,
        time_to_collision=ttc_mod,
    )
    bundle_metrics = metrics.compute_scenario_metrics_for_features_bundle(
        config, scenario.scenario_id, log_features,
        sim_features
    )
    bundle_metrics_mod = metrics.compute_scenario_metrics_for_features_bundle(
        config, scenario.scenario_id, log_features,
        sim_features_mod
    )
    self.assertAlmostEqual(
        bundle_metrics.time_to_collision_likelihood,
        bundle_metrics_mod.time_to_collision_likelihood,
        places=2,
    )

  def test_aggregate_scenario_metrics_returns_correctly(self):
    bundle_metrics = sim_agents_metrics_pb2.SimAgentMetrics(
        scenario_id='1234',
        metametric=0.1,
        average_displacement_error=0.2,
        min_average_displacement_error=0.3,
        linear_speed_likelihood=0.5,
        linear_acceleration_likelihood=0.5,
        angular_speed_likelihood=0.5,
        angular_acceleration_likelihood=0.5,
        distance_to_nearest_object_likelihood=0.0,
        collision_indication_likelihood=1.0,
        time_to_collision_likelihood=0.0,
        distance_to_road_edge_likelihood=1.0,
        offroad_indication_likelihood=1.0,
        traffic_light_violation_likelihood=1.0,
        simulated_collision_rate=0.5,
        simulated_offroad_rate=0.5,
    )
    bundle_metrics_for_dataset = [bundle_metrics] * 42

    dataset_metrics = metrics.aggregate_scenario_metrics(
        bundle_metrics_for_dataset)

    self.assertFalse(dataset_metrics.HasField('scenario_id'))
    # Since each field is the average of the 42 "fake" scenarios we just
    # repeated, each field needs to be the same as aggregated.
    self.assertAlmostEqual(dataset_metrics.average_displacement_error,
                           bundle_metrics.average_displacement_error)
    self.assertAlmostEqual(dataset_metrics.min_average_displacement_error,
                           bundle_metrics.min_average_displacement_error)
    self.assertAlmostEqual(dataset_metrics.linear_speed_likelihood,
                           bundle_metrics.linear_speed_likelihood)
    self.assertAlmostEqual(dataset_metrics.linear_acceleration_likelihood,
                           bundle_metrics.linear_acceleration_likelihood)
    self.assertAlmostEqual(dataset_metrics.angular_speed_likelihood,
                           bundle_metrics.angular_speed_likelihood)
    self.assertAlmostEqual(dataset_metrics.angular_acceleration_likelihood,
                           bundle_metrics.angular_acceleration_likelihood)
    self.assertAlmostEqual(
        dataset_metrics.distance_to_nearest_object_likelihood,
        bundle_metrics.distance_to_nearest_object_likelihood)
    self.assertAlmostEqual(dataset_metrics.collision_indication_likelihood,
                           bundle_metrics.collision_indication_likelihood)
    self.assertAlmostEqual(
        dataset_metrics.time_to_collision_likelihood,
        bundle_metrics.time_to_collision_likelihood,
    )
    self.assertAlmostEqual(
        dataset_metrics.distance_to_road_edge_likelihood,
        bundle_metrics.distance_to_road_edge_likelihood,
    )
    self.assertAlmostEqual(
        dataset_metrics.offroad_indication_likelihood,
        bundle_metrics.offroad_indication_likelihood,
    )
    self.assertAlmostEqual(
        dataset_metrics.traffic_light_violation_likelihood,
        bundle_metrics.traffic_light_violation_likelihood,
    )

  def test_aggregate_metrics_to_buckets_correctly_returns(self):
    config = sim_agents_test_utils.load_test_metrics_config()
    # We change the weight of the collision indication to test out if the
    # function is properly taking the weighted average.
    config.collision_indication.metametric_weight = 2.0
    test_metrics = sim_agents_metrics_pb2.SimAgentMetrics(
        metametric=0.1,
        average_displacement_error=0.2,
        min_average_displacement_error=0.3,
        linear_speed_likelihood=0.5,
        linear_acceleration_likelihood=0.5,
        angular_speed_likelihood=0.5,
        angular_acceleration_likelihood=0.5,
        distance_to_nearest_object_likelihood=0.0,
        collision_indication_likelihood=1.0,
        time_to_collision_likelihood=0.0,
        distance_to_road_edge_likelihood=1.0,
        offroad_indication_likelihood=1.0,
        traffic_light_violation_likelihood=1.0,
        simulated_collision_rate=0.5,
        simulated_offroad_rate=0.5,
        simulated_traffic_light_violation_rate=0.5,
    )

    bucketed_metrics = metrics.aggregate_metrics_to_buckets(
        config, test_metrics)
    self.assertProtoEquals(
        bucketed_metrics,
        sim_agents_metrics_pb2.SimAgentsBucketedMetrics(
            realism_meta_metric=0.1,
            min_ade=0.3,
            kinematic_metrics=0.5,
            interactive_metrics=0.5,
            map_based_metrics=1.0,
            simulated_collision_rate=0.5,
            simulated_offroad_rate=0.5,
            simulated_traffic_light_violation_rate=0.5,
        ),
    )


if __name__ == '__main__':
  tf.test.main()
