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

import random

import tensorflow as tf

from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils.sim_agents import converters
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics


_SimAgentMetricsConfig = sim_agents_metrics_pb2.SimAgentMetricsConfig
_FeatureConfig = _SimAgentMetricsConfig.FeatureConfig
_HistogramEstimate = _SimAgentMetricsConfig.HistogramEstimate
_BernoulliEstimate = _SimAgentMetricsConfig.BernoulliEstimate


class MetricsTest(tf.test.TestCase):

  def test_loads_challenge_config(self):
    config = metrics.load_metrics_config()
    self.assertIsInstance(config, _SimAgentMetricsConfig)

  def test_likelihoods_invariant_to_submission_object_id_permutation(self):
    scenario = test_utils.get_womd_test_scenario()
    test_config = (
        sim_agents_test_utils.load_identity_function_test_metrics_config()
    )

    # Convert the ground truth scenario into a 1-rollout submission.
    log_joint_scene = converters.scenario_to_joint_scene(scenario)
    submission_scenario_rollouts = sim_agents_submission_pb2.ScenarioRollouts(
        joint_scenes=[log_joint_scene]
    )
    # Permute the order of object IDs in submission vs. in GT scenario.
    permuted_simulated_trajectories = list(
        submission_scenario_rollouts.joint_scenes[0].simulated_trajectories
    )
    random.shuffle(permuted_simulated_trajectories)
    permuted_submission_scenario_rollouts = (
        sim_agents_submission_pb2.ScenarioRollouts(
            joint_scenes=[
                sim_agents_submission_pb2.JointScene(
                    simulated_trajectories=permuted_simulated_trajectories
                )
            ]
        )
    )

    # Compute likelihood metrics for the single rollout (logged vs. logged).
    bundle_metrics = metrics.compute_scenario_metrics_for_bundle(
        test_config, scenario, permuted_submission_scenario_rollouts
    )

    self.assertEqual(bundle_metrics.scenario_id, '637f20cafde22ff8')
    self.assertAlmostEqual(bundle_metrics.metametric, 1.0, places=3)
    self.assertAlmostEqual(bundle_metrics.average_displacement_error, 0.0)
    self.assertAlmostEqual(
        bundle_metrics.linear_speed_likelihood, 0.999, places=2
    )
    self.assertAlmostEqual(
        bundle_metrics.linear_acceleration_likelihood, 0.999, places=2
    )
    self.assertAlmostEqual(
        bundle_metrics.angular_speed_likelihood, 0.999, places=2
    )
    self.assertAlmostEqual(
        bundle_metrics.angular_acceleration_likelihood, 0.999, places=2
    )
    self.assertAlmostEqual(
        bundle_metrics.distance_to_nearest_object_likelihood, 0.999, places=2
    )
    self.assertAlmostEqual(
        bundle_metrics.collision_indication_likelihood, 0.999, places=2
    )
    self.assertAlmostEqual(
        bundle_metrics.time_to_collision_likelihood, 0.999, places=2
    )
    self.assertAlmostEqual(
        bundle_metrics.distance_to_road_edge_likelihood, 0.999, places=2
    )
    self.assertAlmostEqual(
        bundle_metrics.offroad_indication_likelihood, 0.999, places=2
    )
    self.assertAlmostEqual(bundle_metrics.min_average_displacement_error, 0.0)

  def test_compute_scenario_metrics(self):
    scenario = test_utils.get_womd_test_scenario()
    submission = sim_agents_test_utils.load_test_submission()
    test_config = sim_agents_test_utils.load_test_metrics_config()

    bundle_metrics = metrics.compute_scenario_metrics_for_bundle(
        test_config, scenario, submission.scenario_rollouts[0])

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
    # The metametric should be higher than 0 (the maximum depends on the weights
    # inside the config).
    self.assertGreater(bundle_metrics.metametric, 0.0)

  def test_aggregate_scenario_metrics_returns_correctly(self):
    scenario = test_utils.get_womd_test_scenario()
    submission = sim_agents_test_utils.load_test_submission()
    test_config = sim_agents_test_utils.load_test_metrics_config()
    bundle_metrics = metrics.compute_scenario_metrics_for_bundle(
        test_config, scenario, submission.scenario_rollouts[0])
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

  def test_aggregate_metrics_to_buckets_correctly_returns(self):
    config = sim_agents_test_utils.load_test_metrics_config()
    # We change the weight of the collision indication to test out if the
    # function is properly taking the weighted average.
    config.collision_indication.metametric_weight = 2.0
    test_metrics = sim_agents_metrics_pb2.SimAgentMetrics(
        metametric=0.1, average_displacement_error=0.2,
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
    )

    bucketed_metrics = metrics.aggregate_metrics_to_buckets(
        config, test_metrics)
    self.assertProtoEquals(
        bucketed_metrics,
        sim_agents_metrics_pb2.SimAgentsBucketedMetrics(
            realism_meta_metric=0.1, min_ade=0.3,
            kinematic_metrics=0.5, interactive_metrics=0.5,
            map_based_metrics=1.0
        ))

if __name__ == '__main__':
  tf.test.main()
