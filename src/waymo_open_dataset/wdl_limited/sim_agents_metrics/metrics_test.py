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

import tensorflow as tf

from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.utils import test_utils
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

if __name__ == '__main__':
  tf.test.main()
