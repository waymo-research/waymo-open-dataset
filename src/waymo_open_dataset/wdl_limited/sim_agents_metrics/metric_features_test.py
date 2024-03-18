# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================

import tensorflow as tf

from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils.sim_agents import converters
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.wdl_limited.sim_agents_metrics import interaction_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import map_metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features


class MetricFeaturesTest(tf.test.TestCase):

  def test_features_from_log_scenario_trajectories(self):
    scenario = test_utils.get_womd_test_scenario()
    joint_scene = converters.scenario_to_joint_scene(scenario)
    simulated_features = (
        metric_features.compute_metric_features(
            scenario, joint_scene, use_log_validity=False))

    batch_size, num_agents, num_steps = 1, 4, 80
    self.assertEqual(simulated_features.object_id.shape, (num_agents,))
    self.assertEqual(simulated_features.valid.shape,
                     (batch_size, num_agents, num_steps))
    self.assertTrue(tf.reduce_all(simulated_features.valid))
    self.assertEqual(
        simulated_features.average_displacement_error.shape,
        (batch_size, num_agents))
    self.assertEqual(simulated_features.linear_speed.shape,
                     (batch_size, num_agents, num_steps))
    self.assertEqual(simulated_features.linear_acceleration.shape,
                     (batch_size, num_agents, num_steps))
    self.assertEqual(simulated_features.angular_speed.shape,
                     (batch_size, num_agents, num_steps))
    self.assertEqual(simulated_features.angular_acceleration.shape,
                     (batch_size, num_agents, num_steps))
    self.assertEqual(simulated_features.distance_to_nearest_object.shape,
                     (batch_size, num_agents, num_steps))
    self.assertEqual(simulated_features.collision_per_step.shape,
                     (batch_size, num_agents, num_steps))
    self.assertEqual(
        simulated_features.time_to_collision.shape,
        (batch_size, num_agents, num_steps),
    )
    self.assertEqual(
        simulated_features.distance_to_road_edge.shape,
        (batch_size, num_agents, num_steps),
    )
    self.assertEqual(simulated_features.offroad_per_step.shape,
                     (batch_size, num_agents, num_steps))
    # Since this joint scene is coming directly from logs, we should expect a
    # zero ADE.
    ade = simulated_features.average_displacement_error
    self.assertAllClose(ade, tf.zeros_like(ade))

  def test_features_from_log_scenario_copies_log_validity(self):
    scenario = test_utils.get_womd_test_scenario()
    joint_scene = converters.scenario_to_joint_scene(scenario)
    simulated_features = (
        metric_features.compute_metric_features(
            scenario, joint_scene, use_log_validity=True))

    self.assertFalse(tf.reduce_all(simulated_features.valid))

    # Check that collision and offroad are correctly filtered by validity.
    distance_to_nearest_with_valids = tf.where(
        simulated_features.valid,
        simulated_features.distance_to_nearest_object,
        interaction_features.EXTREMELY_LARGE_DISTANCE)
    distance_to_road_edge_with_valids = tf.where(
        simulated_features.valid,
        simulated_features.distance_to_road_edge,
        -map_metric_features.EXTREMELY_LARGE_DISTANCE)

    collisions = tf.less(
        tf.reduce_min(distance_to_nearest_with_valids, axis=2),
        interaction_features.COLLISION_DISTANCE_THRESHOLD,
    )
    offroad = tf.greater(
        tf.reduce_max(distance_to_road_edge_with_valids, axis=2),
        map_metric_features.OFFROAD_DISTANCE_THRESHOLD,
    )
    self.assertAllEqual(collisions, tf.reduce_any(
        simulated_features.collision_per_step, axis=2))
    self.assertAllEqual(offroad, tf.reduce_any(
        simulated_features.offroad_per_step, axis=2))

  def test_features_from_linear_extrapolation_test_submission(self):
    scenario = test_utils.get_womd_test_scenario()
    submission = sim_agents_test_utils.load_test_submission()
    log_features, sim_features = (
        metric_features.compute_scenario_rollouts_features(
            scenario, submission.scenario_rollouts[0])
    )

    self.assertEqual(log_features.valid.shape, (1, 4, 80))
    self.assertEqual(sim_features.valid.shape, (32, 4, 80))
    # The ADE for the "log" features should be zero, as it is a direct copy
    # of the original data.
    log_ade = log_features.average_displacement_error
    self.assertEqual(log_ade.shape, (1, 4))
    self.assertAllClose(log_ade, tf.zeros_like(log_ade))
    # The contrary should be true for the submission, which was computed with
    # a linear extrapolation at constant speed.
    sim_ade = sim_features.average_displacement_error
    self.assertEqual(sim_ade.shape, (32, 4))
    self.assertNotAllClose(sim_ade, tf.zeros_like(sim_ade))


if __name__ == '__main__':
  tf.test.main()
