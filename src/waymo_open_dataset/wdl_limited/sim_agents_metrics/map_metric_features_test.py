# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================

import math

import tensorflow as tf

from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils.sim_agents import converters
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.wdl_limited.sim_agents_metrics import map_metric_features


PI = math.pi


class MapMetricFeaturesTest(tf.test.TestCase):

  def test_distance_to_road_edge_has_correct_shape(self):
    scenario = test_utils.get_womd_test_scenario()
    submission = sim_agents_test_utils.load_test_submission()
    simulated_trajectories = converters.joint_scene_to_trajectories(
        submission.scenario_rollouts[0].joint_scenes[0], scenario)
    mask = tf.convert_to_tensor([True] * 4 + [False] * 42 + [True] * 4)

    road_edges = []
    for map_feature in scenario.map_features:
      if map_feature.HasField('road_edge'):
        road_edges.append(map_feature.road_edge.polyline)

    distances = map_metric_features.compute_distance_to_road_edge(
        center_x=simulated_trajectories.x,
        center_y=simulated_trajectories.y,
        center_z=simulated_trajectories.z,
        length=simulated_trajectories.length,
        width=simulated_trajectories.width,
        height=simulated_trajectories.height,
        heading=simulated_trajectories.heading,
        valid=simulated_trajectories.valid,
        evaluated_object_mask=mask,
        road_edge_polylines=road_edges,
    )
    self.assertEqual(
        distances.shape, (8, simulated_trajectories.valid.shape[-1])
    )

  def test_distance_to_road_edge_fails_on_missing_map(self):
    scenario = test_utils.get_womd_test_scenario()
    submission = sim_agents_test_utils.load_test_submission()
    simulated_trajectories = converters.joint_scene_to_trajectories(
        submission.scenario_rollouts[0].joint_scenes[0], scenario)
    mask = tf.convert_to_tensor([True] * 4 + [False] * 42 + [True] * 4)

    with self.assertRaisesRegex(ValueError, 'Missing road edges.'):
      map_metric_features.compute_distance_to_road_edge(
          center_x=simulated_trajectories.x,
          center_y=simulated_trajectories.y,
          center_z=simulated_trajectories.z,
          length=simulated_trajectories.length,
          width=simulated_trajectories.width,
          height=simulated_trajectories.height,
          heading=simulated_trajectories.heading,
          valid=simulated_trajectories.valid,
          evaluated_object_mask=mask,
          road_edge_polylines=[],
      )

  def test_signed_distance_to_polyline_has_correct_sign(self):
    #       R2
    #       ^
    #  P    |     Q
    #       R1
    query_points = tf.constant([[-1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    polyline = tf.constant([[0.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]])
    signed_distances = (
        map_metric_features._compute_signed_distance_to_polylines(
            xyzs=query_points, polylines=polyline[tf.newaxis]
        )
    )
    self.assertAllClose(signed_distances, tf.constant([-1.0, 2.0]))

  def test_signed_distance_to_polylines_has_correct_magnitude(self):
    #      P
    #
    #  R1----->R2
    #
    #               Q
    query_points = tf.constant([[0.0, 1.0, 0.0], [3.0, -1.0, 0.0]])
    polyline = tf.constant([[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]])
    signed_distances = (
        map_metric_features._compute_signed_distance_to_polylines(
            xyzs=query_points, polylines=polyline[tf.newaxis]
        )
    )
    self.assertAllClose(
        tf.math.abs(signed_distances), tf.constant([1.0, math.sqrt(2)])
    )

  def test_signed_distance_to_polylines_circle(self):
    radius = 1.2
    # Square mesh grid from -2*r to 2r with 100 points.
    x = tf.linspace(-2.0 * radius, 2.0 * radius, 10)
    mesh_xys = tf.reshape(tf.stack(tf.meshgrid(x, x), axis=-1), [-1, 2])
    mesh_xyzs = tf.concat([mesh_xys, tf.zeros_like(mesh_xys[:, :1])], axis=-1)
    # Sampled circle with 40 segments.
    t = tf.linspace(0.0, 1.0, 41)
    polyline = tf.stack(
        [
            tf.math.cos(t * 2.0 * PI) * radius,
            tf.math.sin(t * 2.0 * PI) * radius,
            tf.zeros_like(t, dtype=tf.float32),
            tf.ones_like(t, dtype=tf.float32),  # validity flags
        ],
        axis=-1,
    )

    signed_distances = (
        map_metric_features._compute_signed_distance_to_polylines(
            xyzs=mesh_xyzs, polylines=polyline[tf.newaxis]
        )
    )
    expected_signed_distance = tf.linalg.norm(mesh_xys, axis=-1) - radius
    self.assertAllClose(
        signed_distances, expected_signed_distance, rtol=0.02, atol=0.01
    )

  def test_signed_distance_to_polylines_square(self):
    side_length = 0.8
    # Square mesh grid from -2 to 2 with 100 points.
    x = tf.linspace(-2.0, 2.0, 10)
    mesh_xys = tf.reshape(tf.stack(tf.meshgrid(x, x), axis=-1), [-1, 2])
    mesh_xyzs = tf.concat([mesh_xys, tf.zeros_like(mesh_xys[:, :1])], axis=-1)
    # Square with 4 segments.
    polyline = tf.constant(
        [
            [1, 1, 0, 1],
            [-1, 1, 0, 1],
            [-1, -1, 0, 1],
            [1, -1, 0, 1],
            [1, 1, 0, 1],
        ],
        dtype=tf.float32,
    ) * tf.constant([side_length, side_length, 1.0, 1.0])

    signed_distances = (
        map_metric_features._compute_signed_distance_to_polylines(
            xyzs=mesh_xyzs, polylines=polyline[tf.newaxis]
        )
    )
    # Close-form signed distance to box.
    d = tf.math.abs(mesh_xys) - side_length
    expected_signed_distance = tf.linalg.norm(
        tf.maximum(d, 0.0), axis=-1
    ) + tf.minimum(tf.math.reduce_max(d, axis=-1), 0.0)
    self.assertAllClose(signed_distances, expected_signed_distance)

  def test_signed_distance_to_polylines_with_2_lines(self):
    # Square mesh grid from -1 to 4 with 100 points.
    x = tf.linspace(-1.0, 4.0, 10)
    mesh_xys = tf.reshape(tf.stack(tf.meshgrid(x, x), axis=-1), [-1, 2])
    mesh_xyzs = tf.concat([mesh_xys, tf.zeros_like(mesh_xys[:, :1])], axis=-1)
    # Two straight parallel lines.
    polylines = tf.stack(
        [
            tf.constant([[0.0, 10.0, 0.0, 1.0], [0.0, -10.0, 0.0, 1.0]]),
            tf.constant([[2.0, -10.0, 0.0, 1.0], [2.0, 10.0, 0.0, 1.0]]),
        ],
        axis=0,
    )
    signed_distances = (
        map_metric_features._compute_signed_distance_to_polylines(
            xyzs=mesh_xyzs, polylines=polylines
        )
    )
    expected_signed_distance = tf.math.abs(mesh_xys[:, 0] - 1.0) - 1.0
    self.assertAllClose(signed_distances, expected_signed_distance)

  def test_signed_distance_to_polylines_with_padded_lines(self):
    # Square mesh grid from -1 to 4 with 100 points.
    x = tf.linspace(-1.0, 4.0, 10)
    mesh_xys = tf.reshape(tf.stack(tf.meshgrid(x, x), axis=-1), [-1, 2])
    mesh_xyzs = tf.concat([mesh_xys, tf.zeros_like(mesh_xys[:, :1])], axis=-1)
    # Two straight parallel lines, first one with 4 points, second one with 2
    # points + padding.
    polylines = tf.stack(
        [
            tf.constant([
                [0.0, 10.0, 0.0, 1.0],
                [0.0, 3.0, 0.0, 1.0],
                [0.0, -3.0, 0.0, 1.0],
                [0.0, -10.0, 0.0, 1.0],
            ]),
            tf.constant([
                [2.0, -10.0, 0.0, 1.0],
                [2.0, 10.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]),
        ],
        axis=0,
    )
    signed_distances = (
        map_metric_features._compute_signed_distance_to_polylines(
            xyzs=mesh_xyzs, polylines=polylines
        )
    )
    expected_signed_distance = tf.math.abs(mesh_xys[:, 0] - 1.0) - 1.0
    self.assertAllClose(signed_distances, expected_signed_distance)

  def test_signed_distance_to_polylines_with_2_lines_with_different_z(self):
    # Square mesh grid from -1 to 4 with 100 points.
    x = tf.linspace(-1.0, 4.0, 10)
    mesh_xys = tf.reshape(tf.stack(tf.meshgrid(x, x), axis=-1), [-1, 2])
    # Query points at z=8 and z=-8
    top_xyzs = tf.concat(
        [mesh_xys, 8.0 * tf.ones_like(mesh_xys[:, :1])], axis=-1
    )
    bottom_xyzs = tf.concat(
        [mesh_xys, -8.0 * tf.ones_like(mesh_xys[:, :1])], axis=-1
    )
    all_xyzs = tf.concat([bottom_xyzs, top_xyzs], axis=0)
    # 2 lines on the y and x axes at z=-10 and z=10.
    polylines = tf.stack(
        [
            tf.constant([[0.0, -10.0, -10.0, 1.0], [0.0, 10.0, -10.0, 1.0]]),
            tf.constant([[-10.0, 0.0, 10.0, 1.0], [10.0, 0.0, 10.0, 1.0]]),
        ],
        axis=0,
    )
    signed_distances = (
        map_metric_features._compute_signed_distance_to_polylines(
            xyzs=all_xyzs, polylines=polylines
        )
    )
    expected_signed_distance = tf.concat(
        [mesh_xys[:, 0], -mesh_xys[:, 1]], axis=0
    )

    self.assertAllClose(signed_distances, expected_signed_distance)

  def test_signed_distance_to_polylines_with_2_lines_with_close_z(self):
    # Square mesh grid from -1 to 4 with 100 points.
    x = tf.linspace(-1.0, 4.0, 100)
    mesh_xys = tf.reshape(tf.stack(tf.meshgrid(x, x), axis=-1), [-1, 2])
    # Query points at z=1 and z=-1
    top_xyzs = tf.concat(
        [mesh_xys, 1.0 * tf.ones_like(mesh_xys[:, :1])], axis=-1
    )
    bottom_xyzs = tf.concat(
        [mesh_xys, -1.0 * tf.ones_like(mesh_xys[:, :1])], axis=-1
    )
    all_xyzs = tf.concat([bottom_xyzs, top_xyzs], axis=0)
    # 2 lines on the y and x axes at z=-3 and z=3. The z-gap is small so
    # z-stretch is needed.
    polylines = tf.stack(
        [
            tf.constant([[0.0, -10.0, -3.0, 1.0], [0.0, 10.0, -3.0, 1.0]]),
            tf.constant([[-10.0, 0.0, 3.0, 1.0], [10.0, 0.0, 3.0, 1.0]]),
        ],
        axis=0,
    )
    signed_distances = (
        map_metric_features._compute_signed_distance_to_polylines(
            xyzs=all_xyzs, polylines=polylines, z_stretch=3.0
        )
    )
    expected_signed_distance = tf.concat(
        [mesh_xys[:, 0], -mesh_xys[:, 1]], axis=0
    )

    self.assertAllClose(signed_distances, expected_signed_distance)

if __name__ == '__main__':
  tf.test.main()
