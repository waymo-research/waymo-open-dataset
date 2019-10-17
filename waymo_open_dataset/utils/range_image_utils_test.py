# Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for waymo_open_dataset.utils.range_image_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import test_utils


def _transform_range_image(range_image_cartesian, transform):
  """Right multiply transform to every point in the cartesian range image.

  Args:
    range_image_cartesian: [B, H, W, 3].
    transform: [B, 4, 4].

  Returns:
    range_image_transformed: [B, H, W, 3].
  """

  # [B, 3, 3]
  rotation = transform[..., 0:3, 0:3]
  # [B, 3]
  translation = transform[..., 0:3, 3]
  # [B, 1, 1, 3]
  translation = translation[:, tf.newaxis, tf.newaxis, :]

  range_image_transformed = translation + tf.einsum("bij,bhwj->bhwi", rotation,
                                                    range_image_cartesian)
  return range_image_transformed


class RangeImageUtilsTest(tf.test.TestCase):

  def test_compute_range_image_polar(self):
    batch = 3
    # The first entry in the channel is range.
    range_image = test_utils.generate_range_image(
        tf.constant([
            [1, 1],
            [2, 2],
        ], dtype=tf.int32),
        tf.constant([
            [5.0, 0.0, 1.5],
            [5.0, 0.0, 1.6],
        ], dtype=tf.float32), [3, 5, 3], batch)
    range_image = range_image[..., 0]
    extrinsic = test_utils.generate_extrinsic(3.14 / 4, 0.0, 0.0,
                                              [1.0, 0.0, 0.0], batch)
    inclination = tf.tile([[1.0, 0.0, -1.0]], [batch, 1])

    range_image_polar = range_image_utils.compute_range_image_polar(
        range_image, extrinsic, inclination)

    with self.test_session() as sess:
      ri_polar, _ = sess.run([range_image_polar, extrinsic])
      self.assertAllClose(ri_polar.shape, [3, 3, 5, 3])

  def test_compute_range_image_cartesian(self):
    batch = 3
    # The first entry in the channel is range.
    range_image = test_utils.generate_range_image(
        tf.constant([
            [1, 1],
            [2, 2],
        ], dtype=tf.int32),
        tf.constant([
            [5.0, 0.0, 1.5],
            [6.0, 0.0, 1.6],
        ], dtype=tf.float32), [3, 5, 3], batch)
    range_image = range_image[..., 0]
    extrinsic = test_utils.generate_extrinsic(3.14 / 4, 0.0, 0.0,
                                              [1.0, 0.0, 0.0], batch)
    inclination = tf.tile([[1.0, 0.0, -1.0]], [batch, 1])

    range_image_polar = range_image_utils.compute_range_image_polar(
        range_image, extrinsic, inclination)
    range_image_cartesian = range_image_utils.compute_range_image_cartesian(
        range_image_polar, extrinsic)
    # Transform to sensor frame.
    range_image_cartesian_sensor_frame = _transform_range_image(
        range_image_cartesian, tf.linalg.inv(extrinsic))

    with self.test_session() as sess:
      ri_polar, ri_cartesian = sess.run(
          [range_image_polar, range_image_cartesian_sensor_frame])
      self.assertAllClose(ri_polar.shape, [3, 3, 5, 3])
      self.assertAllClose(ri_cartesian.shape, [3, 3, 5, 3])

      for i in range(3):
        self.assertAllClose(np.linalg.norm(ri_cartesian[i, 1, 1, :]), 5.0)
        self.assertAllClose(np.linalg.norm(ri_cartesian[i, 2, 2, :]), 6.0)

  def test_compute_range_image_cartesian_from_raw_range_image(self):
    batch = 3
    # The first entry in the channel is range.
    range_image = test_utils.generate_range_image(
        tf.constant([
            [1, 1],
            [2, 2],
        ], dtype=tf.int32),
        tf.constant([
            [5.0, 0.0, 1.5],
            [6.0, 0.0, 1.6],
        ], dtype=tf.float32), [3, 5, 3], batch)
    range_image = range_image[..., 0]
    extrinsic = test_utils.generate_extrinsic(3.14 / 4, 0.0, 0.0,
                                              [1.0, 0.0, 0.0], batch)
    frame_pose = test_utils.generate_extrinsic(3.14 / 4, 0.0, 0.0,
                                               [1.0, 0.0, 0.0], batch)
    pixel_pose = tf.tile(
        tf.expand_dims(
            tf.expand_dims(
                test_utils.generate_extrinsic(3.14 / 4, 0.0, 0.0,
                                              [1.0, 0.0, 0.0], batch), 1), 1),
        [1, 3, 5, 1, 1])
    inclination = tf.tile([[1.0, 0.0, -1.0]], [batch, 1])

    range_image_polar = range_image_utils.compute_range_image_polar(
        range_image, extrinsic, inclination)
    range_image_cartesian = range_image_utils.compute_range_image_cartesian(
        range_image_polar,
        extrinsic,
        pixel_pose=pixel_pose,
        frame_pose=frame_pose)
    # Transform to sensor frame.
    range_image_cartesian_sensor_frame = _transform_range_image(
        range_image_cartesian, tf.linalg.inv(extrinsic))

    with self.test_session() as sess:
      ri_polar, ri_cartesian = sess.run(
          [range_image_polar, range_image_cartesian_sensor_frame])
      self.assertAllClose(ri_polar.shape, [3, 3, 5, 3])
      self.assertAllClose(ri_cartesian.shape, [3, 3, 5, 3])

  def test_build_range_image_from_point_cloud_from_range_image(self):
    """Builds range image from points extracted from range."""
    batch = 2
    width = 5
    # The first entry in the channel is range.
    index = tf.constant([
        [1, 1],
        [2, 2],
    ], dtype=tf.int32)
    range_image = test_utils.generate_range_image(
        index, tf.constant([
            [5.0, 0.0, 1.5],
            [6.0, 0.0, 1.6],
        ],
                           dtype=tf.float32), [5, width, 3], batch)
    range_image = range_image[..., 0]
    extrinsic = test_utils.generate_extrinsic(3.14 / 4, 0.0, 0.0,
                                              [1.0, 0.1, 0.1], batch)
    inclination = tf.tile([[1.0, 0.5, 0.0, -1.0, -2.0]], [batch, 1])

    range_image_polar = range_image_utils.compute_range_image_polar(
        range_image, extrinsic, inclination)
    range_image_cartesian = range_image_utils.compute_range_image_cartesian(
        range_image_polar, extrinsic)

    indices = tf.constant([[0, 1, 1], [0, 2, 2], [1, 1, 1], [1, 2, 2]],
                          dtype=tf.int32)
    # [4, 3]
    points = tf.gather_nd(range_image_cartesian, indices)
    points = tf.reshape(points, [batch, 2, 3])
    # [batch, 5, 3]
    num_points = tf.constant([2, 2], dtype=tf.int32)

    range_images, range_image_indices, range_image_ranges = (
        range_image_utils.build_range_image_from_point_cloud(
            points,
            num_points,
            extrinsic,
            inclination,
            point_features=points[..., 0:2] / 100.0,
            range_image_size=[5, width]))

    with self.test_session() as sess:
      ri, _, _ = sess.run(
          [range_images, range_image_indices, range_image_ranges])
      self.assertAllClose(ri[..., 0], range_image, atol=0.01)

  def test_build_range_image_from_point_cloud_from_points(self):
    """Builds range image from points directly."""

    # Need large enough range to deal with numerical errors.
    batch = 1
    num_cols = 4000
    inclination = tf.range(1.0, -1.0, -0.001)
    num_rows = inclination.shape[0]

    points = tf.tile(
        tf.expand_dims(
            tf.constant([
                [5.0, 0.0, 1.5],
                [20.0, 0.0, 1.5],
            ], dtype=tf.float32),
            axis=0), [batch, 1, 1])
    num_points = tf.ones([batch], dtype=tf.int32) * 2
    inclination = tf.tile(tf.expand_dims(inclination, axis=0), [batch, 1])
    extrinsic = test_utils.generate_extrinsic(3.14 / 8, 0.0, 0.0,
                                              [1.0, 0.1, 0.1], batch)

    range_image, _, _ = range_image_utils.build_range_image_from_point_cloud(
        points, num_points, extrinsic, inclination, [num_rows, num_cols])
    range_image_mask = tf.compat.v1.where(range_image > 1e-5)

    polar = range_image_utils.compute_range_image_polar(range_image, extrinsic,
                                                        inclination)
    cartesian = range_image_utils.compute_range_image_cartesian(
        polar, extrinsic)
    points_recovered = tf.gather_nd(cartesian, range_image_mask)
    points_recovered = tf.reshape(points_recovered, [batch, -1, 3])

    with self.test_session() as sess:
      points_recovered, points = sess.run([points_recovered, points])
      self.assertAllClose(points_recovered, points, rtol=1e-02, atol=1e-2)

  def test_build_camera_depth_image(self):
    batch = 2
    width = 5
    index = tf.constant([
        [1, 1],
        [2, 2],
        [3, 3],
    ], dtype=tf.int32)
    range_image_cartesian = test_utils.generate_range_image(
        index,
        tf.constant([
            [5.0, 0.0, 1.5],
            [6.0, 0.0, 1.6],
            [6.0, 0.0, 2.6],
        ],
                    dtype=tf.float32), [5, width, 3], batch)
    camera_projection = test_utils.generate_range_image(
        index,
        tf.constant([
            [1, 2, 3, 2, 4, 5],
            [1, 2, 3, 0, 1, 1],
            [0, 0, 0, 1, 4, 9],
        ],
                    dtype=tf.int32), [5, width, 6], batch)
    extrinsic = test_utils.generate_extrinsic(3.14 / 4, 0.0, 0.0,
                                              [1.0, 0.0, 0.0], batch)

    image = range_image_utils.build_camera_depth_image(range_image_cartesian,
                                                       extrinsic,
                                                       camera_projection,
                                                       [11, 6], 1)

    with self.test_session() as sess:
      image = sess.run(image)
      self.assertAlmostEqual(image[0, 3, 2],
                             math.sqrt((5.0 - 1.0) * (5.0 - 1.0) + 1.5 * 1.5),
                             5)
      self.assertAlmostEqual(image[0, 9, 4],
                             math.sqrt((6.0 - 1.0) * (6.0 - 1.0) + 2.6 * 2.6),
                             5)

  def test_crop_range_image(self):
    batch = 2
    num_cols = 4000
    inclination = tf.range(1.0, -1.0, -0.001)
    num_rows = inclination.shape[0]

    points = tf.tile(
        tf.expand_dims(
            tf.constant([
                [5.0, 0.0, 1.5],
                [20.0, 0.0, 1.5],
            ], dtype=tf.float32),
            axis=0), [batch, 1, 1])
    num_points = tf.ones([batch], dtype=tf.int32) * 2
    inclination = tf.tile(tf.expand_dims(inclination, axis=0), [batch, 1])
    extrinsic = test_utils.generate_extrinsic(3.14 / 8, 0.0, 0.0,
                                              [1.0, 0.1, 0.1], batch)
    range_image, _, _ = range_image_utils.build_range_image_from_point_cloud(
        points, num_points, extrinsic, inclination, [num_rows, num_cols])

    range_image_crops = []
    new_widths = [num_cols, num_cols - 1, num_cols - 2, num_cols - 3, 1]
    for new_width in new_widths:
      range_image_crops.append(
          range_image_utils.crop_range_image(range_image, new_width))
    with self.test_session() as sess:
      ri_crops = sess.run(range_image_crops)
      for i, ri_crop in enumerate(ri_crops):
        self.assertEqual(ri_crop.shape[2], new_widths[i])
        self.assertEqual(len(range_image_crops[i].shape), 3)

  def test_compute_inclination(self):
    inclinations = range_image_utils.compute_inclination(
        tf.constant([-1.0, 1.0]), 4)
    with self.test_session() as sess:
      incl = sess.run(inclinations)
      self.assertAllClose(incl, [
          -1.0 + 0.5 * 2 / 4, -1.0 + 1.5 * 2 / 4, -1.0 + 2.5 * 2 / 4,
          -1.0 + 3.5 * 2 / 4
      ])

  def test_encode_lidar_features(self):
    lidar_features = tf.constant(
        [[1.0, 4.0, 0.9], [2.0, 1.5, 0.8], [3.0, 5.5, 1.4]], dtype=tf.float32)
    lidar_features_encoded = range_image_utils.encode_lidar_features(
        lidar_features)
    lidar_features_decoded = range_image_utils.decode_lidar_features(
        lidar_features_encoded)

    index = tf.constant([[0, 0], [0, 0], [0, 0]], dtype=tf.int32)
    scatter_feature = range_image_utils.decode_lidar_features(
        range_image_utils.scatter_nd_with_pool(index, lidar_features_encoded,
                                               [2, 2],
                                               tf.math.unsorted_segment_min))

    with self.test_session():
      self.assertAllClose(
          lidar_features.eval(), lidar_features_decoded.eval(), atol=0.1)

      self.assertAllClose(
          scatter_feature.eval()[0, 0, :],
          lidar_features_decoded.eval()[0, :],
          atol=0.001)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
