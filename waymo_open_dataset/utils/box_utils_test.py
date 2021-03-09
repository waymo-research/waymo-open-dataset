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
"""Tests for waymo_open_dataset.utils.box_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils import transform_utils


class BoxUtilsTest(tf.test.TestCase):

  def test_is_within_box_3d(self):
    # Unit size boxes centered at (center, 0, 2)
    box, _, num_box = test_utils.generate_boxes([5.0, 19.6], 1)
    box = box[0, 0:num_box[0], :]
    point = tf.constant(
        [
            [5.0, 0.0, 1.5],  # in
            [20.0, 0.0, 1.6],  # in
            [20.0, 0.0, 2.6],  # not in
        ],
        dtype=tf.float32)

    point_in_box = box_utils.is_within_box_3d(point, box)

    with self.test_session() as sess:
      point_in_box = sess.run(point_in_box)
      self.assertAllEqual(point_in_box,
                          [[True, False], [False, True], [False, False]])

  def test_compute_num_points_in_box_3d(self):
    # Unit size boxes centered at (center, 0, 2)
    box, _, num_box = test_utils.generate_boxes([5.0, 5.5, 19.6, 50], 1)
    box = box[0, 0:num_box[0], :]
    point = tf.constant(
        [
            [5.0, 0.0, 1.5],  # in
            [5.0, 0.0, 2.5],  # in
            [20.0, 0.0, 1.6],  # in
            [20.0, 0.0, 2.6],  # not in
        ],
        dtype=tf.float32)

    num_points_in_box = box_utils.compute_num_points_in_box_3d(point, box)

    with self.test_session() as sess:
      num_points_in_box = sess.run(num_points_in_box)
      self.assertAllEqual(num_points_in_box, [2, 2, 1, 0])

  def test_transform_point_among_frames(self):
    p = tf.constant([[1.0, 0, 0]], dtype=tf.float32)
    from_frame_pose = transform_utils.get_transform(
        transform_utils.get_yaw_rotation(math.pi * 0.5),
        tf.constant([0.0, 0.0, 2.0], dtype=tf.float32))
    to_frame_pose = transform_utils.get_transform(
        transform_utils.get_yaw_rotation(math.pi * 0.1),
        tf.constant([0.0, 0.0, 0.0], dtype=tf.float32))

    p = p[tf.newaxis, tf.newaxis, tf.newaxis, ...]
    from_frame_pose = from_frame_pose[tf.newaxis, tf.newaxis, tf.newaxis, ...]
    to_frame_pose = to_frame_pose[tf.newaxis, tf.newaxis, tf.newaxis, ...]
    pp = box_utils.transform_point(p, from_frame_pose, to_frame_pose)

    with self.test_session():
      self.assertAllClose(
          pp[0, 0, 0, ...].eval(),
          [[math.cos(math.pi * 0.4),
            math.sin(math.pi * 0.4), 2.0]])

  def test_transform_box_among_frames(self):
    b = tf.constant([[1.0, 0, 0, 2.0, 2.0, 2.0, math.pi * 0.1]],
                    dtype=tf.float32)
    from_frame_pose = transform_utils.get_transform(
        transform_utils.get_yaw_rotation(math.pi * 0.5),
        tf.constant([0.0, 0.0, 1.0], dtype=tf.float32))
    to_frame_pose = transform_utils.get_transform(
        transform_utils.get_yaw_rotation(math.pi * 0.25),
        tf.constant([0.0, 0.0, 0.0], dtype=tf.float32))

    b = b[tf.newaxis, tf.newaxis, tf.newaxis, ...]
    from_frame_pose = from_frame_pose[tf.newaxis, tf.newaxis, tf.newaxis, ...]
    to_frame_pose = to_frame_pose[tf.newaxis, tf.newaxis, tf.newaxis, ...]

    bb = box_utils.transform_box(b, from_frame_pose, to_frame_pose)

    with self.test_session():
      self.assertAllClose(bb[0, 0, 0, ...].eval(), [[
          math.cos(math.pi * 0.25),
          math.sin(math.pi * 0.25), 1.0, 2.0, 2.0, 2.0, math.pi * 0.35
      ]])

  def test_no_points_in_zero_boxes(self):
    # Unit size boxes centered at (center, 0, 2)
    box, _, num_box = test_utils.generate_boxes([0.0], 1)
    box = box[0, 0:num_box[0], :]
    point = tf.constant(
        [
            [5.0, 0.0, 1.5],  # not in
            [20.0, 0.0, 1.6],  # not in
            [20.0, 0.0, 2.6],  # not in
        ],
        dtype=tf.float32)

    point_in_box = box_utils.is_within_box_3d(point, box)

    with self.test_session() as sess:
      point_in_box = sess.run(point_in_box)
      self.assertAllEqual(point_in_box, [[False], [False], [False]])


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
