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

import tensorflow as tf

from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import test_utils


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


if __name__ == "__main__":
  tf.test.main()
