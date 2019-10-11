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
"""Tests for waymo_open_dataset.utils.transform_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from waymo_open_dataset.utils import transform_utils


class TransformUtilsTest(tf.test.TestCase):

  def test_get_transform(self):
    # [3, 3]
    point = tf.constant([
        [5.0, 0.0, 1.5],
        [20.0, 0.0, 1.6],
        [20.0, 0.0, 2.6],
    ],
                        dtype=tf.float32)

    transform = transform_utils.get_transform(
        transform_utils.get_rotation_matrix(1.0, 2.0, 1.5),
        tf.constant([2.0, 3.0, 4.0]))

    point_transformed = tf.einsum('ij,kj->ki', transform[0:3, 0:3],
                                  point) + transform[0:3, 3]

    transform = tf.linalg.inv(transform)
    point_transformed_back = tf.einsum('ij,kj->ki', transform[0:3, 0:3],
                                       point_transformed) + transform[0:3, 3]

    with self.test_session() as sess:
      p1, p2 = sess.run([point, point_transformed_back])
      self.assertAllClose(p1, p2)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
