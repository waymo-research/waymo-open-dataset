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
"""Utils for upright 3d box."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from waymo_open_dataset.utils import transform_utils

__all__ = ['is_within_box_3d', 'compute_num_points_in_box_3d']


def is_within_box_3d(point, box, name=None):
  """Checks whether a point is in a 3d box given a set of points and boxes.

  Args:
    point: [N, 3] tensor. Inner dims are: [x, y, z].
    box: [M, 7] tensor. Inner dims are: [center_x, center_y, center_z, length,
      width, height, heading].
    name: tf name scope.

  Returns:
    point_in_box; [N, M] boolean tensor.
  """

  with tf.name_scope(name, 'IsWithinBox3D', [point, box]):
    center = box[:, 0:3]
    dim = box[:, 3:6]
    heading = box[:, 6]
    # [M, 3, 3]
    rotation = transform_utils.get_yaw_rotation(heading)
    # [M, 4, 4]
    transform = transform_utils.get_transform(rotation, center)
    # [M, 4, 4]
    transform = tf.matrix_inverse(transform)
    # [M, 3, 3]
    rotation = transform[:, 0:3, 0:3]
    # [M, 3]
    translation = transform[:, 0:3, 3]

    # [N, M, 3]
    point_in_box_frame = tf.einsum('nj,mij->nmi', point, rotation) + translation
    # [N, M, 3]
    point_in_box = tf.logical_and(point_in_box_frame <= dim * 0.5,
                                  point_in_box_frame >= -dim * 0.5)
    # [N, M]
    point_in_box = tf.cast(
        tf.reduce_prod(tf.cast(point_in_box, dtype=tf.uint8), axis=-1),
        dtype=tf.bool)

    return point_in_box


def compute_num_points_in_box_3d(point, box, name=None):
  """Computes the number of points in each box given a set of points and boxes.

  Args:
    point: [N, 3] tensor. Inner dims are: [x, y, z].
    box: [M, 7] tenosr. Inner dims are: [center_x, center_y, center_z, length,
      width, height, heading].
    name: tf name scope.

  Returns:
    num_points_in_box: [M] int32 tensor.
  """

  with tf.name_scope(name, 'ComputeNumPointsInBox3D', [point, box]):
    # [N, M]
    point_in_box = tf.cast(is_within_box_3d(point, box, name), dtype=tf.int32)
    num_points_in_box = tf.reduce_sum(point_in_box, axis=0)
    return num_points_in_box
