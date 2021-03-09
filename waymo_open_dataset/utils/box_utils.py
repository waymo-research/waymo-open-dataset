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

__all__ = [
    'is_within_box_3d', 'compute_num_points_in_box_3d', 'is_within_box_2d',
    'get_upright_3d_box_corners', 'transform_point', 'transform_box'
]


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

  with tf.compat.v1.name_scope(name, 'IsWithinBox3D', [point, box]):
    center = box[:, 0:3]
    dim = box[:, 3:6]
    heading = box[:, 6]
    # [M, 3, 3]
    rotation = transform_utils.get_yaw_rotation(heading)
    # [M, 4, 4]
    transform = transform_utils.get_transform(rotation, center)
    # [M, 4, 4]
    transform = tf.linalg.inv(transform)
    # [M, 3, 3]
    rotation = transform[:, 0:3, 0:3]
    # [M, 3]
    translation = transform[:, 0:3, 3]

    # [N, M, 3]
    point_in_box_frame = tf.einsum('nj,mij->nmi', point, rotation) + translation
    # [N, M, 3]
    point_in_box = tf.logical_and(
        tf.logical_and(point_in_box_frame <= dim * 0.5,
                       point_in_box_frame >= -dim * 0.5),
        tf.reduce_all(tf.not_equal(dim, 0), axis=-1, keepdims=True))
    # [N, M]
    point_in_box = tf.cast(
        tf.reduce_prod(
            input_tensor=tf.cast(point_in_box, dtype=tf.uint8), axis=-1),
        dtype=tf.bool)

    return point_in_box


def is_within_box_2d(point, box):
  """Checks whether a point is in a BEV box given a set of points and boxes.

  Args:
    point: [N, 2] tensor. Inner dims are: [x, y].
    box: [M, 5] tensor. Inner dims are: [center_x, center_y, length, width,
      heading].

  Returns:
    point_in_box; [N, M] boolean tensor.
  """

  with tf.name_scope('IsWithinBox2D'):
    center = box[:, 0:2]
    dim = box[:, 2:4]
    heading = box[:, 4]
    # [M, 2, 2]
    rotation = transform_utils.get_yaw_rotation_2d(heading)
    # [M, 3, 3]
    transform = transform_utils.get_transform(rotation, center)
    # [M, 3, 3]
    transform = tf.linalg.inv(transform)
    # [M, 2, 2]
    rotation = transform[:, 0:2, 0:2]
    # [M, 2]
    translation = transform[:, 0:2, 2]

    # [N, M, 2]
    point_in_box_frame = tf.einsum('nj,mij->nmi', point, rotation) + translation
    # [N, M, 2]
    point_in_box = tf.logical_and(
        tf.logical_and(point_in_box_frame <= dim * 0.5,
                       point_in_box_frame >= -dim * 0.5),
        tf.reduce_all(tf.not_equal(dim, 0), axis=-1, keepdims=True))
    # [N, M]
    point_in_box = tf.cast(
        tf.reduce_prod(
            input_tensor=tf.cast(point_in_box, dtype=tf.int32), axis=-1),
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

  with tf.compat.v1.name_scope(name, 'ComputeNumPointsInBox3D', [point, box]):
    # [N, M]
    point_in_box = tf.cast(is_within_box_3d(point, box, name), dtype=tf.int32)
    num_points_in_box = tf.reduce_sum(input_tensor=point_in_box, axis=0)
    return num_points_in_box


def get_upright_3d_box_corners(boxes, name=None):
  """Given a set of upright boxes, return its 8 corners.

  Given a set of boxes, returns its 8 corners. The corners are ordered layers
  (bottom, top) first and then counter-clockwise within each layer.

  Args:
    boxes: tf Tensor [N, 7]. The inner dims are [center{x,y,z}, length, width,
      height, heading].
    name: the name scope.

  Returns:
    corners: tf Tensor [N, 8, 3].
  """
  with tf.compat.v1.name_scope(name, 'GetUpright3dBoxCorners', [boxes]):
    center_x, center_y, center_z, length, width, height, heading = tf.unstack(
        boxes, axis=-1)

    # [N, 3, 3]
    rotation = transform_utils.get_yaw_rotation(heading)
    # [N, 3]
    translation = tf.stack([center_x, center_y, center_z], axis=-1)

    l2 = length * 0.5
    w2 = width * 0.5
    h2 = height * 0.5

    # [N, 8, 3]
    corners = tf.reshape(
        tf.stack([
            l2, w2, -h2, -l2, w2, -h2, -l2, -w2, -h2, l2, -w2, -h2, l2, w2, h2,
            -l2, w2, h2, -l2, -w2, h2, l2, -w2, h2
        ],
                 axis=-1), [-1, 8, 3])
    # [N, 8, 3]
    corners = tf.einsum('nij,nkj->nki', rotation, corners) + tf.expand_dims(
        translation, axis=-2)

    return corners


def transform_point(point, from_frame_pose, to_frame_pose, name=None):
  """Transforms 3d points from one frame to another.

  Args:
    point: [..., N, 3] points.
    from_frame_pose: [..., 4, 4] origin frame poses.
    to_frame_pose: [..., 4, 4] target frame poses.
    name: tf name scope.

  Returns:
    Transformed points of shape [..., N, 3] with the same type as point.
  """
  with tf.compat.v1.name_scope(name, 'TransformPoint'):
    transform = tf.linalg.matmul(tf.linalg.inv(to_frame_pose), from_frame_pose)
    return tf.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                     point) + tf.expand_dims(
                         transform[..., 0:3, 3], axis=-2)


def transform_box(box, from_frame_pose, to_frame_pose, name=None):
  """Transforms 3d upright boxes from one frame to another.

  Args:
    box: [..., N, 7] boxes.
    from_frame_pose: [...,4, 4] origin frame poses.
    to_frame_pose: [...,4, 4] target frame poses.
    name: tf name scope.

  Returns:
    Transformed boxes of shape [..., N, 7] with the same type as box.
  """
  with tf.compat.v1.name_scope(name, 'TransformBox'):
    transform = tf.linalg.matmul(tf.linalg.inv(to_frame_pose), from_frame_pose)
    heading_offset = tf.atan2(transform[..., 1, 0], transform[..., 0, 0])
    heading = box[..., -1] + heading_offset[..., tf.newaxis]
    center = tf.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                       box[..., 0:3]) + tf.expand_dims(
                           transform[..., 0:3, 3], axis=-2)
    return tf.concat([center, box[..., 3:6], heading[..., tf.newaxis]], axis=-1)
