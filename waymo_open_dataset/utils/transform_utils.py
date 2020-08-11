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
"""Utils to manage geometry transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

__all__ = [
    'get_yaw_rotation', 'get_yaw_rotation_2d', 'get_rotation_matrix',
    'get_transform'
]


def get_yaw_rotation(yaw, name=None):
  """Gets a rotation matrix given yaw only.

  Args:
    yaw: x-rotation in radians. This tensor can be any shape except an empty
      one.
    name: the op name.

  Returns:
    A rotation tensor with the same data type of the input. Its shape is
      [input_shape, 3 ,3].
  """
  with tf.compat.v1.name_scope(name, 'GetYawRotation', [yaw]):
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)
    ones = tf.ones_like(yaw)
    zeros = tf.zeros_like(yaw)

    return tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        tf.stack([zeros, zeros, ones], axis=-1),
    ],
                    axis=-2)


def get_yaw_rotation_2d(yaw):
  """Gets a rotation matrix given yaw only for 2d.

  Args:
    yaw: x-rotation in radians. This tensor can be any shape except an empty
      one.

  Returns:
    A rotation tensor with the same data type of the input. Its shape is
      [input_shape, 2, 2].
  """
  with tf.name_scope('GetYawRotation2D'):
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)

    return tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw], axis=-1),
        tf.stack([sin_yaw, cos_yaw], axis=-1),
    ],
                    axis=-2)


def get_rotation_matrix(roll, pitch, yaw, name=None):
  """Gets a rotation matrix given roll, pitch, yaw.

  roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
  x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

  https://en.wikipedia.org/wiki/Euler_angles
  http://planning.cs.uiuc.edu/node102.html

  Args:
    roll : x-rotation in radians.
    pitch: y-rotation in radians. The shape must be the same as roll.
    yaw: z-rotation in radians. The shape must be the same as roll.
    name: the op name.

  Returns:
    A rotation tensor with the same data type of the input. Its shape is
      [input_shape_of_yaw, 3 ,3].
  """
  with tf.compat.v1.name_scope(name, 'GetRotationMatrix', [yaw, pitch, roll]):
    cos_roll = tf.cos(roll)
    sin_roll = tf.sin(roll)
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)
    cos_pitch = tf.cos(pitch)
    sin_pitch = tf.sin(pitch)

    ones = tf.ones_like(yaw)
    zeros = tf.zeros_like(yaw)

    r_roll = tf.stack([
        tf.stack([ones, zeros, zeros], axis=-1),
        tf.stack([zeros, cos_roll, -1.0 * sin_roll], axis=-1),
        tf.stack([zeros, sin_roll, cos_roll], axis=-1),
    ],
                      axis=-2)
    r_pitch = tf.stack([
        tf.stack([cos_pitch, zeros, sin_pitch], axis=-1),
        tf.stack([zeros, ones, zeros], axis=-1),
        tf.stack([-1.0 * sin_pitch, zeros, cos_pitch], axis=-1),
    ],
                       axis=-2)
    r_yaw = tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        tf.stack([zeros, zeros, ones], axis=-1),
    ],
                     axis=-2)

    return tf.matmul(r_yaw, tf.matmul(r_pitch, r_roll))


def get_transform(rotation, translation):
  """Combines NxN rotation and Nx1 translation to (N+1)x(N+1) transform.

  Args:
    rotation: [..., N, N] rotation tensor.
    translation: [..., N] translation tensor. This must have the same type as
      rotation.

  Returns:
    transform: [..., (N+1), (N+1)] transform tensor. This has the same type as
      rotation.
  """
  with tf.name_scope('GetTransform'):
    # [..., N, 1]
    translation_n_1 = translation[..., tf.newaxis]
    # [..., N, N+1]
    transform = tf.concat([rotation, translation_n_1], axis=-1)
    # [..., N]
    last_row = tf.zeros_like(translation)
    # [..., N+1]
    last_row = tf.concat([last_row, tf.ones_like(last_row[..., 0:1])], axis=-1)
    # [..., N+1, N+1]
    transform = tf.concat([transform, last_row[..., tf.newaxis, :]], axis=-2)
    return transform
