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
"""Test utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from waymo_open_dataset.utils import transform_utils

__all__ = ['generate_boxes', 'generate_range_image', 'generate_extrinsic']


def generate_boxes(center_x, batch=1):
  """Generates unit size boxes centered at (center_x, 0.0, 2.0).

  Args:
    center_x: center_x for each box
    batch: batch size

  Returns:
    boxes: [batch, len(center_x) + 5, 7]. Each box has unit size, zero heading.
      They are centered at (center_x, 0.0, 2.0).
    box_classes: [batch, len(center_x) + 5]. The types are random.
    num_boxes: [batch]. Number of boxes for each batch dim.
  """
  center_y = np.zeros_like(center_x)
  center_z = np.ones_like(center_x) * 2.0
  heading = np.zeros_like(center_x)
  dim = np.ones_like(center_x)

  # [len(center_x), 7]
  boxes = np.stack([center_x, center_y, center_z, dim, dim, dim, heading],
                   axis=-1)

  box_padding = np.zeros([5, 7])

  boxes = np.concatenate([boxes, box_padding], axis=0)
  boxes = np.tile(np.expand_dims(boxes, axis=0), [batch, 1, 1])

  box_classes = np.tile(
      np.expand_dims(np.random.randint(1, 5, size=[len(center_x) + 5]), axis=0),
      [batch, 1])
  num_boxes = np.ones([batch]) * len(center_x)

  return (tf.convert_to_tensor(value=boxes, dtype=tf.float32),
          tf.convert_to_tensor(value=box_classes, dtype=tf.uint8),
          tf.convert_to_tensor(value=num_boxes, dtype=tf.int32))


def generate_range_image(indices, values, shape, batch=1):
  """Generate range images by scattering values to indices.

  Args:
    indices: [N, 2] indices.
    values: [N, ...] values.
    shape: [3].
    batch: batch indices, single integer.

  Returns:
    range_image: [batch, shape[...]]

  """
  multiples = tf.concat(
      [tf.constant([batch]),
       tf.ones([len(shape)], dtype=tf.int32)], axis=-1)

  return tf.tile(
      tf.expand_dims(tf.scatter_nd(indices, values, shape), axis=0), multiples)


def generate_extrinsic(yaw, pitch, roll, translation, batch=1):
  """Generates extrinsic.

  Args:
    yaw: scalar tensor
    pitch: scalar tensor
    roll: scalar tensor
    translation: [3] tensor
    batch: integer

  Returns:
    [batch, 4, 4] tensor
  """
  rotation_matrix = transform_utils.get_rotation_matrix(roll, pitch, yaw)
  return tf.tile(
      tf.expand_dims(
          transform_utils.get_transform(
              rotation_matrix, tf.constant(translation, dtype=tf.float32)),
          axis=0), [batch, 1, 1])
