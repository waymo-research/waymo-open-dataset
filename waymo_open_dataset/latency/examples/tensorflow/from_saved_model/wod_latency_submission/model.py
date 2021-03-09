# Copyright 2021 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""Module to load and run a SavedModel."""
from typing import Dict
import importlib_resources
import numpy as np
import tensorflow as tf


class Model(object):
  """A simple wrapper around a SavedModel."""

  def __init__(self):
    """Create an empty object with no underlying model yet."""
    self._model = None
    self._model_fn = None

  def initialize(self, model_dir: str):
    """Load the model from the SavedModel at model_dir.

    Args:
      model_dir: string directory of the SavedModel.
    """
    self._model = tf.saved_model.load(model_dir)
    self._model_fn = self._model.signatures['3d_detector_benchmark']

  def predict(self, range_image: tf.Tensor,
              range_image_cartesian: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Run inference on these range images and return a dict to TF Tensor.

    Args:
      range_image: HxWx3 float32 tensor with channels range, intensity,
        elongation.
      range_image_cartesian: HxWx3 float32 tensor with channels x, y, z.

    Returns:
      Dict from string to TF Tensor with the following key-value pairs:
        box_3d: 1xNx7 float32 tensor with the boxes for each detection (x, y, z,
          length, width, height, heading)
        box_score: 1xN float32 tensor with the confidence scores for each
            detection.
    """
    if not self._model_fn:
      raise ValueError('Model not initialized; call model.initialize()')

    ric = tf.cast(tf.convert_to_tensor(range_image_cartesian), tf.float32)
    ri = tf.cast(tf.convert_to_tensor(range_image), tf.float32)

    r = ri[..., 0]
    i = ri[..., 1]
    e = ri[..., 2]

    r = r * (1.0 / 75.0)
    i = tf.tanh(i)
    e = e * (1.0 / 1.5)

    range_image_normalized = tf.stack([r, i, e], -1)
    return self._model_fn(
        range_image_normalized=range_image_normalized,
        range_image_cartesian=ric)


DATA_FIELDS = ['TOP_RANGE_IMAGE_FIRST_RETURN']
model = Model()


def initialize_model():
  with importlib_resources.path(__package__, 'saved_model') as saved_model_path:
    model.initialize(str(saved_model_path))
  # Run the model once on dummy input to warm it up.
  run_model(np.zeros((64, 2650, 6)))


def run_model(TOP_RANGE_IMAGE_FIRST_RETURN):
  """Run the model on the 6-dimensional range image.

  Args:
    TOP_RANGE_IMAGE_FIRST_RETURN: H x W x 6 numpy ndarray

  Returns:
    Dict from string to numpy ndarray.
  """
  # Convert the numpy array to the desired formats (including converting to TF
  # Tensor.)
  range_image = tf.cast(
      tf.convert_to_tensor(TOP_RANGE_IMAGE_FIRST_RETURN[:, :, :3]), tf.float32)
  range_image_cartesian = tf.cast(
      tf.convert_to_tensor(TOP_RANGE_IMAGE_FIRST_RETURN[:, :, 3:6]), tf.float32)

  # Run the model.
  output_tensors = model.predict(
      range_image=range_image, range_image_cartesian=range_image_cartesian)

  # Return the Tensors converted into numpy arrays.
  return {
      # Take the first example to go from 1 x N (x 7) to N (x 7).
      'boxes': output_tensors['box_3d'][0, ...].numpy(),
      'scores': output_tensors['box_score'][0, ...].numpy(),
      # Add a "classes" field that is always CAR.
      'classes': np.full(output_tensors['box_3d'].shape[1], 1, dtype=np.uint8),
  }


