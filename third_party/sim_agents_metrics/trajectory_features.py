# Copyright 2023 The Waymo Open Dataset Authors.
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
# =============================================================================
"""Utils for computing trajectory-based metrics and features for sim agents."""

from typing import Tuple, Union
import numpy as np
import tensorflow as tf


def one_step_diff(
    t: tf.Tensor, prepend_value: Union[float, bool]) -> tf.Tensor:
  """Computes the 1-step discrete difference along the last axis.

  This function is used to compute 1st and 2nd order derivatives of time-series.

  Args:
    t: A float Tensor of shape [..., steps].
    prepend_value: To maintain the original tensor shape, this value is
      prepended once to the 1-step difference.

  Returns:
    A Tensor of shape [..., steps] containing the 1-step differences, prepended
    with `prepend_value`.
  """
  # Prepare the tensor containing the value(s) to prepend.
  prepend_shape = (*t.shape[:-1], 1)
  prepend_tensor = tf.fill(prepend_shape, prepend_value)
  return tf.concat([prepend_tensor, tf.experimental.numpy.diff(t)], axis=-1)


def compute_displacement_error(
    x: tf.Tensor, y: tf.Tensor, z: tf.Tensor,
    ref_x: tf.Tensor, ref_y: tf.Tensor, ref_z: tf.Tensor) -> tf.Tensor:
  """Computes displacement error (in x,y,z) w.r.t. a reference trajectory.

  Note: This operation doesn't put any constraint on the shape of the tensors,
  except that they are all consistent with each other, so this can be used
  with any arbitrary tensor shape.

  Args:
    x: The x-component of the predicted trajectories.
    y: The y-component of the predicted trajectories.
    z: The z-component of the predicted trajectories.
    ref_x: The x-component of the reference trajectories.
    ref_y: The y-component of the reference trajectories.
    ref_z: The z-component of the reference trajectories.

  Returns:
    A float tensor with the same shape as all the arguments, containing
    the 3D distance between the predicted trajectories and the reference
    trajectories.
  """
  return tf.linalg.norm(
      tf.stack([x, y, z], axis=-1) - tf.stack([ref_x, ref_y, ref_z], axis=-1),
      ord='euclidean', axis=-1)


def compute_kinematic_features(
    x: tf.Tensor,
    y: tf.Tensor,
    z: tf.Tensor,
    heading: tf.Tensor,
    seconds_per_step: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes kinematic features (speeds and accelerations).

  Note: Everything is assumed to be valid, filtering must be done afterwards.
  To maintain the original tensor length, speeds are prepended with 1 np.nan,
  while accelerations with 2 np.nan.

  Args:
    x: A float tensor of shape (..., num_steps) containing x coordinates.
    y: A float tensor of shape (..., num_steps) containing y coordinates.
    z: A float tensor of shape (..., num_steps) containing z coordinates.
    heading: A float tensor of shape (..., num_steps,) containing heading.
    seconds_per_step: The duration (in seconds) of one step. This is used to
      scale speed and acceleration properly. This is always a positive value,
      usually defaulting to `submission_specs.STEP_DURATION_SECONDS`.

  Returns:
    A tuple containing the following 4 tensors:
      linear_speed: Magnitude of speed in (x, y, z). Shape (..., num_steps).
      linear_acceleration: Linear signed acceleration (changes in linear speed).
        Shape (..., num_steps).
      angular_speed: Angular speed (changes in heading). Shape (..., num_steps).
      angular_acceleration: Angular acceleration (changes in `angular_speed`).
        Shape (..., num_steps).
  """
  # First order derivatives.
  dpos = one_step_diff(tf.stack([x, y, z], axis=0), prepend_value=np.nan)
  linear_speed = tf.linalg.norm(
      dpos, ord='euclidean', axis=0) / seconds_per_step
  dh = _wrap_angle(
      one_step_diff(heading, prepend_value=np.nan)) / seconds_per_step
  # Second order derivatives.
  linear_accel = one_step_diff(
      linear_speed, prepend_value=np.nan) / seconds_per_step
  d2h = _wrap_angle(one_step_diff(dh, prepend_value=np.nan)) / seconds_per_step
  return linear_speed, linear_accel, dh, d2h


def _wrap_angle(angle: tf.Tensor) -> tf.Tensor:
  """Wraps angles in the range [-pi, pi]."""
  return (angle + np.pi) % (2 * np.pi) - np.pi
