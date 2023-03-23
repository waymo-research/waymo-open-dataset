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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from waymo_open_dataset.wdl_limited.sim_agents_metrics import trajectory_features


class TrajectoryFeaturesTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'tensor1D', 'tshape': (5,), 'prepend_value': 0.0},
      {'testcase_name': 'nonzero_value', 'tshape': (5,), 'prepend_value': 5.0},
      {'testcase_name': 'tensor2D', 'tshape': (3, 5,), 'prepend_value': 0.0},
  )
  def test_one_step_diff_correct_for_floats(self, tshape, prepend_value):
    # Create range tensor over the last axis, broadcast by repeating for all
    # the other dimensions.
    t = tf.range(0, tshape[-1], dtype=tf.float32)
    t = tf.broadcast_to(t, tshape)
    diff_t = trajectory_features.one_step_diff(t, prepend_value=prepend_value)
    self.assertAllClose(diff_t[..., 0],
                        tf.fill(diff_t[..., 0].shape, prepend_value))
    self.assertAllClose(diff_t[..., 1:],
                        tf.fill(diff_t[..., 1:].shape, 1.0))

  def test_compute_displacement_error_correctly_returns_shape_and_values(self):
    t = reference = tf.random.uniform((4, 5, 3))
    ade = trajectory_features.compute_displacement_error(
        x=t[..., 0], y=t[..., 1], z=t[..., 2],
        ref_x=reference[..., 0], ref_y=reference[..., 1],
        ref_z=reference[..., 2])
    self.assertEqual(ade.shape, (4, 5))
    self.assertAllClose(ade, tf.zeros_like(ade))

  def test_displacement_error_fails_on_wrong_dimensions(self):
    t = reference = tf.random.uniform((4, 5, 3))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      trajectory_features.compute_displacement_error(
          x=t[..., 0], y=t[..., 1], z=t[..., 2],
          ref_x=reference[..., 0], ref_y=reference[..., 1],
          ref_z=tf.zeros((7, 5)))

  def test_compute_kinematic_features_returns_const_speed(self):
    seconds_per_step = 0.1
    speed = tf.constant([1., 2., 3., 5.0])
    s0 = tf.constant([5., 7., 11., 0.])
    t_steps = tf.range(0., 2., delta=seconds_per_step)
    states = s0[tf.newaxis] + t_steps[:, tf.newaxis] * speed[tf.newaxis]
    # Wrap angles to test the angular speed.
    states = tf.concat([
        states[..., 0:3],
        ((states[..., 3] + np.pi) % (2 * np.pi) - np.pi)[..., tf.newaxis]
    ], axis=-1)

    linear_speed, linear_accel, angular_speed, angular_accel = (
        trajectory_features.compute_kinematic_features(
            x=states[..., 0], y=states[..., 1], z=states[..., 2],
            heading=states[..., 3], seconds_per_step=seconds_per_step))

    # Check that the 1st element of speed is nan (unknown initial step).
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_speed[:1])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_speed[:1])))
    # Check that the 1st and 2nd elements of accel are nan (unknown 2 initial)
    # steps.
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_accel[:2])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_accel[:2])))
    # The `linear_speed` needs to be a constant of 3D euclidean norm of the
    # (x, y, z) components of the speed. `linear_accel` needs to be zero..
    self.assertAllClose(linear_speed[1:],
                        tf.fill(linear_speed[1:].shape, tf.sqrt(1.+4.+9.)),
                        atol=1e-3)
    self.assertAllClose(linear_accel[2:], tf.zeros_like(linear_accel[2:]),
                        atol=1e-3)
    # The `angular_speed` needs to match the value defined in `speed`. The
    # `angular_accel` needs to be zero.
    self.assertAllClose(angular_speed[1:],
                        tf.fill(angular_speed[1:].shape, 5.0),
                        atol=1e-3)
    self.assertAllClose(angular_accel[2:], tf.zeros_like(angular_accel[2:]),
                        atol=1e-3)

  def test_compute_kinematic_features_returns_const_linear_accel(self):
    seconds_per_step = 0.1
    accel = tf.constant([1., 2., 3., 5.0])
    s0 = tf.constant([0., 0., 0., 0.])
    t_steps = tf.range(0., 2., delta=seconds_per_step)
    # We use a simple formula for constant acceleration and zero initial speed
    # to compute the states. s = 1/2 * a * t**2 + s0.
    states = (
        s0[tf.newaxis] + 0.5 * t_steps[:, tf.newaxis]**2 * accel[tf.newaxis])

    # All these 4 tensors will have shape: (num_steps=20,)
    linear_speed, linear_accel, angular_speed, angular_accel = (
        trajectory_features.compute_kinematic_features(
            x=states[..., 0], y=states[..., 1], z=states[..., 2],
            heading=states[..., 3], seconds_per_step=seconds_per_step))

    # Check that the 1st element of speed is nan (unknown initial step).
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_speed[:1])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_speed[:1])))
    # Check that the 1st and 2nd elements of accel are nan (unknown 2 initial)
    # steps.
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_accel[:2])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_accel[:2])))
    # The `linear_accel` needs to be a constant of 3D euclidean norm of the
    # (x, y, z) components of the acceleration.
    self.assertAllClose(linear_accel[2:],
                        tf.fill(linear_accel[2:].shape, tf.sqrt(1.+4.+9.)),
                        atol=1e-3)
    # The `angular_accel` needs to be a constant equal to the last value of
    # `accel`.
    self.assertAllClose(angular_accel[2:],
                        tf.fill(angular_accel[2:].shape, 5.0),
                        atol=1e-3)

  def test_compute_kinematic_features_returns_hardcoded_values(self):
    seconds_per_step = 1.0
    states = tf.constant([
        [5., 0., 0., 0.],
        [6., 0., 0., 1.],
        [8., 0., 0., 3.],
        [9., 0., 0., 2.],
    ])

    # All these 4 tensors will have shape: (num_steps=4,)
    linear_speed, linear_accel, angular_speed, angular_accel = (
        trajectory_features.compute_kinematic_features(
            x=states[..., 0], y=states[..., 1], z=states[..., 2],
            heading=states[..., 3], seconds_per_step=seconds_per_step))

    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_speed[:1])))
    self.assertAllClose(linear_speed[1:], tf.constant([1., 2., 1.]))

    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_accel[:2])))
    self.assertAllClose(linear_accel[2:], tf.constant([1., -1.]))

    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_speed[:1])))
    self.assertAllClose(angular_speed[1:], tf.constant([1., 2., -1.]))

    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_accel[:2])))
    self.assertAllClose(angular_accel[2:], tf.constant([1., -3.]))


if __name__ == '__main__':
  tf.test.main()
