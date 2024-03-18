# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from waymo_open_dataset.wdl_limited.sim_agents_metrics import trajectory_features


class TrajectoryFeaturesTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'tensor1D', 'tshape': (5,), 'pad_value': 0.0},
      {'testcase_name': 'nonzero_value', 'tshape': (5,), 'pad_value': 5.0},
      {'testcase_name': 'tensor2D', 'tshape': (3, 5), 'pad_value': 0.0},
      {'testcase_name': 'tensor_length=1', 'tshape': (1,), 'pad_value': 5.0},
  )
  def test_central_diff_returns_correct_broadcasted_values(
      self, tshape, pad_value
  ):
    # Create range tensor over the last axis, broadcast by repeating for all
    # the other dimensions.
    t = tf.range(0, tshape[-1], dtype=tf.float32)
    t = tf.broadcast_to(t, tshape)
    diff_t = trajectory_features.central_diff(t, pad_value=pad_value)
    self.assertDTypeEqual(diff_t, tf.float32)
    self.assertAllClose(
        diff_t[..., 0], tf.fill(diff_t[..., 0].shape, pad_value)
    )
    self.assertAllClose(
        diff_t[..., -1], tf.fill(diff_t[..., -1].shape, pad_value)
    )
    self.assertAllClose(
        diff_t[..., 1:-1], tf.fill(diff_t[..., 1:-1].shape, 1.0)
    )

  def test_central_diff_returns_correct_hardcoded_values(self):
    t = tf.constant([0.0, 0.5, 1.5, 2.0, 3.5])
    diff_t = trajectory_features.central_diff(t, pad_value=7.0)
    self.assertAllClose(diff_t, tf.constant([7.0, 0.75, 0.75, 1.0, 7.0]))

  @parameterized.named_parameters(
      {'testcase_name': 'tensor1D_false', 'tshape': (5,), 'pad_value': False},
      {'testcase_name': 'tensor1D_true', 'tshape': (5,), 'pad_value': True},
      {'testcase_name': 'tensor2D', 'tshape': (3, 5), 'pad_value': False},
  )
  def test_central_logical_and_returns_correct_broadcasted_values(
      self, tshape, pad_value
  ):
    # Create range tensor over the last axis, broadcast by repeating for all
    # the other dimensions.
    t = tf.fill(tshape, True)
    diff_t = trajectory_features.central_logical_and(t, pad_value=pad_value)
    self.assertDTypeEqual(diff_t, tf.bool)
    self.assertAllEqual(
        diff_t[..., 0], tf.fill(diff_t[..., 0].shape, pad_value)
    )
    self.assertAllEqual(
        diff_t[..., -1], tf.fill(diff_t[..., -1].shape, pad_value)
    )
    self.assertAllEqual(
        diff_t[..., 1:-1], tf.fill(diff_t[..., 1:-1].shape, True)
    )

  def test_central_logical_and_returns_correct_hardcoded_values(self):
    t = tf.constant([True, False, True, True, False])
    diff_t = trajectory_features.central_logical_and(t, pad_value=True)
    self.assertAllEqual(diff_t, tf.constant([True, True, False, False, True]))

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
    # Check that the last element of speed is nan (unknown final step). True
    # only for central difference.
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_speed[-1:])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_speed[-1:])))
    # Check that the 1st and 2nd elements of accel are nan (unknown 2 initial
    # steps).
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_accel[:2])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_accel[:2])))
    # Check that the last and second to last elements of accel are nan
    # (unknown 2 initial and finale steps). True only for central difference.
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_accel[-2:])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_accel[-2:])))

    # The `linear_speed` needs to be a constant of 3D euclidean norm of the
    # (x, y, z) components of the speed. `linear_accel` needs to be zero..
    self.assertAllClose(
        linear_speed[1:-1],
        tf.fill(linear_speed[1:-1].shape, tf.sqrt(1.0 + 4.0 + 9.0)),
        atol=1e-3,
    )
    self.assertAllClose(
        linear_accel[2:-2], tf.zeros_like(linear_accel[2:-2]), atol=1e-3
    )
    # The `angular_speed` needs to match the value defined in `speed`. The
    # `angular_accel` needs to be zero.
    self.assertAllClose(
        angular_speed[1:-1], tf.fill(angular_speed[1:-1].shape, 5.0), atol=1e-3
    )
    self.assertAllClose(
        angular_accel[2:-2], tf.zeros_like(angular_accel[2:-2]), atol=1e-3
    )

  def test_compute_kinematic_features_returns_const_linear_accel(self):
    seconds_per_step = 0.1
    accel = tf.constant([1.0, 2.0, 3.0, 0.5])
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
    # Check that the last element of speed is nan (unknown final step). True
    # only for central difference.
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_speed[-1:])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_speed[-1:])))
    # Check that the 1st and 2nd elements of accel are nan (unknown 2 initial
    # steps).
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_accel[:2])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_accel[:2])))
    # Check that the last and second to last elements of accel are nan
    # (unknown 2 initial and finale steps). True only for central difference.
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_accel[-2:])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_accel[-2:])))

    # The `linear_accel` needs to be a constant of 3D euclidean norm of the
    # (x, y, z) components of the acceleration.
    self.assertAllClose(
        linear_accel[2:-2],
        tf.fill(linear_accel[2:-2].shape, tf.sqrt(1.0 + 4.0 + 9.0)),
        atol=1e-3,
    )
    # The `angular_accel` needs to be a constant equal to the last value of
    # `accel`.
    self.assertAllClose(
        angular_accel[2:-2], tf.fill(angular_accel[2:-2].shape, 0.5), atol=1e-3
    )

  def test_compute_kinematic_features_returns_hardcoded_values(self):
    seconds_per_step = 1.0
    states = tf.constant([
        [5.0, 0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0, 1.0],
        [8.0, 0.0, 0.0, 3.0],
        [9.0, 0.0, 0.0, 2.0],
        [9.0, 0.0, 0.0, 0.0],
    ])

    # All these 4 tensors will have shape: (num_steps=4,)
    linear_speed, linear_accel, angular_speed, angular_accel = (
        trajectory_features.compute_kinematic_features(
            x=states[..., 0], y=states[..., 1], z=states[..., 2],
            heading=states[..., 3], seconds_per_step=seconds_per_step))

    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_speed[:1])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_speed[-1:])))
    self.assertAllClose(linear_speed[1:-1], tf.constant([1.5, 1.5, 0.5]))

    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_accel[:2])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(linear_accel[-2:])))
    self.assertAllClose(linear_accel[2:-2], tf.constant([-0.5]))

    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_speed[:1])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_speed[-1:])))
    self.assertAllClose(angular_speed[1:-1], tf.constant([1.5, 0.5, -1.5]))

    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_accel[:2])))
    self.assertTrue(tf.reduce_all(tf.math.is_nan(angular_accel[-2:])))
    self.assertAllClose(angular_accel[2:-2], tf.constant([-1.5]))

  def test_compute_kinematic_validity_returns_correct_values(self):
    validity = tf.constant([
        [True, True, True, True, True],
        [False, True, True, False, False],
        [False, True, False, True, False],
        [True, False, True, False, True],
    ])
    expected_speed_validity = tf.constant([
        [False, True, True, True, False],
        [False, False, False, False, False],
        [False, False, True, False, False],
        [False, True, False, True, False],
    ])
    expected_acceleration_validity = tf.constant([
        [False, False, True, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, True, False, False],
    ])

    speed_validity, acceleration_validity = (
        trajectory_features.compute_kinematic_validity(validity)
    )

    with self.subTest('speed_validity'):
      self.assertShapeEqual(speed_validity, validity)
      self.assertAllEqual(speed_validity, expected_speed_validity)

    with self.subTest('acceleration_validity'):
      self.assertShapeEqual(acceleration_validity, validity)
      self.assertAllEqual(acceleration_validity, expected_acceleration_validity)


if __name__ == '__main__':
  tf.test.main()
