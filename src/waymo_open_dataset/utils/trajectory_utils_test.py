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
"""Tests for waymo_open_dataset.utils.trajectories."""

import tensorflow as tf

from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils import trajectory_utils


def _empty_trajectories_with_object_ids(
    object_ids: tf.Tensor) -> trajectory_utils.ObjectTrajectories:
  return trajectory_utils.ObjectTrajectories(
      x=tf.zeros((1,)), y=tf.zeros((1,)), z=tf.zeros((1,)),
      heading=tf.zeros((1,)), valid=tf.zeros((1,)), length=tf.zeros((1,)),
      width=tf.zeros((1,)), height=tf.zeros((1,)), object_type=tf.zeros((1,)),
      object_id=object_ids)


class TrajectoriesTest(tf.test.TestCase):

  def _validate_trajectories_tensor_shape(
      self, trajectories, n_objects, n_steps):
    time_fields = [
        'x', 'y', 'z', 'heading', 'valid', 'length', 'width', 'height'
    ]
    for field in time_fields:
      self.assertEqual(getattr(trajectories, field).shape, (n_objects, n_steps))
    self.assertEqual(trajectories.object_id.shape, (n_objects,))
    self.assertEqual(trajectories.object_type.shape, (n_objects,))

  def test_scenario_to_trajectories_correctly_returns(self):
    test_scenario = test_utils.get_womd_test_scenario()

    trajectories = trajectory_utils.ObjectTrajectories.from_scenario(
        test_scenario)

    self._validate_trajectories_tensor_shape(
        trajectories, n_objects=83, n_steps=91)

  def test_slice_time_correctly_slices(self):
    test_scenario = test_utils.get_womd_test_scenario()
    trajectories = trajectory_utils.ObjectTrajectories.from_scenario(
        test_scenario)

    sliced_trajectories = trajectories.slice_time(0, 10)

    self._validate_trajectories_tensor_shape(
        sliced_trajectories, n_objects=83, n_steps=10)

  def test_gather_objects_correctly_selects_objects(self):
    test_scenario = test_utils.get_womd_test_scenario()
    trajectories = trajectory_utils.ObjectTrajectories.from_scenario(
        test_scenario)
    track_indices = tf.convert_to_tensor([0, 6, 64])

    sliced_trajectories = trajectories.gather_objects(track_indices)

    self._validate_trajectories_tensor_shape(
        sliced_trajectories, n_objects=3, n_steps=91)
    self.assertAllEqual(
        sliced_trajectories.object_id.numpy(),
        trajectories.object_id.numpy()[track_indices.numpy()],
    )

  def test_gather_object_by_id_correctly_gathers(self):
    test_scenario = test_utils.get_womd_test_scenario()
    trajectories = trajectory_utils.ObjectTrajectories.from_scenario(
        test_scenario)
    # These IDs correspond to the same track indices as above.
    objects_to_gather = tf.convert_to_tensor([1580, 1603, 1707])

    sliced_trajectories = trajectories.gather_objects_by_id(objects_to_gather)

    self._validate_trajectories_tensor_shape(
        sliced_trajectories, n_objects=3, n_steps=91)
    self.assertAllEqual(sliced_trajectories.object_id, objects_to_gather)

  def test_gather_object_by_id_fails_on_wrong_object_id_rank(self):
    test_scenario = test_utils.get_womd_test_scenario()
    trajectories = trajectory_utils.ObjectTrajectories.from_scenario(
        test_scenario)
    objects_to_gather = tf.convert_to_tensor([[1580, 1603, 1707]])

    with self.assertRaisesRegex(
        ValueError, 'Tensor  must have rank 1.  Received rank 2'):
      trajectories.gather_objects_by_id(objects_to_gather)

  def test_gather_object_by_id_fails_on_wrong_trajectory_object_id_rank(self):
    trajectories = _empty_trajectories_with_object_ids(
        tf.convert_to_tensor([[1580, 1603, 1707]]))
    objects_to_gather = tf.convert_to_tensor([1580, 1603, 1707])

    with self.assertRaisesRegex(
        ValueError, 'Tensor  must have rank 1.  Received rank 2'):
      trajectories.gather_objects_by_id(objects_to_gather)

  def test_gather_object_by_id_failson_missing_items(self):
    trajectories = _empty_trajectories_with_object_ids(
        tf.convert_to_tensor([1580, 1603]))
    objects_to_gather = tf.convert_to_tensor([1580, 1603, 1707])

    with self.assertRaisesRegex(
        ValueError,
        'Some items in `reference_tensor` are missing from `tensor`.'):
      trajectories.gather_objects_by_id(objects_to_gather)

  def test_gather_object_by_id_failson_repeated_items(self):
    trajectories = _empty_trajectories_with_object_ids(
        tf.convert_to_tensor([1580, 1580, 1603, 1707]))
    objects_to_gather = tf.convert_to_tensor([1580, 1603, 1707])

    with self.assertRaisesRegex(
        ValueError,
        'Some items in `tensor` are repeated.'):
      trajectories.gather_objects_by_id(objects_to_gather)


if __name__ == '__main__':
  tf.random.set_seed(42)
  tf.test.main()
