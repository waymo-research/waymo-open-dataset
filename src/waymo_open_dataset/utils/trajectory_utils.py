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
"""Library for trajectory related features and util functions."""

from __future__ import annotations

import dataclasses
from typing import Optional
import tensorflow as tf

from waymo_open_dataset.protos import scenario_pb2


@dataclasses.dataclass(frozen=True)
class ObjectTrajectories:
  """Trajectories of 3D boxes representing the objects in the world.

  While all the tensors in `ObjectTrajectories` are expected to have consistent
  dimensions (as documented below), no shapes are enforced. Users making a
  custom use of this class must be aware of the tensor shapes and how the
  tensors are treated by the methods in this class.

  Attributes:
    x: Tensor containing the x-component of the object positions in the world.
      Same convention as the original WOMD Scenario data.
      Dtype: tf.float32, shape: (n_objects, n_steps).
    y: Y-component of the object positions.
      Dtype: tf.float32, shape: (n_objects, n_steps).
    z: Z-component of the object positions.
      Dtype: tf.float32, shape: (n_objects, n_steps).
    heading: Heading of the object (see Scenario proto for definition).
      Dtype: tf.float32, shape: (n_objects, n_steps).
    length: Lengths of the object boxes over time.
      Dtype: tf.float32, shape: (n_objects, n_steps).
    width: Widths of the object boxes over time.
      Dtype: tf.float32, shape: (n_objects, n_steps).
    height: Heights of the object boxes over time.
      Dtype: tf.float32, shape: (n_objects, n_steps).
    valid: Validity of the objects over time.
      Dtype: tf.bool, shape (n_objects, n_steps).
    object_id: Int tensor of object IDs, as specified in the Scenario proto.
      Dtype: tf.int32, shape (n_objects,).
    object_type: Int tensor of object types, as specified in the Scenario proto
      (`scenario_pb2.Track.ObjectType`).
      Dtype: tf.int32, shape (n_objects,).
  """
  x: tf.Tensor
  y: tf.Tensor
  z: tf.Tensor
  heading: tf.Tensor
  length: tf.Tensor
  width: tf.Tensor
  height: tf.Tensor
  valid: tf.Tensor
  object_id: tf.Tensor
  object_type: tf.Tensor

  def slice_time(self: ObjectTrajectories,
                 start_index: int = 0,
                 end_index: Optional[int] = None) -> ObjectTrajectories:
    """Slices all the `ObjectTrajectories` tensors in the time axis.

    Args:
      start_index: Start index for the slice, in steps.
      end_index: End index for the slice, in steps. Following the Python
        convention, if None is passed the slice will span until the last time
        step.

    Returns:
      An `ObjectTrajectories` object where all the time-series tensors
      (position, dimensions and valididity) has been sliced in time (last axis).
    """
    return ObjectTrajectories(
        x=self.x[..., start_index:end_index],
        y=self.y[..., start_index:end_index],
        z=self.z[..., start_index:end_index],
        heading=self.heading[..., start_index:end_index],
        length=self.length[..., start_index:end_index],
        width=self.width[..., start_index:end_index],
        height=self.height[..., start_index:end_index],
        valid=self.valid[..., start_index:end_index],
        object_id=self.object_id,
        object_type=self.object_type,
    )

  def gather_objects(self: ObjectTrajectories,
                     object_indices: tf.Tensor) -> ObjectTrajectories:
    """Gathers all given indices of `ObjectTrajectories` in the objects axis.

    Args:
      object_indices: A 1D tensor containing the indices of the objects to be
        selected. If the indices are out of range an error will be raised by
        tf.gather().

    Returns:
      An `ObjectTrajectories` object where all the tensors have been gathered
      over the object dimension (second-last axis for time-series, last axis for
      per-object values), selecting the specified indices. This has the same
      behaviour of tf.gather(), so indices can be selected more than once.
    """
    tf.assert_rank(object_indices, 1)
    return ObjectTrajectories(
        x=tf.gather(self.x, object_indices, axis=-2),
        y=tf.gather(self.y, object_indices, axis=-2),
        z=tf.gather(self.z, object_indices, axis=-2),
        heading=tf.gather(self.heading, object_indices, axis=-2),
        length=tf.gather(self.length, object_indices, axis=-2),
        width=tf.gather(self.width, object_indices, axis=-2),
        height=tf.gather(self.height, object_indices, axis=-2),
        valid=tf.gather(self.valid, object_indices, axis=-2),
        object_id=tf.gather(self.object_id, object_indices, axis=-1),
        object_type=tf.gather(self.object_type, object_indices, axis=-1),
    )

  def gather_objects_by_id(self: ObjectTrajectories,
                           object_ids: tf.Tensor) -> ObjectTrajectories:
    """Gather all given object IDs of `ObjectTrajectories` in the objects axis.

    Since this is based on a tf.gather() operation, both the `object_id` tensor
    inside `ObjectTrajectories` and the arg `object_ids` must be 1D. Moreover,
    all the object IDs to be selected need to be present in the original
    `ObjectTrajectories`.

    Args:
      object_ids: An int 1D tensor containing all the object IDs to be gathered.

    Returns:
      An `ObjectTrajectories` object, gathered on the object dimension, to
      match the object IDs provided in `object_ids` (with the same ordering).
    """
    indices = _arg_gather(self.object_id, object_ids)
    return self.gather_objects(indices)

  @classmethod
  def from_scenario(cls, scenario: scenario_pb2.Scenario) -> ObjectTrajectories:
    """Extracts `ObjectTrajectories` from a Scenario proto.

    Args:
      scenario: A Scenario proto.

    Returns:
      An `ObjectTrajectories` containing the trajectories of all the objects in
      a scenario. Note: n_steps is 91 for the train and validation set, 11 for
      the test set.
    """
    states, dimensions, objects = [], [], []
    for track in scenario.tracks:
      # Iterate over a single object's states.
      track_states, track_dimensions = [], []
      for state in track.states:
        track_states.append((state.center_x, state.center_y, state.center_z,
                             state.heading, state.valid))
        track_dimensions.append((state.length, state.width, state.height))
      # Adds to the global states.
      states.append(list(zip(*track_states)))
      dimensions.append(list(zip(*track_dimensions)))
      objects.append((track.id, track.object_type))

    # Unpack and convert to tf tensors.
    x, y, z, heading, valid = [tf.convert_to_tensor(s) for s in zip(*states)]
    length, width, height = [tf.convert_to_tensor(s) for s in zip(*dimensions)]
    object_ids, object_types = [tf.convert_to_tensor(s) for s in zip(*objects)]
    return ObjectTrajectories(x=x, y=y, z=z, heading=heading,
                              length=length, width=width, height=height,
                              valid=valid, object_id=object_ids,
                              object_type=object_types)


def _arg_gather(
    tensor: tf.Tensor, reference_tensor: tf.Tensor) -> tf.Tensor:
  """Finds corresponding idxs in `tensor` for each element in `reference_tensor.

  This function returns the arguments for a gather op such that:
    tf.gather(tensor, indices=_arg_gather(tensor, reference_tensor))
    == reference_tensor.

  Note: All the items in `tensor` must be present in `reference_tensor`,
    otherwise the 0-indexed element of `tensor` will be returned, invalidating
    the expected equivalence above. Moreover, `tensor` must be without
    repetitions to guarantee the correct result.

  Args:
    tensor: The tensor to map. Must be a 1D tensor because the logic used here
      does not apply to multi-dimensional gather ops.
    reference_tensor: A 1D tensor containing items from `tensor`, each of which
      will be searched for in `tensor`.

  Returns:
    The list of indices on `tensor` which, if taken, maps directly to
    `reference_tensor`. Specifically, if we apply tf.gather(tensor, indices) we
    obtain the reference tensor back.
  """
  tf.assert_rank(tensor, 1)
  tf.assert_rank(reference_tensor, 1)
  # Create the comparison matrix and verify matching items.
  bit_mask = tensor[tf.newaxis, :] == reference_tensor[:, tf.newaxis]
  bit_mask_sum = tf.reduce_sum(tf.cast(bit_mask, tf.int32), axis=1)
  if tf.reduce_any(bit_mask_sum < 1):
    raise ValueError(
        'Some items in `reference_tensor` are missing from `tensor`:'
        f' \n{reference_tensor} \nvs. \n{tensor}.'
    )
  if tf.reduce_any(bit_mask_sum > 1):
    raise ValueError('Some items in `tensor` are repeated.')
  return tf.matmul(
      tf.cast(bit_mask, tf.int32),
      tf.range(0, tensor.shape[0], dtype=tf.int32)[:, tf.newaxis])[:, 0]
