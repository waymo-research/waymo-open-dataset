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
# ==============================================================================
"""LiDAR data components."""


import dataclasses
import enum

import pyarrow as pa
import tensorflow as tf

from waymo_open_dataset.v2 import component
from waymo_open_dataset.v2.perception import base


_column = component.create_column


@dataclasses.dataclass(frozen=True)
class RangeImage:
  """A [H, W, 4] tensor of range image.

  Inner dimensions are:
    * channel 0: range
    * channel 1: intensity
    * channel 2: elongation
    * channel 3: is in any no label zone.

  Attributes:
    values: A list of float values of the flattened range image.
    shape: The range image shape. Should be [H, W, 4].
    tensor: A tf.Tensor with shaped values.
  """
  values: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32(), 3))

  @property
  def tensor(self) -> tf.Tensor:
    return tf.reshape(tf.convert_to_tensor(self.values), self.shape)


@dataclasses.dataclass(frozen=True)
class LiDARComponent(base.LaserComponent):
  """A dataset component for LiDAR sensor data in range image format."""
  range_image_return1: RangeImage = _column(default=None)
  range_image_return2: RangeImage = _column(default=None)

  @property
  def range_image_returns(self) -> list[RangeImage]:
    return [self.range_image_return1, self.range_image_return2]


@dataclasses.dataclass(frozen=True)
class CameraProjectionRangeImage:
  """A [H, W, 6] tensor of camera projection of each pixel in range image.

  Lidar point to camera image projections. A point can be projected to multiple
  camera images. We pick the first two at the following order:
  [FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT].

  Inner dimensions are:
    * channel 0: camera name of 1st projection. Set to UNKNOWN if no
        projection.
    * channel 1: x (axis along image width)
    * channel 2: y (axis along image height)
    * channel 3: camera name of 2nd projection. Set to UNKNOWN if no
        projection.
    * channel 4: x (axis along image width)
    * channel 5: y (axis along image height)
  Note: pixel 0 corresponds to the left edge of the first pixel in the image.

  Attributes:
    values: A list of float values of the flattened camera projection range
      image.
    shape: The range image shape. Should be [H, W, 6].
    tensor: A tf.Tensor with shaped values.
  """
  values: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32(), 3))

  @property
  def tensor(self) -> tf.Tensor:
    return tf.reshape(tf.convert_to_tensor(self.values), self.shape)


@dataclasses.dataclass(frozen=True)
class LiDARCameraProjectionComponent(base.LaserComponent):
  """A dataset component for camera projections for each range image pixel."""
  range_image_return1: CameraProjectionRangeImage = _column()
  range_image_return2: CameraProjectionRangeImage = _column()

  @property
  def range_image_returns(self) -> list[CameraProjectionRangeImage]:
    return [self.range_image_return1, self.range_image_return2]


@dataclasses.dataclass(frozen=True)
class PoseRangeImage:
  """A [H, W, 6] tensor of pose of each pixel in range image.

  Inner dimensions are [roll, pitch, yaw, x, y, z] represents a transform
  from vehicle frame to global frame for every range image pixel.
  This is ONLY populated for the first return. The second return is assumed
  to have exactly the same pose.

  The roll, pitch and yaw are specified as 3-2-1 Euler angle rotations,
  meaning that rotating from the navigation to vehicle frame consists of a
  yaw, then pitch and finally roll rotation about the z, y and x axes
  respectively. All rotations use the right hand rule and are positive
  in the counter clockwise direction.

  Attributes:
    values: A list of float values of the flattened per-pixel pose in range
      image.
    shape: The range image shape. Should be [H, W, 6].
    tensor: A tf.Tensor with shaped values.
  """
  values: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32(), 3))

  @property
  def tensor(self) -> tf.Tensor:
    return tf.reshape(tf.convert_to_tensor(self.values), self.shape)


@dataclasses.dataclass(frozen=True)
class LiDARPoseComponent(base.LaserComponent):
  """A dataset component for pose of each range image pixel.

  NOTE: 'range_image_pose_compressed' is only populated for the first range
  image return. The second return has the exact the same range image pose as the
  first one.
  """
  range_image_return1: PoseRangeImage = _column()


@dataclasses.dataclass(frozen=True)
class FlowRangeImage:
  """A [H, W, 4] tensor of scene flow of each pixel in range image.

  If the point is not annotated with scene flow information, class is set
  to -1. A point is not annotated if it is in a no-label zone or if its label
  bounding box does not have a corresponding match in the previous frame,
  making it infeasible to estimate the motion of the point.
  Otherwise, (vx, vy, vz) are velocity along (x, y, z)-axis for this point
  and class is set to one of the following values:
   -1: no-flow-label, the point has no flow information.
    0: unlabeled or "background,", i.e., the point is not contained in a
       bounding box.
    1: vehicle, i.e., the point corresponds to a vehicle label box.
    2: pedestrian, i.e., the point corresponds to a pedestrian label box.
    3: sign, i.e., the point corresponds to a sign label box.
    4: cyclist, i.e., the point corresponds to a cyclist label box.

  Attributes:
    values: A list of float values of the flattened flow range image.
    shape: The range image shape. Should be [H, W, 4].
    tensor: A tf.Tensor with shaped values.
  """
  values: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32(), 3))

  @property
  def tensor(self) -> tf.Tensor:
    return tf.reshape(tf.convert_to_tensor(self.values), self.shape)


@dataclasses.dataclass(frozen=True)
class LiDARFlowComponent(base.LaserComponent):
  """A dataset component for scene flows."""
  range_image_return1: FlowRangeImage = _column()
  range_image_return2: FlowRangeImage = _column()

  @property
  def range_image_returns(self) -> list[FlowRangeImage]:
    return [self.range_image_return1, self.range_image_return2]


class LaserName(enum.Enum):
  """Name of laser/LiDAR."""
  UNKNOWN = 0
  TOP = 1
  FRONT = 2
  SIDE_LEFT = 3
  SIDE_RIGHT = 4
  REAR = 5

