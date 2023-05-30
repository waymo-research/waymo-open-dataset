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
"""Object asset components."""

import dataclasses
from typing import Optional

import pyarrow as pa
import tensorflow as tf

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2 import component
from waymo_open_dataset.v2.perception import base

_column = component.create_column


@dataclasses.dataclass(frozen=True)
class LidarDistanceImage:
  """A [H, W] tensor of the liDAR projection distance on the camera image.

  Note: each pixel stores the distance (in metres) between the camera ray origin
    and the point of the first LiDAR return. The value becomes zero if there is
    no return.

  Attributes:
    values: A list of float values of the flattened distance image.
    shape: The range image shape. Should be [H, W].
    tensor: A tf.Tensor with shaped values.
  """

  values: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32(), 2))

  @property
  def tensor(self) -> tf.Tensor:
    return tf.reshape(tf.convert_to_tensor(self.values), self.shape)


@dataclasses.dataclass(frozen=True)
class CameraRayImage:
  """A [H, W, 3] tensor of the ray origin and ray direction.

  Note: the ray origin and ray direction are defined in the vehicle
    coordinate frame synced to the frame start.
  Attributes:
    values: A list of float values of the flattened ray image.
    shape: The range image shape. Should be [H, W, 3].
    tensor: A tf.Tensor with shaped values.
  """

  values: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32(), 3))

  @property
  def tensor(self) -> tf.Tensor:
    return tf.reshape(tf.convert_to_tensor(self.values), self.shape)


@dataclasses.dataclass(frozen=True)
class PointCoordinates:
  """A [N, 3] tensor of the LiDAR point coordinates.

  Note: the point coordinates are defined in the vehicle coordinate frame
    at the frame start.
  Attributes:
    values: A list of float values of the flattened point coordinates.
    shape: The range image shape. Should be [N, 3].
    tensor: A tf.Tensor with shaped values.
  """

  values: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32(), 2))

  @property
  def tensor(self) -> tf.Tensor:
    return tf.reshape(tf.convert_to_tensor(self.values), self.shape)


@dataclasses.dataclass(frozen=True)
class ObjectAssetLiDARSensorComponent(component.Component):
  """Object asset lidar sensor and corresponding vehicle pose component.

  Note: we only store the LiDAR points related to the object within the 7-DOF
    Box.
  Attributes:
    points_xyz: A PointCoordinates instance, representing the object 3D points
      within the 7-DOF Box in the vehicle frame, synced to the frame start.
  """

  key: base.LaserLabelKey
  points_xyz: PointCoordinates = _column()


@dataclasses.dataclass(frozen=True)
class ObjectAssetCameraSensorComponent(component.Component):
  """Object asset camera sensor and corresponding vehicle pose component.

  Note: we only store the cropped camera patch and relevant information.

  Attributes:
    camera_name: Camera name of the cropped patch.
    camera_region: A BoxAxisAligned2d instance, representing the cropped box
      region from the corresponding camera.
    rgb_image: A PNG-encoded image (uint8) of the camera patch region.
    proj_points_dist: The projected distance on the camera patch region. The
      value becomes zero for any pixel without a corresponding projection.
    proj_points_mask: A PNG-encoded mask (uint8), indicating the valid pixels of
      the 'proj_points_dist` above.
  """

  key: base.ObjectAssetKey
  camera_region: Optional[column_types.BoxAxisAligned2d] = _column(default=None)
  rgb_image: Optional[bytes] = _column(arrow_type=pa.binary(), default=None)
  proj_points_dist: Optional[LidarDistanceImage] = _column(default=None)
  proj_points_mask: Optional[bytes] = _column(
      arrow_type=pa.binary(), default=None
  )


@dataclasses.dataclass(frozen=True)
class ObjectAssetAutoLabelComponent(component.Component):
  """Object asset auto labeled components.

  These fields are inferred from model outputs as the auto-label.

  Attributes:
    semantic_mask: A PNG-encoded semantic mask (uint8), representing the dense
      semantic labels for each pixel within the camera cropped region.
    instance_mask: A PNG-encoded instance mask (uint16), representing the dense
      instance labels for each pixel within the camera cropped region.
    object_mask: A PNG-encoded object mask (uint8), indicating the object of
      interest defined by the `laser_object_id`.
  """

  key: base.ObjectAssetKey
  semantic_mask: Optional[bytes] = _column(arrow_type=pa.binary(), default=None)
  instance_mask: Optional[bytes] = _column(arrow_type=pa.binary(), default=None)
  object_mask: Optional[bytes] = _column(arrow_type=pa.binary(), default=None)


@dataclasses.dataclass(frozen=True)
class ObjectAssetRayComponent(component.Component):
  """Object asset camera ray origin and ray direction component.

  We transform each ray into the object-centered canonical frame by taking into
    account the ego-motion, object motion, and rolling shutter effect.

  Attributes:
    ray_origin: Camera ray origin defined in the object canonical frame.
    ray_direction: Camera ray direction defined in the object canonical frame.
  """

  key: base.ObjectAssetKey
  ray_origin: Optional[CameraRayImage] = _column(default=None)
  ray_direction: Optional[CameraRayImage] = _column(default=None)
