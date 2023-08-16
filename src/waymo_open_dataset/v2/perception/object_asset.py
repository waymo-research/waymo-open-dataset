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

import numpy as np
import pyarrow as pa
import tensorflow as tf

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2 import component
from waymo_open_dataset.v2.perception import base

_column = component.create_column


class _ValueShapeMixin:
  """A helper with convenience methods to output shaped tensors/arrays."""

  values: list[float]
  shape: list[int]

  @property
  def tensor(self) -> tf.Tensor:
    return tf.reshape(tf.convert_to_tensor(self.values), self.shape)

  @property
  def numpy(self) -> np.ndarray:
    return np.array(self.values).reshape(self.shape)


@dataclasses.dataclass(frozen=True)
class LidarDistanceImage(_ValueShapeMixin):
  """A [H, W] tensor of the liDAR projection distance on the camera image.

  Note: each pixel stores the distance (in metres) between the camera ray origin
    and the point of the first LiDAR return. The value becomes zero if there is
    no return.

  Attributes:
    values: A list of float values of the flattened distance image.
    shape: The range image shape. Should be [H, W].
    tensor: A tf.Tensor with shaped values.
    numpy: A np.ndarray with shaped values.
  """

  values: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32(), 2))


@dataclasses.dataclass(frozen=True)
class CameraRayImage(_ValueShapeMixin):
  """A [H, W, 3] tensor of the ray origin and ray direction.

  Note: the ray origin and ray direction are defined in the object-centered
    coordinate frame, bounded by a box with the same dimension as the labeled
    laser box.
  Attributes:
    values: A list of float values of the flattened ray image.
    shape: The range image shape. Should be [H, W, 3].
    tensor: A tf.Tensor with shaped values.
    numpy: A np.ndarray with shaped values.
  """

  values: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32(), 3))


@dataclasses.dataclass(frozen=True)
class PointCoordinates(_ValueShapeMixin):
  """A [N, 3] tensor of the LiDAR point coordinates.

  Note: the point coordinates are defined in the labeled laser box frame.
  Attributes:
    values: A list of float values of the flattened point coordinates.
    shape: The range image shape. Should be [N, 3].
    tensor: A tf.Tensor with shaped values.
    numpy: A np.ndarray with shaped values.
  """

  values: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32(), 2))


@dataclasses.dataclass(frozen=True)
class ObjectAssetLiDARSensorComponent(component.Component):
  """Object asset lidar sensor component.

  Note: we only store the LiDAR points related to the object within the 7-DOF
    Box.
  Attributes:
    points_xyz: A PointCoordinates instance, representing the object 3D points
      within the 7-DOF Box in the labeled laser box frame.
  """

  key: base.LaserLabelKey
  points_xyz: PointCoordinates = _column()


@dataclasses.dataclass(frozen=True)
class ObjectAssetCameraSensorComponent(component.Component):
  """Object asset camera sensor component.

  Note: we only store the cropped camera patch and relevant information.

  Attributes:
    camera_name: Camera name of the cropped patch.
    camera_region: A BoxAxisAligned2d instance, representing the cropped box
      region from the corresponding camera.
    rgb_image: A PNG-encoded image (uint8) within camera patch region.
    proj_points_dist: The projected distance on camera image, within the camera
      patch region. The value becomes zero for any pixel without a corresponding
      projection.
    proj_points_mask: A PNG-encoded mask (uint8), indicating the valid pixels of
      the 'proj_points_dist` above.
    rgb_image_numpy: A np.ndarray, representing the decoded RGB image within the
      camera patch region.
    proj_points_mask_numpy: A np.ndarray, representing the binary mask that
      indicating the valid pixels of the projected points, within the camera
      patch region.
  """

  key: base.ObjectAssetKey
  camera_region: Optional[column_types.BoxAxisAligned2d] = _column(default=None)
  rgb_image: Optional[bytes] = _column(arrow_type=pa.binary(), default=None)
  proj_points_dist: Optional[LidarDistanceImage] = _column(default=None)
  proj_points_mask: Optional[bytes] = _column(
      arrow_type=pa.binary(), default=None
  )

  @property
  def rgb_image_numpy(self) -> np.ndarray:
    return tf.image.decode_png(self.rgb_image).numpy()

  @property
  def proj_points_mask_numpy(self) -> np.ndarray:
    return tf.image.decode_png(self.proj_points_mask).numpy()


@dataclasses.dataclass(frozen=True)
class ObjectAssetRefinedPoseComponent(component.Component):
  """Object asset refined box pose component.

  Attributes:
    box_from_vehicle: A Transform instance, representing the refined box pose as
      a [4, 4] matrix from the SDC frame. This field is only populated for each
      vehicle object, which is further refined across the entire sequence
      through shape registration.
  """

  key: base.LaserLabelKey
  box_from_vehicle: column_types.Transform = _column(default=None)


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
    semantic_mask_numpy: A np.ndarray, representing the decoded semantic label
      maps within the camera patch region.
    instance_mask_numpy: A np.ndarray, representing the decoded instance label
      maps within the camera patch region.
    object_mask_numpy: A np.ndarray, representing the decoded per-pixel object
      mask within the camera patch region.
  """

  key: base.ObjectAssetKey
  semantic_mask: Optional[bytes] = _column(arrow_type=pa.binary(), default=None)
  instance_mask: Optional[bytes] = _column(arrow_type=pa.binary(), default=None)
  object_mask: Optional[bytes] = _column(arrow_type=pa.binary(), default=None)

  @property
  def semantic_mask_numpy(self) -> np.ndarray:
    return tf.image.decode_png(self.semantic_mask).numpy()

  @property
  def instance_mask_numpy(self) -> np.ndarray:
    return tf.image.decode_png(self.instance_mask, dtype=tf.uint16).numpy()

  @property
  def object_mask_numpy(self) -> np.ndarray:
    return tf.image.decode_png(self.object_mask).numpy()


@dataclasses.dataclass(frozen=True)
class ObjectAssetRayComponent(component.Component):
  """Object asset camera ray origin and ray direction component.

  We transform each ray into the object-centered frame by taking into account
    the ego-motion, object motion, and rolling shutter effect. For each pixel,
    we shift the labeled laser box along the heading direction (assuming the box
    is moving at a constant velocity) to the same timestamp when the pixel is
    captured. We then compute pixel ray origin and direction in the shifted box
    coordinate frame.

  Attributes:
    ray_origin: Camera ray origin defined in the object-centered frame bounded
      by a box with the same dimension as the labeled laser box.
    ray_direction: Camera ray direction defined in the object-centered frame
      bounded by a box with the same dimension as the labeled laser box.
  """

  key: base.ObjectAssetKey
  ray_origin: Optional[CameraRayImage] = _column(default=None)
  ray_direction: Optional[CameraRayImage] = _column(default=None)


@dataclasses.dataclass(frozen=True)
class ObjectAssetRayCompressedComponent(component.Component):
  """Object asset camera ray origin and ray direction compressed component.

  Use `ObjcetAssetRayCodec` to encode a
  `ObjectAssetRayComponent` into `ObjectAssetRayCompressedComponent` and decode
  back.

  The codec implements a lossy compression algorithm. Depending on which
  parameters were used to compress the data the compression ratio could reach
  ~0.07. This results in a significant reduction in dataset size, while
  introducing a negligible error (~1e-5 m) in ray origins and directions. This
  level of error is generally deemed acceptable for most applications.

  Attributes:
    reference: A reference value used to decompress camera ray origins and
      directions. It is an implementation detail and should not be used
      directly.
    quantized_values: compressed data.
    shape: shape of the original data.
  """

  key: base.ObjectAssetKey
  reference: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  quantized_values: list[int] = _column(arrow_type=pa.list_(pa.int32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32()))
