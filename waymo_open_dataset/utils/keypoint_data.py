# Copyright 2022 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""Data structures to represent and tools to parse human keypoint data."""
import dataclasses
import enum
from typing import Collection, Dict, List, Mapping, Optional

import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import keypoint_pb2

KeypointType = keypoint_pb2.KeypointType

# Canonical order of keypoint types for different modalities in various
# representations, where coordinates or visibility flags for all keypoints are
# stored as a single tensor or vector.
_SHARED_TYPES = (KeypointType.KEYPOINT_TYPE_NOSE,
                 KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER,
                 KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER,
                 KeypointType.KEYPOINT_TYPE_LEFT_ELBOW,
                 KeypointType.KEYPOINT_TYPE_RIGHT_ELBOW,
                 KeypointType.KEYPOINT_TYPE_LEFT_WRIST,
                 KeypointType.KEYPOINT_TYPE_RIGHT_WRIST,
                 KeypointType.KEYPOINT_TYPE_LEFT_HIP,
                 KeypointType.KEYPOINT_TYPE_RIGHT_HIP,
                 KeypointType.KEYPOINT_TYPE_LEFT_KNEE,
                 KeypointType.KEYPOINT_TYPE_RIGHT_KNEE,
                 KeypointType.KEYPOINT_TYPE_LEFT_ANKLE,
                 KeypointType.KEYPOINT_TYPE_RIGHT_ANKLE)
CANONICAL_ORDER_CAMERA = _SHARED_TYPES + (KeypointType.KEYPOINT_TYPE_FOREHEAD,)
# Refer to waymo_open_dataset/protos/keypoint.proto for a description of the
# differences between head center and forehead keypoints.
CANONICAL_ORDER_LASER = _SHARED_TYPES + (
    KeypointType.KEYPOINT_TYPE_HEAD_CENTER,)
CANONICAL_ORDER_ALL = _SHARED_TYPES + (KeypointType.KEYPOINT_TYPE_FOREHEAD,
                                       KeypointType.KEYPOINT_TYPE_HEAD_CENTER)


@dataclasses.dataclass(frozen=True)
class Point:
  """Point location in space.

  Attributes:
    location: 2D or 3D coordinate.
    is_occluded: Is True, if the keypoint is occluded by any object or a body
      part. Is False if the keypoint is clearly visible
  """
  location: np.ndarray
  is_occluded: bool = False


CameraType = dataset_pb2.CameraName.Name
ObjectType = label_pb2.Label.Type
KeypointType = keypoint_pb2.KeypointType
PointByType = Mapping['KeypointType', Point]


@dataclasses.dataclass(frozen=True)
class LaserLabel:
  """A dataclass to store laser labels for keypoint visualization.

  Attributes:
    object_type: type of the object.
    box: 3D bounding box.
    keypoints: an optional list of labeled 3D keypoints.
  """
  object_type: 'ObjectType'
  box: label_pb2.Label.Box
  keypoints: Optional[Collection[keypoint_pb2.LaserKeypoint]] = None


@dataclasses.dataclass(frozen=True)
class CameraLabel:
  """A dataclass to store camera labels for keypoint visualization.

  Attributes:
    box: 2D bounding box.
    keypoints: an optional list of labeled 2D keypoints.
  """
  box: label_pb2.Label.Box
  keypoints: Optional[Collection[keypoint_pb2.CameraKeypoint]] = None


# A mapping between camera type and all its camera labels.
CameraLabelByType = Dict['CameraType', CameraLabel]


@dataclasses.dataclass(frozen=True)
class ObjectLabel:
  """A dataclass to store object labels for keypoint visualization.

  Attributes:
    laser: laser labels.
    camera: camera labels.
    object_type: type of the object.
  """
  laser: LaserLabel
  camera: CameraLabelByType = dataclasses.field(default_factory=dict)

  @property
  def object_type(self) -> ObjectType:
    return self.laser.object_type


def _index_laser_labels(frame: dataset_pb2.Frame) -> Dict[str, LaserLabel]:
  labels = {}
  for ll in frame.laser_labels:
    labels[ll.id] = LaserLabel(
        object_type=ll.type,
        box=ll.box,
        keypoints=getattr(ll, 'laser_keypoints', None))
  return labels


def _get_accosiated_id(label: label_pb2.Label) -> Optional[str]:
  if label.HasField('association'):
    return label.association.laser_object_id
  return None


def _index_camera_labels(
    frame: dataset_pb2.Frame) -> Dict[Optional[str], CameraLabelByType]:
  labels = {}
  for cl in frame.camera_labels:
    for l in cl.labels:
      laser_object_id = _get_accosiated_id(l)
      labels.setdefault(laser_object_id, {})[cl.name] = CameraLabel(
          box=l.box, keypoints=getattr(l, 'camera_keypoints', None))
  return labels


def group_object_labels(frame: dataset_pb2.Frame) -> Dict[str, ObjectLabel]:
  """Groups all object labels by laser object id.

  It uses 2d-to-3d assosiation labels to find which camera labels correspond to
  which laser labels.

  Args:
    frame: a Frame proto to extract labels for keypoint visualization.

  Returns:
    a dictionary where keys are laser object ids and values are labels.
  """
  object_labels = {}
  laser_labels = _index_laser_labels(frame)
  camera_labels = _index_camera_labels(frame)
  for object_id, ll in laser_labels.items():
    object_labels[object_id] = ObjectLabel(
        laser=ll, camera=camera_labels.get(object_id))
  return object_labels


def _vec2d_as_array(location_px: keypoint_pb2.Vec2d) -> np.ndarray:
  return np.array([location_px.x, location_px.y])


def _vec3d_as_array(location_m: keypoint_pb2.Vec3d) -> np.ndarray:
  return np.array([location_m.x, location_m.y, location_m.z])


def point_from_camera(k: keypoint_pb2.CameraKeypoint) -> Point:
  return Point(
      location=_vec2d_as_array(k.keypoint_2d.location_px),
      is_occluded=k.keypoint_2d.visibility.is_occluded)


def camera_keypoint_coordinates(
    keypoints: Collection[keypoint_pb2.CameraKeypoint]) -> PointByType:
  return {k.type: point_from_camera(k) for k in keypoints}


def point_from_laser(k: keypoint_pb2.LaserKeypoint) -> Point:
  return Point(
      location=_vec3d_as_array(k.keypoint_3d.location_m),
      is_occluded=k.keypoint_3d.visibility.is_occluded)


def laser_keypoint_coordinates(
    keypoints: Collection[keypoint_pb2.LaserKeypoint]) -> PointByType:
  return {k.type: point_from_laser(k) for k in keypoints}


def select_subset(values: tf.Tensor, src_order: Collection['KeypointType'],
                  dst_order: Collection['KeypointType']) -> tf.Tensor:
  """Returns a subset of values.

  Args:
    values: a tensor with shape [batch_size, num_points, ...],
    src_order: a list of types (e.g., nose, left_shoulder) in the input
      `values`.
    dst_order: a list of types (e.g., nose, left_shoulder) in the returned
      `values`.

  Returns:
    a tensor of the same type as the input `values`, with the shape
  """
  if values.shape.rank < 2:
    raise ValueError('Rank of input values has to be at least 2, got: '
                     f'{values.shape.rank}')
  num_points = values.shape[1]
  if num_points != len(src_order):
    raise ValueError(
        'Mismatch between number of input types and shape of values: '
        f'{num_points} != {len(src_order)}')
  if not set(dst_order).issubset(src_order):
    raise ValueError(f'Types in the requested order '
                     f'{dst_order} is not a subset of {src_order}')
  src_index = {kp_type: i for i, kp_type in enumerate(src_order)}
  src_values = tf.split(values, axis=1, num_or_size_splits=num_points)
  return tf.concat([src_values[src_index[kp_type]] for kp_type in dst_order],
                   axis=1)


class Visibility(enum.Enum):
  """Constants for visibility attribute represented as integer tensor values."""
  MISSING = 0
  OCCLUDED = 1
  VISIBLE = 2

  @classmethod
  def from_point(cls, point: Optional[Point]) -> 'Visibility':
    if point is None:
      return cls.MISSING
    elif point.is_occluded:
      return cls.OCCLUDED
    else:
      return cls.VISIBLE


@dataclasses.dataclass
class KeypointsTensors:
  """Tensors to represent 2D or 3D keypoints.

  Shape descriptions below use the following notation:
    B - number of samples in the batch (aka batch_size).
    N - number of keypoints.
    D - number of dimensions (e.g. 2 or 3).

  Attributes:
    location: a float tensor with shape [B, N, D] or [N, D].
    visibility: a int32 tensor with shape [B, N] or [N], with values: 0 -
      corresponding point is missing (not labeled or not detected), 1 - present,
      but marked as occluded, 2 - marked as visible or not occluded.
    mask: a float tensor with shape [B, N], with values: 0 - if corresponding
      point is missing (not labeled or not detected), 1 - otherwise.
    has_batch_dimension: True if location and visibility have batch dimensions.
    dims: number of dimensions (2 or 3).
  """
  location: tf.Tensor
  visibility: tf.Tensor

  def __post_init__(self):
    if self.location.shape.rank == 2:
      self.visibility.shape.assert_has_rank(1)
      self.visibility.shape.assert_is_compatible_with(self.location.shape[:1])
    elif self.location.shape.rank == 3:
      self.visibility.shape.assert_has_rank(2)
      self.visibility.shape.assert_is_compatible_with(self.location.shape[:2])
    else:
      raise ValueError('Rank of the location has to be 2 or 3, '
                       f'got {self.location.shape.rank}')
    if self.location.shape[-1] not in (2, 3):
      raise ValueError(
          f'Support only 2 or 3 dimensions: got {self.location.shape[-1]}')

  @property
  def mask(self) -> tf.Tensor:
    return tf.cast(self.visibility > 0, tf.float32)

  @property
  def has_batch_dimension(self) -> bool:
    return self.location.shape.rank == 3

  @property
  def dims(self) -> int:
    return self.location.shape[-1]

  def subset(self, src_order: Collection['KeypointType'],
             dst_order: Collection['KeypointType']) -> 'KeypointsTensors':
    """Returns a subset of keypoints."""
    return KeypointsTensors(
        location=select_subset(self.location, src_order, dst_order),
        visibility=select_subset(self.visibility, src_order, dst_order),
    )


def create_tensors_from_points(
    point_by_type: Mapping['KeypointType', Point],
    default_location: np.ndarray,
    order: Collection['KeypointType'],
    dtype: tf.DType = tf.float32) -> KeypointsTensors:
  """Creates tensors for points in all samples.

  Args:
    point_by_type: a mapping between type and corresponding point. Can be empty
      if there is no points for the object/frame.
    default_location: coordinates used for missing keypoints.
    order: an order of keypoint types in which output points will be stored.
    dtype: numeric type of the location tensor.

  Returns:
    Keypoint tensors without batch dimention with `num_points=len(order)`.
  """
  make_const = lambda v: tf.constant(v, dtype=dtype)
  default_location = make_const(default_location)
  points = [point_by_type.get(t, None) for t in order]
  location = [make_const(p.location) if p else default_location for p in points]
  return KeypointsTensors(
      location=tf.stack(location, axis=0),
      visibility=tf.constant([Visibility.from_point(p).value for p in points]))


def stack_keypoints(values: List[KeypointsTensors],
                    axis: int = 0) -> KeypointsTensors:
  """Dispatch method to support tf.stack for `BoundingBoxTensors`."""
  return KeypointsTensors(
      location=tf.stack([v.location for v in values], axis),
      visibility=tf.stack([v.visibility for v in values], axis))


def create_laser_keypoints_tensors(
    protos: Collection[keypoint_pb2.LaserKeypoint],
    default_location: np.ndarray,
    order: Collection['KeypointType'],
    dtype: tf.DType = tf.float32) -> KeypointsTensors:
  """Creates tensors for laser keypoints.

  Args:
    protos: all available keypoints for a sample.
    default_location: coordinates used for missing keypoints.
    order: an order of keypoint types in which output keypoints will be stored.
    dtype: numeric type of the location tensor.

  Returns:
    Keypoint tensors without batch dimension with `num_points=len(order)`.
  """
  return create_tensors_from_points(
      laser_keypoint_coordinates(protos), default_location, order, dtype=dtype)


def create_camera_keypoints_tensors(
    protos: Collection[keypoint_pb2.CameraKeypoint],
    default_location: np.ndarray,
    order: Collection['KeypointType'],
    dtype: tf.DType = tf.float32) -> KeypointsTensors:
  """Creates tensors for camera keypoints in all samples.

  Args:
    protos: all available keypoints for a sample.
    default_location: coordinates used for missing keypoints.
    order: an order of keypoint types in which output keypoints will be stored.
    dtype: numeric type of the location tensor.

  Returns:
    Keypoint tensors without batch dimension with `num_points=len(order)`.
  """
  return create_tensors_from_points(
      camera_keypoint_coordinates(protos), default_location, order, dtype=dtype)


@dataclasses.dataclass
class BoundingBoxTensors:
  """Tensors to represent 2D or 3D bounding boxes.

  Shape descriptions below use the following notation:
    B - number of samples in the batch (aka batch_size).
    D - number of dimensions (e.g. 2 or 3).

  Attributes:
    center: a float tensor with shape [B, D] or [D].
    size: a float tensor with shape [B, D] or [D].
    heading: The heading of the bounding box (in radians). It is a float tensor
      with shape [B]. Boxes are axis aligned in 2D, so it is None for camera
      bounding boxes. For 3D boxes the heading is the angle required to rotate
      +x to the surface normal of the box front face. It is normalized to [-pi,
      pi).
    scale: a float tensor with shape [B] or [], which means square root of the
      box's area in 2D and cubic root the volume in 3D.
    min_corner: corner with smallest coordinates, e.g. top left corner for a 2D
      box.
    max_corner: corner with largest coordinates, e.g. bottom right corner for a
      2D box.
    has_batch_dimension: True if location and visibility have batch dimensions.
    dims: number of dimensions (2 or 3).
  """
  center: tf.Tensor
  size: tf.Tensor
  heading: Optional[tf.Tensor] = None

  @property
  def scale(self) -> tf.Tensor:
    return tf.pow(tf.reduce_prod(self.size, axis=-1), 1.0 / self.size.shape[-1])

  @property
  def min_corner(self) -> tf.Tensor:
    return self.center - self.size / 2

  @property
  def max_corner(self) -> tf.Tensor:
    return self.center + self.size / 2

  def __post_init__(self):
    if self.center.shape.rank not in (1, 2):
      raise ValueError(
          f'Rank of the center has to be 1 or 2, got {self.center.shape.rank}')
    if self.center.shape[-1] not in (2, 3):
      raise ValueError(
          f'Support only 2 or 3 dimensions: got {self.center.shape[-1]}')
    self.center.shape.assert_is_compatible_with(self.size.shape)

  @property
  def has_batch_dimension(self) -> bool:
    return self.center.shape.rank == 2

  @property
  def dims(self) -> int:
    return self.center.shape[-1]


def create_camera_box_tensors(
    proto: label_pb2.Label.Box,
    dtype: tf.DType = tf.float32) -> BoundingBoxTensors:
  """Creates tensors for the bounding box for camera labels."""
  return BoundingBoxTensors(
      center=tf.constant([proto.center_x, proto.center_y], dtype=dtype),
      size=tf.constant([proto.length, proto.width], dtype=dtype))


def create_laser_box_tensors(
    proto: label_pb2.Label.Box,
    dtype: tf.DType = tf.float32) -> BoundingBoxTensors:
  """Creates tensors for the bounding box for laser labels."""
  return BoundingBoxTensors(
      center=tf.constant([proto.center_x, proto.center_y, proto.center_z],
                         dtype=dtype),
      size=tf.constant([proto.length, proto.width, proto.height], dtype=dtype),
      heading=tf.constant(proto.heading, dtype=dtype))


def stack_boxes(values: List[BoundingBoxTensors],
                axis: int = 0) -> BoundingBoxTensors:
  """Dispatch method to support tf.stack for `BoundingBoxTensors`."""
  return BoundingBoxTensors(
      center=tf.stack([v.center for v in values], axis),
      size=tf.stack([v.size for v in values], axis))
