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
"""A library with utils to work with keypoint labels."""

import dataclasses
import enum
from typing import Optional

import pyarrow as pa

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2 import component
from waymo_open_dataset.v2.perception import base


_column = component.create_column


@dataclasses.dataclass(frozen=True)
class Visibility:
  """Attributes related to the keypoint's visibility.

  Attributes:
    is_occluded: Is True, if the keypoint is occluded by any object, a body part
      or its location can be determined only with large uncertainty. Is false if
      the keypoint is clearly visible.
  """

  is_occluded: list[bool] = _column(arrow_type=pa.list_(pa.bool_()))


@dataclasses.dataclass(frozen=True)
class Keypoint2d:
  """Keypoints relative to a specific camera image.

  Attributes:
    location_px: Camera image coordinates (in pixels, x=col_id, y=row_id).
    visibility: Visibility attributes determined based on camera image only.
  """

  location_px: column_types.Vec2dList = _column()
  visibility: Visibility = _column()


@dataclasses.dataclass(frozen=True)
class Keypoint3d:
  """3D keypoints in world or vehicle coordinate frame.

  Coordinate frame should be defined in where this object is used and may change
  after applying a transformation.

  Attributes:
    location_m: column_types.Vec3dList
    visibility: Visibility attributes determined based on all available data
      (camera image and or lidar).
  """

  location_m: column_types.Vec3dList = _column()
  visibility: Visibility = _column()


class KeypointType(enum.Enum):
  """Type of a keypoint.

  All types of keypoints except (NOSE and HEAD_CENTER) are defined as the 3D
  location where corresponding bones meet - inside the body. We use
  person-centric coordinates in this task. For example, the person's right
  shoulder will be located on the left side of the image for frontal views and
  on the right side of the image for back views. Similarly for the other body
  joints.

  Enum values have to match corresponding enum proto values in
    waymo_open_dataset/protos/keypoint.proto

  Attributes:
    UNSPECIFIED: Default value, it means something was not properly initialized.
    NOSE: Tip of nose.
    LEFT_SHOULDER: center of the left shoulder (a location where head of humerus
      meet glenoid cavity).
    LEFT_ELBOW: center of the left elbow (a location where trochlear notch meet
      the trochlea).
    LEFT_WRIST: center of the left wrist (a location where radius and ulna meet
      scaphoid and lunate).
    LEFT_HIP: center of the left hip joint (head of femur).
    LEFT_KNEE: center of the left knee (intercondylar eminence).
    LEFT_ANKLE: center of the left ankle joint (center of talus).
    RIGHT_SHOULDER: Center of the right shoulder joint (location where head of
      humerus meet the glenoid cavity).
    RIGHT_ELBOW: center of the right elbow (a location where trochlear notch
      meet the trochlea).
    RIGHT_WRIST: center of the right wrist (a location where radius and ulna
      meet scaphoid and lunate).
    RIGHT_HIP: center of the right hip joint (head of femur).
    RIGHT_KNEE: center of the right knee (intercondylar eminence).
    RIGHT_ANKLE: center of the right ankle joint (center of talus).
    FOREHEAD: Center of the forehead area.
    KEYPOINT_TYPE_HEAD_CENTER: A point in the center of head - a point in the
      middle between two ears. The nose and head center together create an
      imaginary line in the direction that the person is looking (i.e. head
      orientation).
  """

  UNSPECIFIED = 0
  NOSE = 1
  LEFT_SHOULDER = 5
  LEFT_ELBOW = 6
  LEFT_WRIST = 7
  LEFT_HIP = 8
  LEFT_KNEE = 9
  LEFT_ANKLE = 10
  RIGHT_SHOULDER = 13
  RIGHT_ELBOW = 14
  RIGHT_WRIST = 15
  RIGHT_HIP = 16
  RIGHT_KNEE = 17
  RIGHT_ANKLE = 18
  FOREHEAD = 19
  KEYPOINT_TYPE_HEAD_CENTER = 20


@dataclasses.dataclass(frozen=True)
class CameraHumanKeypoints:
  """Keypoints for a specific camera image.

  Camera keypoints may have 2d keypoints or 3d keypoints or both.

  Attributes:
    type: integer values corresponding to KeypointType enum values. Stored as
      ints to enable serialization to/from a columnar file format which doesn't
      support enums. Different objects may different keypoints annotated.
    keypoint_2d: camera coordinates.
    keypoint_3d: 3D keypoint in camera coordinate frame (originated from lidar
      keypoints by changing coordinate frame if lidar->camera object association
      is known).
  """

  type: list[int] = _column(arrow_type=pa.list_(pa.int8()))
  keypoint_2d: Optional[Keypoint2d] = _column(default=None)
  keypoint_3d: Optional[Keypoint3d] = _column(default=None)


@dataclasses.dataclass(frozen=True)
class CameraHumanKeypointsComponent(base.CameraLabelComponent):
  """A dataset component for human keypoints for a camera label object."""

  camera_keypoints: CameraHumanKeypoints = _column(is_repeated=True)


@dataclasses.dataclass(frozen=True)
class LiDARHumanKeypoints:
  """Keypoints for a laser labeled object.

  Attributes:
    type: integer values corresponding to KeypointType enum values. Stored as
      ints to enable serialization to/from a columnar file format which doesn't
      support enums.
    keypoint_3d: 3D keypoint in vehicle coordinate frame.
  """

  type: list[int] = _column(arrow_type=pa.list_(pa.int8()))
  keypoint_3d: Keypoint3d = _column()


@dataclasses.dataclass(frozen=True)
class LiDARHumanKeypointsComponent(base.LaserLabelComponent):
  """A dataset component for human keypoints for a lidar label object."""

  lidar_keypoints: LiDARHumanKeypoints = _column(is_repeated=True)
