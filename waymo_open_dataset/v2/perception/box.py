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
"""Box label components."""

import dataclasses
import enum
from typing import Optional

import pyarrow as pa

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2 import component
from waymo_open_dataset.v2.perception import base


_column = component.create_column


class BoxType(enum.Enum):
  """Object type of a box (a.k.a. class/category)."""
  TYPE_UNKNOWN = 0
  TYPE_VEHICLE = 1
  TYPE_PEDESTRIAN = 2
  TYPE_SIGN = 3
  TYPE_CYCLIST = 4


class DifficultyLevelType(enum.Enum):
  """The difficulty level types. The higher, the harder."""
  UNKNOWN = 0
  LEVEL_1 = 1
  LEVEL_2 = 2


@dataclasses.dataclass(frozen=True)
class DifficultyLevel:
  """Difficulty levels for detection and tracking problems."""
  detection: int = _column(arrow_type=pa.int8())
  tracking: int = _column(arrow_type=pa.int8())


@dataclasses.dataclass(frozen=True)
class LiDARBoxComponent(base.LaserLabelComponent):
  """LiDAR bounding box label component.

  Native 3D labels that correspond to the LiDAR sensor data. The 3D labels are
  defined w.r.t. the frame vehicle pose coordinate system.

  Attributes:
    box: 7-DOF 3D box (a.k.a. upright 3D box).
    type: Semantic object type of the box. See `BoxType`.
    speed: 3D linear velocity vector of the box.
    acceleration: 3D acceleration vector of the box.
    num_lidar_points_in_box: The total number of lidar points in this box.
    num_top_lidar_points_in_box: The total number of top lidar points in this
      box.
    difficulty_level: Difficulty levels for detection and tracking problems.
  """
  box: column_types.Box3d = _column()
  type: int = _column(arrow_type=pa.int8())
  num_lidar_points_in_box: int = _column(arrow_type=pa.int64())
  num_top_lidar_points_in_box: int = _column(arrow_type=pa.int64())
  speed: Optional[column_types.Vec3d] = _column(default=None)
  acceleration: Optional[column_types.Vec3d] = _column(default=None)
  difficulty_level: Optional[DifficultyLevel] = _column(default=None)


@dataclasses.dataclass(frozen=True)
class LiDARCameraSyncedBoxComponent(base.LaserLabelComponent):
  """A camera-synchronized box of a LiDAR box label.

  Attributes:
    most_visible_camera_name: In which camera this LiDAR box is mostly visible.
    camera_synced_box: A camera-synchronized box corresponding to the camera
      indicated by `most_visible_camera_name`. Currently, the boxes are shifted
      to the time when the most visible camera captures the center of the box,
      taking into account the rolling shutter of that camera.
      Specifically, given the object box living at the start of the Open Dataset
      frame (t_frame) with center position (c) and velocity (v), we aim to find
      the camera capture time (t_capture), when the camera indicated by
      `most_visible_camera_name` captures the center of the object. To this end,
      we solve the rolling shutter optimization considering both ego and object
      motion:
        t_capture = image_column_to_time(
            camera_projection(c + v * (t_capture - t_frame),
                              transform_vehicle(t_capture - t_ref),
                              cam_params)),
      where transform_vehicle(t_capture - t_frame) is the vehicle transform from
      a pose reference time t_ref to t_capture considering the ego motion, and
      cam_params is the camera extrinsic and intrinsic parameters.
      We then move the label box to t_capture by updating the center of the box
      as follows:
        c_camra_synced = c + v * (t_capture - t_frame),
      while keeping the box dimensions and heading direction.
      We use the camera_synced_box as the ground truth box for the 3D
      Camera-Only Detection Challenge. This makes the assumption that the users
      provide the detection at the same time as the most visible camera captures
      the object center.
  """
  most_visible_camera_name: int = _column(arrow_type=pa.int8())
  camera_synced_box: column_types.Box3d = _column()


@dataclasses.dataclass(frozen=True)
class ProjectedLaserLabelKey(base.LaserLabelKey, base.CameraKey):
  """Key for projected laser label component.

  This key combines base.LaserLabelKey and base.CameraKey so that it contains
  fields of `segment_context_name`, `frame_timestamp_micros`, `laser_object_id`,
  and `camera_name`.

  The projected laser label is uniquely identified by the combination of the
  `laser_object_id` and the `camera_name`. As a result, this special combination
  of keys is necessary.
  """


@dataclasses.dataclass(frozen=True)
class ProjectedLiDARBoxComponent(base.CameraLabelComponent):
  """Projected LiDAR bounding box label component.

  The native 3D LiDAR labels (laser_labels) projected to camera images. A
  projected box is the smallest image axis aligned rectangle that can cover all
  projected points from the 3d LiDAR box. The projected label is ignored if the
  projection is fully outside a camera image. The projected label is clamped to
  the camera image if it is partially outside.

  Attributes:
    key: A key component with `segment_context_name`, `frame_timestamp_micros`,
      `laser_name`, `laser_object_id`, `camera_name`.
    box: Projected image axis aligned box.
    type: Semantic object type of the box. See `BoxType`.
  """
  key: ProjectedLaserLabelKey = _column()
  box: column_types.BoxAxisAligned2d = _column()
  type: int = _column(arrow_type=pa.int8())


@dataclasses.dataclass(frozen=True)
class CameraBoxComponent(base.CameraLabelComponent):
  """Camera bounding box label component.

  Native 2D camera labels.

  Attributes:
    box: Image axis aligned bounding boxes.
    type: emantic object type of the box. See `BoxType`.
    difficulty_level: Difficulty levels for detection and tracking problems.
  """
  box: column_types.BoxAxisAligned2d = _column()
  type: int = _column(arrow_type=pa.int8())
  difficulty_level: Optional[DifficultyLevel] = _column(default=None)


@dataclasses.dataclass(frozen=True)
class CameraToLiDARBoxAssociationKey(base.CameraLabelKey):
  """Information to cross reference between labels for different modalities."""
  laser_object_id: str = _column(arrow_type=pa.string())


@dataclasses.dataclass(frozen=True)
class CameraToLiDARBoxAssociationComponent(base.CameraLabelComponent):
  """Component of info associating a camera box label to lidar box label.

  Note that currently only camera labels with class `TYPE_PEDESTRIAN` store
  information about associated lidar objects.
  """
  key: CameraToLiDARBoxAssociationKey
