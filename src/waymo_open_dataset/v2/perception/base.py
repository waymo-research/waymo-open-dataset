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
"""Interfaces for base components.

Base component classes are supposed to have only keys defined - it is required
to know how to join different components into a single table. Child classes are
supposed to define all fields they need.
"""
import dataclasses

import pyarrow as pa

from waymo_open_dataset.v2 import component

ARROW_STRING_DICTIONARY = pa.dictionary(
    index_type=pa.int32(), value_type=pa.string(), ordered=False
)


@dataclasses.dataclass(frozen=True)
class SegmentKey(component.Key):
  """Base class for SegmentComponent keys."""

  # Automatically created schema for partitioning using hive flavor uses
  # a dictionary for the key field.
  # segment_context_name: str = component.create_column(
  # arrow_type=ARROW_STRING_DICTIONARY)
  segment_context_name: str = component.create_column(arrow_type=pa.string())


@dataclasses.dataclass(frozen=True)
class SegmentComponent(component.Component):
  key: SegmentKey


@dataclasses.dataclass(frozen=True)
class FrameKey(SegmentKey):
  frame_timestamp_micros: int = component.create_column(arrow_type=pa.int64())


@dataclasses.dataclass(frozen=True)
class FrameComponent(component.Component):
  key: FrameKey


@dataclasses.dataclass(frozen=True)
class SegmentLaserKey(SegmentKey):
  """Base class for SegmentLaserComponent keys.

  Attributes:
    laser_name: integer value corresponding to lidar.LaserName enum value.
  """
  laser_name: int = component.create_column(arrow_type=pa.int8())


@dataclasses.dataclass(frozen=True)
class SegmentLaserComponent(SegmentComponent):
  key: SegmentLaserKey


@dataclasses.dataclass(frozen=True)
class LaserKey(FrameKey):
  """Base class for LaserComponent keys.

  Attributes:
    laser_name: integer value corresponding to lidar.LaserName enum value.
  """
  laser_name: int = component.create_column(arrow_type=pa.int8())


@dataclasses.dataclass(frozen=True)
class LaserComponent(component.Component):
  key: LaserKey


@dataclasses.dataclass(frozen=True)
class SegmentCameraKey(SegmentKey):
  """Base class for SegmentCameraComponent keys.

  Attributes:
    camera_name: integer value corresponding to camera_image.CameraName enum
      value.
  """
  camera_name: int = component.create_column(arrow_type=pa.int8())


@dataclasses.dataclass(frozen=True)
class SegmentCameraComponent(SegmentComponent):
  key: SegmentCameraKey


@dataclasses.dataclass(frozen=True)
class CameraKey(FrameKey):
  """Base class for CameraComponent keys.

  Attributes:
    camera_name: integer value corresponding to camera_image.CameraName enum
      value.
  """
  camera_name: int = component.create_column(arrow_type=pa.int8())


@dataclasses.dataclass(frozen=True)
class CameraComponent(component.Component):
  key: CameraKey


@dataclasses.dataclass(frozen=True)
class LaserLabelKey(FrameKey):
  # Note that LaserLabelKey inherits from FrameKey and does not have
  # `laser_name` attribute.
  # This is because that laser labels apply to the entire 3D scene, rather than
  # limited to any specifc LiDAR.
  laser_object_id: str = component.create_column(arrow_type=pa.string())


@dataclasses.dataclass(frozen=True)
class LaserLabelComponent(component.Component):
  key: LaserLabelKey


@dataclasses.dataclass(frozen=True)
class CameraLabelKey(CameraKey):
  camera_object_id: str = component.create_column(arrow_type=pa.string())


@dataclasses.dataclass(frozen=True)
class CameraLabelComponent(component.Component):
  key: CameraLabelKey


@dataclasses.dataclass(frozen=True)
class ObjectAssetKey(LaserLabelKey):
  camera_name: int = component.create_column(arrow_type=pa.int8())
