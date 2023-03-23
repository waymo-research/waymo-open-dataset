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
"""Tools to convert box labels v2 to and from v1 Frame protos."""
from typing import Iterable, Optional

from waymo_open_dataset import label_pb2 as _v1_label_pb2
from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2.perception import base as _v2_base
from waymo_open_dataset.v2.perception import box as _v2_box
from waymo_open_dataset.v2.perception import camera_image as _v2_camera_image
from waymo_open_dataset.v2.perception.compat_v1 import interfaces


def _get_box3d(v1_box: _v1_label_pb2.Label.Box) -> column_types.Box3d:
  assert v1_box.HasField('center_x')
  assert v1_box.HasField('center_y')
  assert v1_box.HasField('center_z')
  assert v1_box.HasField('length')
  assert v1_box.HasField('width')
  assert v1_box.HasField('height')
  assert v1_box.HasField('heading')
  return column_types.Box3d(
      center=column_types.Vec3d(
          x=v1_box.center_x,
          y=v1_box.center_y,
          z=v1_box.center_z,
      ),
      size=column_types.Vec3d(
          x=v1_box.length,
          y=v1_box.width,
          z=v1_box.height,
      ),
      heading=v1_box.heading,
  )


def _get_box_axis_aligned_2d(
    v1_box: _v1_label_pb2.Label.Box,
) -> column_types.BoxAxisAligned2d:
  assert v1_box.HasField('center_x')
  assert v1_box.HasField('center_y')
  assert v1_box.HasField('length')
  assert v1_box.HasField('width')
  assert not v1_box.HasField('center_z')
  assert not v1_box.HasField('height')
  assert not v1_box.HasField('heading')
  return column_types.BoxAxisAligned2d(
      center=column_types.Vec2d(
          x=v1_box.center_x,
          y=v1_box.center_y,
      ),
      size=column_types.Vec2d(
          x=v1_box.length,
          y=v1_box.width,
      ),
  )


def _get_metadata_vector(
    label: _v1_label_pb2.Label, prefix: str
) -> Optional[column_types.Vec3d]:
  """Returns a speed or acceleration vector for the label proto."""
  if not label.HasField('metadata'):
    return None
  metadata = label.metadata
  assert metadata.HasField(f'{prefix}_x')
  assert metadata.HasField(f'{prefix}_y')
  if metadata.HasField(f'{prefix}_z'):
    z = getattr(metadata, f'{prefix}_z')
  else:
    # Average value of speed_z when it is populated is ~0 m.

    z = 0
  return column_types.Vec3d(
      x=getattr(metadata, f'{prefix}_x'),
      y=getattr(metadata, f'{prefix}_y'),
      z=z,
  )


def _get_difficulty(
    label: _v1_label_pb2.Label,
) -> Optional[_v2_box.DifficultyLevel]:
  if not label.HasField('detection_difficulty_level') and (
      not label.HasField('tracking_difficulty_level')
  ):
    return None
  return _v2_box.DifficultyLevel(
      detection=label.detection_difficulty_level,
      tracking=label.tracking_difficulty_level,
  )


class LiDARBoxComponentExtractor(interfaces.LiDARLabelComponentExtractor):
  """Extracts LiDAR box label component."""

  def __call__(
      self, src: interfaces.LiDARLabelComponentSrc
  ) -> Iterable[_v2_box.LiDARBoxComponent]:
    if not src.lidar_label.HasField('box'):
      return []
    key = _v2_base.LaserLabelKey(
        segment_context_name=src.frame.context.name,
        frame_timestamp_micros=src.frame.timestamp_micros,
        laser_object_id=src.lidar_label.id,
    )
    yield _v2_box.LiDARBoxComponent(
        key=key,
        box=_get_box3d(src.lidar_label.box),
        type=src.lidar_label.type,
        speed=_get_metadata_vector(src.lidar_label, 'speed'),
        acceleration=_get_metadata_vector(src.lidar_label, 'accel'),
        num_lidar_points_in_box=src.lidar_label.num_lidar_points_in_box,
        num_top_lidar_points_in_box=src.lidar_label.num_top_lidar_points_in_box,
        difficulty_level=_get_difficulty(src.lidar_label),
    )


class LiDARCameraSyncedBoxComponentExtractor(
    interfaces.LiDARLabelComponentExtractor
):
  """Extracts component of camera-synchronized box of LiDAR box label."""

  def __call__(
      self, src: interfaces.LiDARLabelComponentSrc
  ) -> Iterable[_v2_box.LiDARCameraSyncedBoxComponent]:
    if not src.lidar_label.most_visible_camera_name:
      return []
    if not src.lidar_label.HasField('camera_synced_box'):
      return []
    key = _v2_base.LaserLabelKey(
        segment_context_name=src.frame.context.name,
        frame_timestamp_micros=src.frame.timestamp_micros,
        laser_object_id=src.lidar_label.id,
    )
    yield _v2_box.LiDARCameraSyncedBoxComponent(
        key=key,
        most_visible_camera_name=_v2_camera_image.CameraName[
            src.lidar_label.most_visible_camera_name
        ].value,
        camera_synced_box=_get_box3d(src.lidar_label.camera_synced_box),
    )


def _parse_lidar_object_id(projected_lidar_id: str) -> tuple[str, int]:
  """Split projected lidar id into lidar object id and camera name."""
  # TODO(jingweij): verify that lidar_object_id is always 22-char long.
  lidar_object_id = projected_lidar_id[:22]
  camera_name = _v2_camera_image.CameraName[projected_lidar_id[23:]].value
  return lidar_object_id, camera_name


class ProjectedLiDARBoxComponentExtractor(
    interfaces.CameraLabelComponentExtractor
):
  """Extracts component of projected box of LiDAR box label."""

  def __call__(
      self,
      src: interfaces.CameraLabelComponentSrc,
  ) -> Iterable[_v2_box.ProjectedLiDARBoxComponent]:
    if not src.label.HasField('box'):
      return []
    lidar_object_id, camera_name = _parse_lidar_object_id(src.label.id)
    key = _v2_box.ProjectedLaserLabelKey(
        segment_context_name=src.frame.context.name,
        frame_timestamp_micros=src.frame.timestamp_micros,
        camera_name=camera_name,
        laser_object_id=lidar_object_id,
    )
    yield _v2_box.ProjectedLiDARBoxComponent(
        key=key,
        box=_get_box_axis_aligned_2d(src.label.box),
        type=src.label.type,
    )


class CameraBoxComponentExtractor(interfaces.CameraLabelComponentExtractor):
  """Extracts camera box label component."""

  def __call__(
      self,
      src: interfaces.CameraLabelComponentSrc,
  ) -> Iterable[_v2_box.CameraBoxComponent]:
    if not src.label.HasField('box'):
      return []
    key = _v2_base.CameraLabelKey(
        segment_context_name=src.frame.context.name,
        frame_timestamp_micros=src.frame.timestamp_micros,
        camera_name=src.camera_labels.name,
        camera_object_id=src.label.id,
    )
    yield _v2_box.CameraBoxComponent(
        key=key,
        box=_get_box_axis_aligned_2d(src.label.box),
        type=src.label.type,
        difficulty_level=_get_difficulty(src.label),
    )


class CameraToLiDARBoxAssociationComponentExtractor(
    interfaces.CameraLabelComponentExtractor
):
  """Extracts component of camera box to LiDAR box association."""

  def __call__(
      self, src: interfaces.CameraLabelComponentSrc
  ) -> Iterable[_v2_box.CameraToLiDARBoxAssociationComponent]:
    if src.label.HasField('association'):
      yield _v2_box.CameraToLiDARBoxAssociationComponent(
          key=_v2_box.CameraToLiDARBoxAssociationKey(
              segment_context_name=src.frame.context.name,
              frame_timestamp_micros=src.frame.timestamp_micros,
              camera_name=src.camera_labels.name,
              camera_object_id=src.label.id,
              laser_object_id=src.label.association.laser_object_id,
          )
      )
