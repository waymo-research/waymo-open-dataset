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
"""Tools to convert keypoint labels v2 to and from v1 Frame protos."""
from typing import Iterable, Optional

from waymo_open_dataset import label_pb2 as _v1_label_pb2
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2.perception import base as _v2_base
from waymo_open_dataset.v2.perception import keypoints as _v2_keypoints
from waymo_open_dataset.v2.perception.compat_v1 import interfaces


def _assert_camera_keypoint_has_required_fields(
    keypoint: keypoint_pb2.CameraKeypoint,
) -> None:
  interfaces.assert_has_fields(keypoint, ['type', 'keypoint_2d'])
  interfaces.assert_has_fields(
      keypoint.keypoint_2d, ['location_px', 'visibility']
  )


def _camera_hkp_from_label_proto(
    proto: _v1_label_pb2.Label,
) -> Optional[_v2_keypoints.CameraHumanKeypoints]:
  """Creates camera HKP column from the v1 label proto."""
  if not proto.HasField('camera_keypoints'):
    return None
  type_ = []
  is_occluded = []
  x, y = [], []
  for kp in proto.camera_keypoints.keypoint:
    _assert_camera_keypoint_has_required_fields(kp)
    type_.append(kp.type)
    is_occluded.append(kp.keypoint_2d.visibility.is_occluded)
    x.append(kp.keypoint_2d.location_px.x)
    y.append(kp.keypoint_2d.location_px.y)
  return _v2_keypoints.CameraHumanKeypoints(
      type=type_,
      keypoint_2d=_v2_keypoints.Keypoint2d(
          location_px=column_types.Vec2dList(x=x, y=y),
          visibility=_v2_keypoints.Visibility(is_occluded=is_occluded),
      ),
  )


def _assert_lidar_hkp_has_required_fields(
    keypoint: keypoint_pb2.LaserKeypoint,
) -> None:
  interfaces.assert_has_fields(keypoint, ['type', 'keypoint_3d'])
  interfaces.assert_has_fields(
      keypoint.keypoint_3d, ['location_m', 'visibility']
  )


def _lidar_hkp_from_label_proto(
    proto: _v1_label_pb2.Label,
) -> Optional[_v2_keypoints.LiDARHumanKeypoints]:
  """Creates LiDAR HKP column from the v1 label proto."""
  if not proto.HasField('laser_keypoints'):
    return None
  type_ = []
  is_occluded = []
  x, y, z = [], [], []
  for kp in proto.laser_keypoints.keypoint:
    _assert_lidar_hkp_has_required_fields(kp)
    type_.append(kp.type)
    is_occluded.append(kp.keypoint_3d.visibility.is_occluded)
    x.append(kp.keypoint_3d.location_m.x)
    y.append(kp.keypoint_3d.location_m.y)
    z.append(kp.keypoint_3d.location_m.z)
  return _v2_keypoints.LiDARHumanKeypoints(
      type=type_,
      keypoint_3d=_v2_keypoints.Keypoint3d(
          location_m=column_types.Vec3dList(x=x, y=y, z=z),
          visibility=_v2_keypoints.Visibility(is_occluded=is_occluded),
      ),
  )


class CameraHumanKeypointsFrameExtractor(
    interfaces.CameraLabelComponentExtractor
):
  """Extracts human keypoints from a single camera label."""

  def __call__(
      self, src: interfaces.CameraLabelComponentSrc
  ) -> Iterable[_v2_keypoints.CameraHumanKeypointsComponent]:
    """Extracts all camera keypoints a single camera label."""
    camera_keypoints = _camera_hkp_from_label_proto(src.label)
    if camera_keypoints:
      yield _v2_keypoints.CameraHumanKeypointsComponent(
          key=_v2_base.CameraLabelKey(
              segment_context_name=src.frame.context.name,
              frame_timestamp_micros=src.frame.timestamp_micros,
              camera_name=src.camera_labels.name,
              camera_object_id=src.label.id,
          ),
          camera_keypoints=camera_keypoints,
      )


class LiDARHumanKeypointsFrameExtractor(
    interfaces.LiDARLabelComponentExtractor
):
  """Extracts human keypoints from a single LiDAR label."""

  def __call__(
      self, src: interfaces.LiDARLabelComponentSrc
  ) -> Iterable[_v2_keypoints.LiDARHumanKeypointsComponent]:
    lidar_keypoints = _lidar_hkp_from_label_proto(src.lidar_label)
    if lidar_keypoints:
      yield _v2_keypoints.LiDARHumanKeypointsComponent(
          key=_v2_base.LaserLabelKey(
              segment_context_name=src.frame.context.name,
              frame_timestamp_micros=src.frame.timestamp_micros,
              laser_object_id=src.lidar_label.id,
          ),
          lidar_keypoints=lidar_keypoints,
      )
