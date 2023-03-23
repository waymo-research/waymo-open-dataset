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
"""Tools to convert segment-level context v2 to and from v1 Frame protos."""
from typing import Iterable, Optional

from waymo_open_dataset import dataset_pb2 as _v1_dataset_pb2
from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2.perception import base as _v2_base
from waymo_open_dataset.v2.perception import context as _v2_context
from waymo_open_dataset.v2.perception.compat_v1 import interfaces


class CameraCalibrationComponentExtractor(
    interfaces.CameraCalibrationComponentExtractor
):
  """Extracts camera calibration component."""

  def __call__(
      self, src: interfaces.CameraCalibrationComponentSrc
  ) -> Iterable[_v2_context.CameraCalibrationComponent]:
    intrinsic_v1 = src.camera_calibration.intrinsic
    intrinsic = _v2_context.Intrinsic(
        f_u=intrinsic_v1[0],
        f_v=intrinsic_v1[1],
        c_u=intrinsic_v1[2],
        c_v=intrinsic_v1[3],
        k1=intrinsic_v1[4],
        k2=intrinsic_v1[5],
        p1=intrinsic_v1[6],
        p2=intrinsic_v1[7],
        k3=intrinsic_v1[8],
    )
    extrinsic = column_types.Transform(
        transform=list(src.camera_calibration.extrinsic.transform),
    )
    yield _v2_context.CameraCalibrationComponent(
        key=_v2_base.SegmentCameraKey(
            segment_context_name=src.frame.context.name,
            camera_name=src.camera_calibration.name,
        ),
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        width=src.camera_calibration.width,
        height=src.camera_calibration.height,
        rolling_shutter_direction=(
            src.camera_calibration.rolling_shutter_direction
        ),
    )


class LiDARCalibrationComponentExtractor(
    interfaces.LiDARCalibrationComponentExtractor
):
  """Extracts LiDAR calibration component."""

  def __call__(
      self, src: interfaces.LiDARCalibrationComponentSrc
  ) -> Iterable[_v2_context.LiDARCalibrationComponent]:
    extrinsic = column_types.Transform(
        transform=list(src.lidar_calibration.extrinsic.transform),
    )
    if src.lidar_calibration.beam_inclinations:
      beam_inclination = _v2_context.BeamInclination(
          min=src.lidar_calibration.beam_inclination_min,
          max=src.lidar_calibration.beam_inclination_max,
          values=list(src.lidar_calibration.beam_inclinations),
      )
    else:
      beam_inclination = _v2_context.BeamInclination(
          min=src.lidar_calibration.beam_inclination_min,
          max=src.lidar_calibration.beam_inclination_max,
      )
    yield _v2_context.LiDARCalibrationComponent(
        key=_v2_base.SegmentLaserKey(
            segment_context_name=src.frame.context.name,
            laser_name=src.lidar_calibration.name,
        ),
        extrinsic=extrinsic,
        beam_inclination=beam_inclination,
    )


def _parse_object_counts(
    object_counts: list[_v1_dataset_pb2.Context.Stats.ObjectCount],
) -> Optional[_v2_context.ObjectCounts]:
  if not object_counts:
    return None
  types = [object_count.type for object_count in object_counts]
  counts = [object_count.count for object_count in object_counts]
  return _v2_context.ObjectCounts(
      types=types, counts=counts,
  )


class StatsComponentExtractor(
    interfaces.FrameComponentExtractor
):
  """Extracts stats component."""

  def __call__(
      self, src: interfaces.FrameComponentSrc
  ) -> Iterable[_v2_context.StatsComponent]:
    yield _v2_context.StatsComponent(
        key=_v2_base.FrameKey(
            segment_context_name=src.frame.context.name,
            frame_timestamp_micros=src.frame.timestamp_micros,
        ),
        time_of_day=src.frame.context.stats.time_of_day,
        location=src.frame.context.stats.location,
        weather=src.frame.context.stats.weather,
        lidar_object_counts=_parse_object_counts(
            list(src.frame.context.stats.laser_object_counts)
        ),
        camera_object_counts=_parse_object_counts(
            list(src.frame.context.stats.camera_object_counts)
        ),
    )
