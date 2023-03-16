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
"""Tools to convert LiDAR data v2 to and from v1 Frame protos."""
from typing import Iterable

from waymo_open_dataset.v2.perception import lidar as _v2_lidar
from waymo_open_dataset.v2.perception.compat_v1 import interfaces
from waymo_open_dataset.v2.perception.utils import lidar_utils


class LiDARComponentExtractor(interfaces.LiDARComponentExtractor):
  """Extracts LiDAR data component captured by a single LiDAR sensor."""

  def __call__(
      self, src: interfaces.LiDARComponentSrc
  ) -> Iterable[_v2_lidar.LiDARComponent]:
    range_image_return1 = lidar_utils.parse_range_image(
        src.laser.ri_return1.range_image_compressed,
        _v2_lidar.RangeImage,
    )
    range_image_return2 = lidar_utils.parse_range_image(
        src.laser.ri_return2.range_image_compressed,
        _v2_lidar.RangeImage,
    )
    # Skip if both returns are None
    if not (range_image_return1 is None and range_image_return2 is None):
      yield _v2_lidar.LiDARComponent(
          key=lidar_utils.get_laser_key(src),
          range_image_return1=range_image_return1,
          range_image_return2=range_image_return2,
      )


class LiDARCameraProjectionComponentExtractor(
    interfaces.LiDARComponentExtractor
):
  """Extracts LiDARCameraProjectionComponent."""

  def __call__(
      self, src: interfaces.LiDARComponentSrc
  ) -> Iterable[_v2_lidar.LiDARCameraProjectionComponent]:
    range_image_return1 = lidar_utils.parse_range_image(
        src.laser.ri_return1.camera_projection_compressed,
        _v2_lidar.CameraProjectionRangeImage,
    )
    range_image_return2 = lidar_utils.parse_range_image(
        src.laser.ri_return2.camera_projection_compressed,
        _v2_lidar.CameraProjectionRangeImage,
    )
    # Skip if both returns are None
    if not (range_image_return1 is None and range_image_return2 is None):
      yield _v2_lidar.LiDARCameraProjectionComponent(
          key=lidar_utils.get_laser_key(src),
          range_image_return1=range_image_return1,
          range_image_return2=range_image_return2,
      )


class LiDARPoseComponentExtractor(interfaces.LiDARComponentExtractor):
  """"Extracts LiDARPoseComponent."""

  def __call__(
      self, src: interfaces.LiDARComponentSrc
  ) -> Iterable[_v2_lidar.LiDARPoseComponent]:
    range_image_return1 = lidar_utils.parse_range_image(
        src.laser.ri_return1.range_image_pose_compressed,
        _v2_lidar.PoseRangeImage,
    )
    if range_image_return1 is not None:
      yield _v2_lidar.LiDARPoseComponent(
          key=lidar_utils.get_laser_key(src),
          range_image_return1=range_image_return1,
      )
