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
"""Context components."""

import dataclasses
import enum
from typing import Optional

import pyarrow as pa

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2 import component
from waymo_open_dataset.v2.perception import base


_column = component.create_column


@dataclasses.dataclass(frozen=True)
class Intrinsic:
  """Intrinsic parameters of a pinhole camera model.

  Note that this intrinsic corresponds to the images after scaling.

  Lens distortion:
    Radial distortion coefficients: k1, k2, k3.
    Tangential distortion coefficients: p1, p2.
  k_{1, 2, 3}, p_{1, 2} follows the same definition as OpenCV.
  See https://en.wikipedia.org/wiki/Distortion_(optics) and
  https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
  for definitions.
  """
  f_u: float = _column(arrow_type=pa.float64())
  f_v: float = _column(arrow_type=pa.float64())
  c_u: float = _column(arrow_type=pa.float64())
  c_v: float = _column(arrow_type=pa.float64())
  k1: float = _column(arrow_type=pa.float64())
  k2: float = _column(arrow_type=pa.float64())
  p1: float = _column(arrow_type=pa.float64())
  p2: float = _column(arrow_type=pa.float64())
  k3: float = _column(arrow_type=pa.float64())


class RollingShutterReadOutDirection(enum.Enum):
  UNKNOWN = 0
  TOP_TO_BOTTOM = 1
  LEFT_TO_RIGHT = 2
  BOTTOM_TO_TOP = 3
  RIGHT_TO_LEFT = 4
  GLOBAL_SHUTTER = 5


@dataclasses.dataclass(frozen=True)
class CameraCalibrationComponent(base.SegmentCameraComponent):
  """Component of parameters for calibration of a camera sensor.

  Attributes:
    intrinsic: Intrinsic parameters of a pinhole camera model.
    extrinsic: Trasformation from camera frame to vehicle frame.
    width: Camera image width.
    height: Camera image height.
    rolling_shutter_direction: The direction of readout of this camera sensor.
      Integer value of RollingShutterReadOutDirection.
  """
  intrinsic: Intrinsic = _column()
  extrinsic: column_types.Transform = _column()
  width: int = _column(arrow_type=pa.int32())
  height: int = _column(arrow_type=pa.int32())
  rolling_shutter_direction: int = _column(arrow_type=pa.int8())


@dataclasses.dataclass(frozen=True)
class BeamInclination:
  """The beam inclination/pitch for mapping between LiDAR point and range image.

  Attributes:
    min: Beam inclination min (in radians) used for determine the mapping.
    max: Beam inclination max (in radians) used for determine the mapping.
    values: If provided, the beam pitch (in radians) is non-uniform. When
      constructing a range image, this mapping is used to map from beam pitch to
      range image row. If this is not provided, we assume a uniform distribution
      determined by the `min` and `max`.
  """
  min: float = _column(arrow_type=pa.float64())
  max: float = _column(arrow_type=pa.float64())
  values: Optional[list[float]] = _column(
      arrow_type=pa.list_(pa.float64()), default=None
  )


@dataclasses.dataclass(frozen=True)
class LiDARCalibrationComponent(base.SegmentLaserComponent):
  """Component of parameters for calibration of a LiDAR sensor.

  Attributes:
    extrinsic: Transformation from LiDAR frame to vehicle frame.
    beam_inclination: The LiDAR beam inclination/pitch in radians for mapping
      each row of a range image to a circle in the 3D coordinate system.
  """
  extrinsic: column_types.Transform = _column()
  beam_inclination: BeamInclination = _column()


@dataclasses.dataclass(frozen=True)
class ObjectCounts:
  """Counts of occurring object of each type.

  Attributes:
    types: A list of integer values of box.BoxType enum.
    counts: A list of counts corresponding to the `types`.
  """
  types: list[int] = _column(arrow_type=pa.list_(pa.int8()))
  counts: list[int] = _column(arrow_type=pa.list_(pa.int32()))


@dataclasses.dataclass(frozen=True)
class StatsComponent(base.FrameComponent):
  """A component of miscellaneous statistics.

  Attributes:
    time_of_day: Day, Dawn/Dusk or Night, determined from sun elevation.
    location: Human readable location (e.g. CHD, SF) of the run segment.
    weather: Currently either Sunny or Rain.
    lidar_object_counts: Counts of object captured by LiDAR sensors.
    camera_object_counts: Counts of object captured by camera sensors.
  """
  time_of_day: str = _column(arrow_type=pa.string())
  location: str = _column(arrow_type=pa.string())
  weather: str = _column(arrow_type=pa.string())
  lidar_object_counts: Optional[ObjectCounts] = _column(default=None)
  camera_object_counts: Optional[ObjectCounts] = _column(default=None)
