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
# =============================================================================
"""Tools related to working with human keypoints."""
import abc
import dataclasses
from typing import Collection, Mapping, Optional, Tuple

import immutabledict
from matplotlib import collections as mc
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.utils import keypoint_data as _data

KeypointType = keypoint_pb2.KeypointType
ColorByType = Mapping['KeypointType', str]

DEFAULT_COLOR = '#424242'
OCCLUDED_COLOR = '#000000'
OCCLUDED_BORDER_WIDTH = 3


@dataclasses.dataclass(frozen=True)
class Line:
  """Line in 2D or 3D space for visualization of keypoints.

  It is a primitive for visualization purposes.

  Attributes:
    start: Start point of the line, an array with 2 or 3 elements.
    end:  End point of the line, an array with 2 or 3 elements.
    color: a hex string representing an RGB color.
    width: thickness of the line in units specifix to a rendering function.
  """
  start: np.ndarray
  end: np.ndarray
  color: str
  width: float


@dataclasses.dataclass(frozen=True)
class Dot:
  """A small circle/sphere in 2D or 3D space for visualization of keypoints.

  It is a primitive for visualization purposes.

  Attributes:
    location: Point in space, an array with 2 or 3 elements.
    color: a hex string representing an RGB color.
    size: diameter of the dot in units specifix to a rendering function.
    name: name of the keypoint.
    border_color: optional color of the border, default is None.
    actual_border_color: same as the `border_color` if it was set and same as
      `color` otherwise.
  """
  location: np.ndarray
  color: str
  size: float
  name: str
  border_color: Optional[str] = None

  @property
  def actual_border_color(self) -> str:
    if self.border_color:
      return self.border_color
    return self.color


@dataclasses.dataclass(frozen=True)
class Wireframe:
  """Data required to render visual representation of keypoints.

  Attributes:
    lines: set of line segments between keypoints.
    dots: set of points in keypoints locations.
  """
  lines: Collection[Line]
  dots: Collection[Dot]


class Edge(abc.ABC):
  """Base class for all wireframe edges."""

  @abc.abstractmethod
  def create_lines(self, point_by_type: _data.PointByType,
                   colors: ColorByType) -> Collection[Line]:
    """Creates all lines to visualize an edge.

    Args:
      point_by_type: a mapping between keypoint type and its coordinates.
      colors: a mpping between keypoitn type and its color.

    Returns:
      a list of line representing the edge in the wireframe.
    """


@dataclasses.dataclass(frozen=True)
class SolidLineEdge(Edge):
  """Represents an edge between two keypoints a single line segment.

  Attributes:
    start: type of the start keypoint.
    end: type of the end keypoint.
    width: thickness of the line in units specifix to a rendering function.
  """
  start: 'KeypointType'
  end: 'KeypointType'
  width: float

  def create_lines(self, point_by_type: _data.PointByType,
                   colors: Mapping['KeypointType', str]) -> Collection[Line]:
    """See base class."""
    if self.start not in point_by_type or self.end not in point_by_type:
      return []
    color = colors.get(self.start, DEFAULT_COLOR)
    return [
        Line(point_by_type[self.start].location,
             point_by_type[self.end].location, color, self.width)
    ]


def _bicolor_lines(start: np.ndarray, start_color: str, end: np.ndarray,
                   end_color: str, width: float) -> Collection[Line]:
  middle = (start + end) / 2
  start_half = Line(start, middle, start_color, width)
  end_half = Line(middle, end, end_color, width)
  return [start_half, end_half]


@dataclasses.dataclass(frozen=True)
class BicoloredEdge(SolidLineEdge):
  """Edge with two line segments colored according to keypoint type."""

  def create_lines(self, point_by_type: _data.PointByType,
                   colors: Mapping['KeypointType', str]) -> Collection[Line]:
    """See base class."""
    if self.start not in point_by_type or self.end not in point_by_type:
      return []
    start = np.array(point_by_type[self.start].location)
    start_color = colors.get(self.start, DEFAULT_COLOR)
    end = np.array(point_by_type[self.end].location)
    end_color = colors.get(self.end, DEFAULT_COLOR)
    return _bicolor_lines(start, start_color, end, end_color, self.width)


def _combine_colors(colors: Collection[str]) -> str:
  if len(colors) == 1:
    return next(iter(colors))
  else:
    return DEFAULT_COLOR


@dataclasses.dataclass(frozen=True)
class MultipointEdge(Edge):
  """An edge with start/end points computed by averaging input coordinates."""
  start_avg: Collection['KeypointType']
  end_avg: Collection['KeypointType']
  width: float

  def create_lines(self, point_by_type: _data.PointByType,
                   colors: Mapping['KeypointType', str]) -> Collection[Line]:
    """See base class."""
    has_start = set(self.start_avg).issubset(point_by_type.keys())
    has_end = set(self.end_avg).issubset(point_by_type.keys())
    if not has_start or not has_end:
      return []
    start = np.mean([point_by_type[s].location for s in self.start_avg], axis=0)
    start_color = _combine_colors([colors[s] for s in self.start_avg])
    end = np.mean([point_by_type[e].location for e in self.end_avg], axis=0)
    end_color = _combine_colors([colors[s] for s in self.end_avg])
    return _bicolor_lines(start, start_color, end, end_color, self.width)


_COMMON_EDGES = (BicoloredEdge(KeypointType.KEYPOINT_TYPE_LEFT_ANKLE,
                               KeypointType.KEYPOINT_TYPE_LEFT_KNEE, 2),
                 BicoloredEdge(KeypointType.KEYPOINT_TYPE_LEFT_KNEE,
                               KeypointType.KEYPOINT_TYPE_LEFT_HIP, 2),
                 BicoloredEdge(KeypointType.KEYPOINT_TYPE_LEFT_HIP,
                               KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER, 2),
                 BicoloredEdge(KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER,
                               KeypointType.KEYPOINT_TYPE_LEFT_ELBOW, 2),
                 BicoloredEdge(KeypointType.KEYPOINT_TYPE_LEFT_ELBOW,
                               KeypointType.KEYPOINT_TYPE_LEFT_WRIST, 2),
                 BicoloredEdge(KeypointType.KEYPOINT_TYPE_RIGHT_ANKLE,
                               KeypointType.KEYPOINT_TYPE_RIGHT_KNEE, 2),
                 BicoloredEdge(KeypointType.KEYPOINT_TYPE_RIGHT_KNEE,
                               KeypointType.KEYPOINT_TYPE_RIGHT_HIP, 2),
                 BicoloredEdge(KeypointType.KEYPOINT_TYPE_RIGHT_HIP,
                               KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER, 2),
                 BicoloredEdge(KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER,
                               KeypointType.KEYPOINT_TYPE_RIGHT_ELBOW, 2),
                 BicoloredEdge(KeypointType.KEYPOINT_TYPE_RIGHT_ELBOW,
                               KeypointType.KEYPOINT_TYPE_RIGHT_WRIST, 2),
                 BicoloredEdge(KeypointType.KEYPOINT_TYPE_LEFT_HIP,
                               KeypointType.KEYPOINT_TYPE_RIGHT_HIP, 2),
                 BicoloredEdge(KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER,
                               KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER, 2))
_DEFAULT_CAMERA_EDGES = _COMMON_EDGES + (
    BicoloredEdge(KeypointType.KEYPOINT_TYPE_NOSE,
                  KeypointType.KEYPOINT_TYPE_FOREHEAD, 1),
    MultipointEdge([KeypointType.KEYPOINT_TYPE_FOREHEAD], [
        KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER,
        KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER
    ], 1))
_DEFAULT_LASER_EDGES = _COMMON_EDGES + (
    BicoloredEdge(KeypointType.KEYPOINT_TYPE_NOSE,
                  KeypointType.KEYPOINT_TYPE_HEAD_CENTER, 1),
    MultipointEdge([KeypointType.KEYPOINT_TYPE_HEAD_CENTER], [
        KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER,
        KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER
    ], 1))

_DEFAULT_POINT_COLORS = immutabledict.immutabledict({
    KeypointType.KEYPOINT_TYPE_NOSE: '#00FF00',
    KeypointType.KEYPOINT_TYPE_FOREHEAD: '#00FFFF',
    KeypointType.KEYPOINT_TYPE_HEAD_CENTER: '#00FFFF',
    KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER: '#FFA6FE',
    KeypointType.KEYPOINT_TYPE_LEFT_ELBOW: '#FFE502',
    KeypointType.KEYPOINT_TYPE_LEFT_WRIST: '#006401',
    KeypointType.KEYPOINT_TYPE_LEFT_HIP: '#010067',
    KeypointType.KEYPOINT_TYPE_LEFT_KNEE: '#95003A',
    KeypointType.KEYPOINT_TYPE_LEFT_ANKLE: '#007DB5',
    KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER: '#774D00',
    KeypointType.KEYPOINT_TYPE_RIGHT_ELBOW: '#90FB92',
    KeypointType.KEYPOINT_TYPE_RIGHT_WRIST: '#0076FF',
    KeypointType.KEYPOINT_TYPE_RIGHT_HIP: '#D5FF00',
    KeypointType.KEYPOINT_TYPE_RIGHT_KNEE: '#A75740',
    KeypointType.KEYPOINT_TYPE_RIGHT_ANKLE: '#6A826C'
})

# Keypoint sigmas from https://cocodataset.org/#keypoints-eval
_OKS_SCALES = immutabledict.immutabledict({
    KeypointType.KEYPOINT_TYPE_NOSE: 0.052,
    KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER: 0.158,
    KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER: 0.158,
    KeypointType.KEYPOINT_TYPE_LEFT_ELBOW: 0.144,
    KeypointType.KEYPOINT_TYPE_RIGHT_ELBOW: 0.144,
    KeypointType.KEYPOINT_TYPE_LEFT_WRIST: 0.124,
    KeypointType.KEYPOINT_TYPE_RIGHT_WRIST: 0.124,
    KeypointType.KEYPOINT_TYPE_LEFT_HIP: 0.214,
    KeypointType.KEYPOINT_TYPE_RIGHT_HIP: 0.214,
    KeypointType.KEYPOINT_TYPE_LEFT_KNEE: 0.174,
    KeypointType.KEYPOINT_TYPE_RIGHT_KNEE: 0.174,
    KeypointType.KEYPOINT_TYPE_LEFT_ANKLE: 0.178,
    KeypointType.KEYPOINT_TYPE_RIGHT_ANKLE: 0.178,
    KeypointType.KEYPOINT_TYPE_FOREHEAD: 0.158,
    KeypointType.KEYPOINT_TYPE_HEAD_CENTER: 0.158
})


def _default_point_sizes():
  max_scale = max(s for s in _OKS_SCALES.values())
  return {n: s / max_scale for n, s in _OKS_SCALES.items()}


@dataclasses.dataclass(frozen=True)
class WireframeConfig:
  """Settings to build a wireframe out of a set of keypoints.

  Attributes:
    edges: types of keypoints to connect with line segments.
    point_colors: colors of keypoint types for corresponding dots and line
      segments.
    point_sizes: nominal sizes of dots.
    dot_size_factor: a factor used to compute actual sizes of dots using
      `point_sizes`.
    line_width_factor: a factor used to compute actual width of lines using
      configured edge widths.
  """
  edges: Collection[BicoloredEdge]
  point_colors: Mapping['KeypointType', str]
  point_sizes: Mapping['KeypointType', float]
  dot_size_factor: float = 1.0
  line_width_factor: float = 1.0


DEFAULT_CAMERA_WIREFRAME_CONFIG = WireframeConfig(
    edges=_DEFAULT_CAMERA_EDGES,
    point_colors=_DEFAULT_POINT_COLORS,
    point_sizes=_default_point_sizes(),
    dot_size_factor=6,
    line_width_factor=1)
DEFAULT_LASER_WIREFRAME_CONFIG = WireframeConfig(
    edges=_DEFAULT_LASER_EDGES,
    point_colors=_DEFAULT_POINT_COLORS,
    point_sizes=_default_point_sizes(),
    dot_size_factor=15,
    line_width_factor=4)


def _removeprefix(x, y):
  if x.startswith(y):
    return x[len(y):]
  return x


def point_name(point_type: 'KeypointType') -> str:
  """Returns a short name of the keypoint type."""
  name = KeypointType.Name(point_type)
  return _removeprefix(name, 'KEYPOINT_TYPE_')


def _build_wireframe(point_by_type: _data.PointByType,
                     config: WireframeConfig) -> Wireframe:
  """Creates a wireframe for a collection of keypoint coordinates."""
  lines = []
  for e in config.edges:
    e = dataclasses.replace(e, width=e.width * config.line_width_factor)
    lines.extend(e.create_lines(point_by_type, config.point_colors))
  dots = []
  for point_type, point in point_by_type.items():
    border_color = OCCLUDED_COLOR if point.is_occluded else None
    occlusion_marker = '?' if point.is_occluded else ''
    dots.append(
        Dot(point.location,
            config.point_colors[point_type],
            config.point_sizes[point_type] * config.dot_size_factor,
            name=f'{point_name(point_type)}{occlusion_marker}',
            border_color=border_color))
  return Wireframe(lines=lines, dots=dots)


def build_camera_wireframe(
    keypoints: Collection[keypoint_pb2.CameraKeypoint],
    config: WireframeConfig = DEFAULT_CAMERA_WIREFRAME_CONFIG) -> Wireframe:
  """Creates a wireframe for camera keypoints."""
  point_by_type = _data.camera_keypoint_coordinates(keypoints)
  return _build_wireframe(point_by_type, config)


def build_laser_wireframe(
    keypoints: Collection[keypoint_pb2.LaserKeypoint],
    config: WireframeConfig = DEFAULT_LASER_WIREFRAME_CONFIG) -> Wireframe:
  """Creates a wireframe for laser keypoints."""
  point_by_type = _data.laser_keypoint_coordinates(keypoints)
  return _build_wireframe(point_by_type, config)


def draw_camera_wireframe(ax: plt.Axes, wireframe: Wireframe) -> None:
  """Draws a camera wireframe onto the axes."""
  ax.add_collection(
      mc.LineCollection(
          segments=[[l.start, l.end] for l in wireframe.lines],
          colors=[l.color for l in wireframe.lines],
          linewidths=[l.width for l in wireframe.lines],
          antialiased=True))
  dots_sizes = [d.size for d in wireframe.dots]
  ax.add_collection(
      mc.EllipseCollection(
          widths=dots_sizes,
          heights=dots_sizes,
          angles=0,
          units='xy',
          offsets=[d.location for d in wireframe.dots],
          facecolors=[d.color for d in wireframe.dots],
          edgecolors=[d.actual_border_color for d in wireframe.dots],
          linewidth=OCCLUDED_BORDER_WIDTH,
          transOffset=ax.transData,
          antialiased=True))


def draw_laser_wireframe(fig: go.Figure, wireframe: Wireframe) -> None:
  """Draws a laser wireframe onto the plotly Figure."""
  for line in wireframe.lines:
    points = np.stack([line.start, line.end], axis=0)
    fig.add_trace(
        go.Scatter3d(
            mode='lines',
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            line=dict(color=line.color, width=line.width)))
  dot_coords = np.stack([d.location for d in wireframe.dots], axis=0)
  fig.add_trace(
      go.Scatter3d(
          text=[d.name for d in wireframe.dots],
          mode='markers',
          x=dot_coords[:, 0],
          y=dot_coords[:, 1],
          z=dot_coords[:, 2],
          marker=dict(
              color=[d.color for d in wireframe.dots],
              size=[d.size for d in wireframe.dots],
              line=dict(
                  width=OCCLUDED_BORDER_WIDTH,
                  color=[d.actual_border_color for d in wireframe.dots]))))


def crop_camera_keypoints(
    image: np.ndarray,
    keypoints: Collection[keypoint_pb2.CameraKeypoint],
    box: label_pb2.Label.Box,
    margin: float = 0
) -> Tuple[np.ndarray, Collection[keypoint_pb2.CameraKeypoint]]:
  """Crops camera image to the specified bounding box and shifts keypoints.

  Args:
    image: input image to crop, an array with shape [height, width, 3].
    keypoints: a collection of camera keypoints.
    box: a 2D bounding box to extract from the input image.
    margin: a ratio of the extra margin to add to the image relative to the
      input image size.

  Returns:
    a tuple (new image, shifted keypoints).
  """
  new_camera_keypoints = []
  crop_width = (1 + margin) * box.length
  crop_height = (1 + margin) * box.width
  min_x = max(0, int(box.center_x - crop_width / 2))
  min_y = max(0, int(box.center_y - crop_height / 2))
  for old_kp in keypoints:
    new_kp = keypoint_pb2.CameraKeypoint()
    new_kp.CopyFrom(old_kp)
    new_p = new_kp.keypoint_2d.location_px
    new_p.x -= min_x
    new_p.y -= min_y
    new_camera_keypoints.append(new_kp)
  max_x = min(image.shape[1] - 1, int(box.center_x + crop_width / 2))
  max_y = min(image.shape[0] - 1, int(box.center_y + crop_height / 2))
  new_image = image[min_y:max_y, min_x:max_x, :]
  return new_image, new_camera_keypoints
