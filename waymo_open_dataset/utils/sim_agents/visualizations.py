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
# =============================================================================
"""Visualizations for the Sim Agents Challenge tutorial."""

from matplotlib import animation
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.protos import scenario_pb2

# ==== CONSTANTS ====
_ROAD_EDGE_COLOR = 'k--'
_ROAD_EDGE_ALPHA = 1.0
_ROAD_LINE_COLOR = 'k--'
_ROAD_LINE_ALPHA = 0.5

_BBOX_ALPHA = 0.5

WAYMO_COLORS = np.array([
    [0., 120., 255.],    # Waymo Blue
    [0., 232., 157.],    # Waymo Green
    [255., 205., 85.],   # Amber
    [244., 175., 145.],  # Coral
    [145., 80., 200.],   # Purple
    [0., 51., 102.],     # Navy
]) / 255.

# Interval between each frame in an animation. This effectively plays back the
# 10Hz data at 1x speed.
_ANIMATION_INTERVAL_MS = 100


def add_map(axis: plt.Axes,
            scenario: scenario_pb2.Scenario) -> None:
  """Adds the supported map features to a pyplot axis."""
  for map_feature in scenario.map_features:
    if map_feature.WhichOneof('feature_data') == 'road_edge':
      add_road_edge(axis, map_feature.road_edge)
    elif map_feature.WhichOneof('feature_data') == 'road_line':
      add_road_line(axis, map_feature.road_line)
    else:
      # Skip other features.
      pass


def add_road_edge(axis: plt.Axes, road_edge: map_pb2.RoadEdge) -> None:
  """Adds a road edge to a pyplot axis."""
  x, y = zip(*[(p.x, p.y) for p in road_edge.polyline])
  axis.plot(x, y, _ROAD_EDGE_COLOR, alpha=_ROAD_EDGE_ALPHA)


def add_road_line(axis: plt.Axes, road_line: map_pb2.RoadLine) -> None:
  """Adds a road line to a pyplot axis."""
  x, y = zip(*[(p.x, p.y) for p in road_line.polyline])
  axis.plot(x, y, _ROAD_LINE_COLOR, alpha=_ROAD_LINE_ALPHA)


def get_bbox_patch(
    x: float, y: float, bbox_yaw: float, length: float, width: float,
    color_idx: int = 0) -> patches.Rectangle:
  """Creates a pyplot rectangle Patch for the given bounding box.

  Args:
    x: The x-coordinate of the box's center.
    y: The y-coordinate of the box's center.
    bbox_yaw: The yaw of the bounding box, with the same specification as the
      Scenario protos in the dataset.
    length: Length of the bounding box, in meters.
    width: Width of the bounding box, in meters.
    color_idx: Index of the color for the bounding box, assigned from the
      `WAYMO_COLORS` palette.

  Returns:
    A rectangular patch to be added to a pyplot figure.
  """
  # Pyplot uses the left-rear corner of the box as a reference, so we first
  # define that in the object's reference frame.
  left_rear_object = np.array([-length / 2, -width / 2])
  # Rotate this point in the object's reference frame based on the yaw.
  rotation_matrix = np.array([[np.cos(bbox_yaw), -np.sin(bbox_yaw)],
                              [np.sin(bbox_yaw), np.cos(bbox_yaw)]])
  left_rear_rotated = rotation_matrix.dot(left_rear_object)
  # Lastly, translate to the box center in the global reference frame.
  left_rear_global = np.array([x, y]) + left_rear_rotated
  # Set the color from the palette. This is an RGBA value.
  color = list(WAYMO_COLORS[color_idx]) + [_BBOX_ALPHA]
  rect = patches.Rectangle(
      left_rear_global, length, width, angle=np.rad2deg(bbox_yaw), color=color)
  return rect


def add_all_current_objects(
    axis: plt.Axes, x: tf.Tensor, y: tf.Tensor, yaw: tf.Tensor,
    length: tf.Tensor, width: tf.Tensor, color_idx: tf.Tensor
    ) -> list[patches.Rectangle]:
  """Draws all bounding boxes at a given timestep.

  Args:
    axis: The pyplot axis to which bounding boxes are added.
    x: Array of shape (num_objects,) of x-coordinates.
    y: Array of shape (num_objects,) of y-coordinates.
    yaw: Array of shape (num_objects,) of bounding box yaws.
    length: Array of shape (num_objects,) of object lengths.
    width: Array of shape (num_objects,) of object width.
    color_idx: Array of shape (num_objects,) of color indices, picked from the
      `WAYMO_COLORS` palette.

  Returns:
    A list of Rectangle patches.
  """
  bboxes = []
  for i in range(x.shape[0]):
    bboxes.append(axis.add_patch(get_bbox_patch(
        x[i], y[i], yaw[i], length[i], width[i], color_idx[i])))
  return bboxes


def get_animated_states(
    fig: plt.Figure, axis: plt.Axes, scenario: scenario_pb2.Scenario,
    x: tf.Tensor, y: tf.Tensor, yaw: tf.Tensor, length: tf.Tensor,
    width: tf.Tensor, color_idx: tf.Tensor) -> animation.FuncAnimation:
  """Animates the states in a pyplot figure.

  Args:
    fig: The pyplot figure to animate.
    axis: The pyplot axis to which bounding boxes and map are added.
    scenario: The Scenario proto from which the map is extracted.
    x: Array of shape (num_objects, num_steps) of x-coordinates.
    y: Array of shape (num_objects, num_steps) of y-coordinates.
    yaw: Array of shape (num_objects, num_steps) of bounding box yaws.
    length: Array of shape (num_objects, num_steps) of object lengths.
    width: Array of shape (num_objects, num_steps) of object width.
    color_idx: Array of shape (num_objects, num_steps) of color indices, picked
      from the `WAYMO_COLORS` palette.

  Returns:
    An animation of the object states over the static map.
  """
  # To avoid a double figure (one static and one animated), we need to first
  # close the existing pyplot figure.
  plt.close(fig)

  # Add the static map features to the animation.
  add_map(axis, scenario)

  def animate(t: int) -> list[patches.Rectangle]:
    # At each animation step, we need to remove the existing patches. This can
    # only be done using the `pop()` operation.
    for _ in range(len(axis.patches)):
      axis.patches.pop()
    bboxes = add_all_current_objects(
        axis=axis, x=x[:, t], y=y[:, t], yaw=yaw[:, t], length=length[:, t],
        width=width[:, t], color_idx=color_idx[:, t])
    return bboxes

  return animation.FuncAnimation(
      fig, animate, frames=x.shape[1], interval=_ANIMATION_INTERVAL_MS,
      blit=True)
