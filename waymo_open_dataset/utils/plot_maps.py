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
"""Functions for plotting open dataset map data with 3D interactive plots."""

import dataclasses
import enum
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from waymo_open_dataset.protos import map_pb2


# Constants defined for map drawing parameters.
class FeatureType(enum.Enum):
  """Definintions for map feature types."""
  UNKNOWN_FEATURE = 0
  FREEWAY_LANE = 1
  SURFACE_STREET_LANE = 2
  BIKE_LANE = 3
  BROKEN_SINGLE_WHITE_BOUNDARY = 6
  SOLID_SINGLE_WHITE_BOUNDARY = 7
  SOLID_DOUBLE_WHITE_BOUNDARY = 8
  BROKEN_SINGLE_YELLOW_BOUNDARY = 9
  BROKEN_DOUBLE_YELLOW_BOUNDARY = 10
  SOLID_SINGLE_YELLOW_BOUNDARY = 11
  SOLID_DOUBLE_YELLOW_BOUNDARY = 12
  PASSING_DOUBLE_YELLOW_BOUNDARY = 13
  ROAD_EDGE_BOUNDARY = 15
  ROAD_EDGE_MEDIAN = 16
  STOP_SIGN = 17
  CROSSWALK = 18
  SPEED_BUMP = 19
  DRIVEWAY = 20


@dataclasses.dataclass(frozen=True)
class MapPoints:
  """A container for map point data."""

  x: list[float] = dataclasses.field(default_factory=list)
  y: list[float] = dataclasses.field(default_factory=list)
  z: list[float] = dataclasses.field(default_factory=list)
  types: list[FeatureType] = dataclasses.field(default_factory=list)
  ids: list[int] = dataclasses.field(default_factory=list)

  def append_point(
      self, point: map_pb2.MapPoint, feature_type: FeatureType, feature_id: int
  ):
    """Append a given point to the container."""
    self.x.append(point.x)
    self.y.append(point.y)
    self.z.append(point.z)
    self.types.append(feature_type)
    self.ids.append(feature_id)


def plot_map_points(map_points: MapPoints) -> go._figure.Figure:
  """Creates an interactive visualization of map data.

  Args:
    map_points: The set of map points to plot.

  Returns:
    A plotly figure object.
  """

  line_dict = {
      FeatureType.UNKNOWN_FEATURE: ('black', 'solid'),
      FeatureType.FREEWAY_LANE: ('white', 'solid'),
      FeatureType.SURFACE_STREET_LANE: ('royalblue', 'solid'),
      FeatureType.BIKE_LANE: ('magenta', 'solid'),
      FeatureType.BROKEN_SINGLE_WHITE_BOUNDARY: ('lightgray', 'dash'),
      FeatureType.SOLID_SINGLE_WHITE_BOUNDARY: ('lightgray', 'solid'),
      FeatureType.SOLID_DOUBLE_WHITE_BOUNDARY: ('lightgray', 'solid'),
      FeatureType.BROKEN_SINGLE_YELLOW_BOUNDARY: ('yellow', 'dash'),
      FeatureType.BROKEN_DOUBLE_YELLOW_BOUNDARY: ('yellow', 'dash'),
      FeatureType.SOLID_SINGLE_YELLOW_BOUNDARY: ('yellow', 'solid'),
      FeatureType.SOLID_DOUBLE_YELLOW_BOUNDARY: ('yellow', 'solid'),
      FeatureType.PASSING_DOUBLE_YELLOW_BOUNDARY: ('yellow', 'dash'),
      FeatureType.ROAD_EDGE_BOUNDARY: ('green', 'solid'),
      FeatureType.ROAD_EDGE_MEDIAN: ('green', 'solid'),
      FeatureType.STOP_SIGN: ('red', 'solid'),
      FeatureType.CROSSWALK: ('orange', 'solid'),
      FeatureType.SPEED_BUMP: ('cyan', 'solid'),
      FeatureType.DRIVEWAY: ('blue', 'solid'),
  }

  # Create a scatter plot of all points in the roadgraph data.
  feature_types_str = list(map(str, map_points.types))
  data1 = {
      'x': map_points.x,
      'y': map_points.y,
      'z': map_points.z,
      'feature_type': feature_types_str,
      'id': map_points.ids,
  }
  df = pd.DataFrame(data1)

  color_dict = {}
  for k in line_dict:
    color_dict[str(k)] = line_dict[k][0]

  fig = px.scatter_3d(
      df,
      x='x',
      y='y',
      z='z',
      color='feature_type',
      color_discrete_map=color_dict,
  )
  fig.update_traces(marker_size=1.25)

  # Plot connecting lines for each individual roadgraph feature.
  start_index = 0
  end_index = 0
  point_type = map_points.types[0]
  feature_id = map_points.ids[0]

  color_dict = {}
  for k in line_dict:
    color_dict[k] = line_dict[k][0]

  num_points = len(map_points.x)
  while start_index < num_points:
    while end_index < num_points and map_points.ids[end_index] == feature_id:
      end_index += 1

    xvals = map_points.x[start_index:end_index]
    yvals = map_points.y[start_index:end_index]
    zvals = map_points.z[start_index:end_index]

    # Plot stop signs as larger points in 3D.
    width = 1.5
    if point_type == 1 or point_type == 2 or point_type == 3:
      width = 2.5
    if point_type == 17:
      fig.add_trace(
          go.Scatter3d(
              x=xvals,
              y=yvals,
              z=zvals,
              mode='markers',
              marker=dict(
                  size=4,
                  color=color_dict[point_type],
              ),
          )
      )
    else:
      fig.add_trace(
          go.Scatter3d(
              x=xvals,
              y=yvals,
              z=zvals,
              mode='lines',
              line=dict(
                  dash=line_dict[point_type][1],
                  color=color_dict[point_type],
                  width=width,
              ),
          )
      )

    start_index = end_index
    if start_index < num_points:
      point_type = map_points.types[start_index]
      feature_id = map_points.ids[start_index]

  # Format the plot.
  axis_config = dict(
      backgroundcolor='rgb(0, 0, 0)',
      gridcolor='gray',
      showgrid=False,
      showline=False,
      showticklabels=False,
      showbackground=True,
      zerolinecolor='gray',
      tickfont=dict(color='gray'),
  )
  fig.update_layout(
      showlegend=False,
      scene=dict(xaxis=axis_config, yaxis=axis_config, zaxis=axis_config),
      width=1600,
      height=1200,
      paper_bgcolor='black',
      plot_bgcolor='rgba(0,0,0,0)',
      scene_aspectmode='data',
  )
  fig.update_yaxes(
      scaleanchor='x',
      scaleratio=1,
  )

  # Set the initial camera position. This sets the camera to be directly above
  # the center of the scene looking down from a height to view the majority
  # of the scene.
  camera = dict(
      up=dict(x=0, y=0, z=1),
      center=dict(x=0, y=0, z=0),
      eye=dict(x=0, y=0, z=3),
  )
  fig.update_layout(scene_camera=camera)

  return fig


def plot_map_features(
    map_features: List[map_pb2.MapFeature],
) -> go._figure.Figure:
  """Plots the map data for a Scenario proto from the open dataset.

  Args:
    map_features: A list of map features to be plotted.

  Returns:
    A plotly figure object.
  """
  lane_types = {
      map_pb2.LaneCenter.TYPE_UNDEFINED: FeatureType.UNKNOWN_FEATURE,
      map_pb2.LaneCenter.TYPE_FREEWAY: FeatureType.FREEWAY_LANE,
      map_pb2.LaneCenter.TYPE_SURFACE_STREET: FeatureType.SURFACE_STREET_LANE,
      map_pb2.LaneCenter.TYPE_BIKE_LANE: FeatureType.BIKE_LANE,
  }
  road_line_types = {
      map_pb2.RoadLine.TYPE_UNKNOWN: (
          FeatureType.UNKNOWN_FEATURE
      ),
      map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: (
          FeatureType.BROKEN_SINGLE_WHITE_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: (
          FeatureType.SOLID_SINGLE_WHITE_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: (
          FeatureType.SOLID_DOUBLE_WHITE_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: (
          FeatureType.BROKEN_SINGLE_YELLOW_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: (
          FeatureType.BROKEN_DOUBLE_YELLOW_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: (
          FeatureType.SOLID_SINGLE_YELLOW_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: (
          FeatureType.PASSING_DOUBLE_YELLOW_BOUNDARY
      ),
  }
  road_edge_types = {
      map_pb2.RoadEdge.TYPE_UNKNOWN: FeatureType.UNKNOWN_FEATURE,
      map_pb2.RoadEdge.TYPE_ROAD_EDGE_BOUNDARY: FeatureType.ROAD_EDGE_BOUNDARY,
      map_pb2.RoadEdge.TYPE_ROAD_EDGE_MEDIAN: FeatureType.ROAD_EDGE_MEDIAN,
  }

  def add_points(
      feature_id: int,
      points: List[map_pb2.MapPoint],
      feature_type: FeatureType,
      map_points: MapPoints,
      is_polygon=False,
  ):
    if feature_type is None:
      return
    for point in points:
      map_points.append_point(point, feature_type, feature_id)

    if is_polygon:
      map_points.append_point(points[0], feature_type, feature_id)

  # Create arrays of the map points to be plotted.
  map_points = MapPoints()

  for feature in map_features:
    if feature.HasField('lane'):
      add_points(
          feature.id,
          list(feature.lane.polyline),
          lane_types.get(feature.lane.type),
          map_points,
      )
    elif feature.HasField('road_line'):
      feature_type = road_line_types.get(feature.road_line.type)
      add_points(
          feature.id, list(feature.road_line.polyline), feature_type, map_points
      )
    elif feature.HasField('road_edge'):
      feature_type = road_edge_types.get(feature.road_edge.type)
      add_points(
          feature.id, list(feature.road_edge.polyline), feature_type, map_points
      )
    elif feature.HasField('stop_sign'):
      add_points(
          feature.id,
          [feature.stop_sign.position],
          FeatureType.STOP_SIGN,
          map_points,
      )
    elif feature.HasField('crosswalk'):
      add_points(
          feature.id,
          list(feature.crosswalk.polygon),
          FeatureType.CROSSWALK,
          map_points,
          True,
      )
    elif feature.HasField('speed_bump'):
      add_points(
          feature.id,
          list(feature.speed_bump.polygon),
          FeatureType.SPEED_BUMP,
          map_points,
          True,
      )
    elif feature.HasField('driveway'):
      add_points(
          feature.id,
          list(feature.driveway.polygon),
          FeatureType.DRIVEWAY,
          map_points,
          True,
      )

  # Return the interactive 3D map plot.
  return plot_map_points(map_points)
