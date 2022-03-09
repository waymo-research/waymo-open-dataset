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
"""Renders occupancy and flow ground truth from inputs."""

import dataclasses
import math
from typing import List, Mapping, Sequence, Tuple

import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data

_ObjectType = scenario_pb2.Track.ObjectType


@dataclasses.dataclass
class _SampledPoints:
  """Set of points sampled from agent boxes.

  All fields have shape -
  [batch_size, num_agents, num_steps, num_points] where num_points is
  (points_per_side_length * points_per_side_width).
  """
  # [batch, num_agents, num_steps, points_per_agent].
  x: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  y: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  z: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  agent_type: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  valid: tf.Tensor


def render_occupancy_from_inputs(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    include_observed: bool,
    include_occluded: bool,
) -> occupancy_flow_data.AgentGrids:
  """Creates topdown renders of agents grouped by agent class.

  Renders agent boxes by densely sampling points from their boxes.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    config: OccupancyFlowTaskConfig proto message.
    include_observed: Whether to include currently-observed agents.
    include_occluded: Whether to include currently-occluded agents.

  Returns:
    An AgentGrids object containing:
      vehicles: [batch_size, height, width, steps] float32 in [0, 1].
      pedestrians: [batch_size, height, width, steps] float32 in [0, 1].
      cyclists: [batch_size, height, width, steps] float32 in [0, 1].
      where steps is the number of timesteps covered in `times`.
  """
  sampled_points = _sample_and_filter_agent_points(
      inputs=inputs,
      times=times,
      config=config,
      include_observed=include_observed,
      include_occluded=include_occluded,
  )

  agent_x = sampled_points.x
  agent_y = sampled_points.y
  agent_type = sampled_points.agent_type
  agent_valid = sampled_points.valid

  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  batch_size, num_agents, num_steps, points_per_agent = agent_x.shape.as_list()
  topdown_shape = [
      batch_size, config.grid_height_cells, config.grid_width_cells, num_steps
  ]

  # Transform from world coordinates to topdown image coordinates.
  # All 3 have shape: [batch, num_agents, num_steps, points_per_agent]
  agent_x, agent_y, point_is_in_fov = _transform_to_image_coordinates(
      points_x=agent_x,
      points_y=agent_y,
      config=config,
  )
  assert_shapes([(point_is_in_fov,
                  [batch_size, num_agents, num_steps, points_per_agent])])

  # Filter out points from invalid objects.
  agent_valid = tf.cast(agent_valid, tf.bool)
  point_is_in_fov_and_valid = tf.logical_and(point_is_in_fov, agent_valid)

  occupancies = {}
  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # Collect points for each agent type, i.e., pedestrians and vehicles.
    agent_type_matches = tf.equal(agent_type, object_type)
    should_render_point = tf.logical_and(point_is_in_fov_and_valid,
                                         agent_type_matches)

    assert_shapes([
        (should_render_point,
         [batch_size, num_agents, num_steps, points_per_agent]),
    ])

    # Scatter points across topdown maps for each timestep.  The tensor
    # `point_indices` holds the indices where `should_render_point` is True.
    # It is a 2-D tensor with shape [n, 4], where n is the number of valid
    # agent points inside FOV.  Each row in this tensor contains indices over
    # the following 4 dimensions: (batch, agent, timestep, point).

    # [num_points_to_render, 4]
    point_indices = tf.cast(tf.where(should_render_point), tf.int32)
    # [num_points_to_render, 1]
    x_img_coord = tf.gather_nd(agent_x, point_indices)[..., tf.newaxis]
    y_img_coord = tf.gather_nd(agent_y, point_indices)[..., tf.newaxis]

    num_points_to_render = point_indices.shape.as_list()[0]
    assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                   (y_img_coord, [num_points_to_render, 1])])

    # [num_points_to_render, 4]
    xy_img_coord = tf.concat(
        [
            point_indices[:, :1],
            tf.cast(y_img_coord, tf.int32),
            tf.cast(x_img_coord, tf.int32),
            point_indices[:, 2:3],
        ],
        axis=1,
    )
    # [num_points_to_render]
    gt_values = tf.squeeze(tf.ones_like(x_img_coord, dtype=tf.float32), axis=-1)

    # [batch_size, grid_height_cells, grid_width_cells, num_steps]
    topdown = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
    assert_shapes([(topdown, topdown_shape)])

    # scatter_nd() accumulates values if there are repeated indices.  Since
    # we sample densely, this happens all the time.  Clip the final values.
    topdown = tf.clip_by_value(topdown, 0.0, 1.0)
    occupancies[object_type] = topdown

  return occupancy_flow_data.AgentGrids(
      vehicles=occupancies[_ObjectType.TYPE_VEHICLE],
      pedestrians=occupancies[_ObjectType.TYPE_PEDESTRIAN],
      cyclists=occupancies[_ObjectType.TYPE_CYCLIST],
  )


def render_flow_from_inputs(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    include_observed: bool,
    include_occluded: bool,
) -> occupancy_flow_data.AgentGrids:
  """Compute top-down flow between timesteps `waypoint_size` apart.

  Returns (dx, dy) for each timestep.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    config: OccupancyFlowTaskConfig proto message.
    include_observed: Whether to include currently-observed agents.
    include_occluded: Whether to include currently-occluded agents.

  Returns:
    An AgentGrids object containing:
      vehicles: [batch_size, height, width, num_flow_steps, 2] float32
      pedestrians: [batch_size, height, width, num_flow_steps, 2] float32
      cyclists: [batch_size, height, width, num_flow_steps, 2] float32
      where num_flow_steps = num_steps - waypoint_size, and num_steps is the
      number of timesteps covered in `times`.
  """
  sampled_points = _sample_and_filter_agent_points(
      inputs=inputs,
      times=times,
      config=config,
      include_observed=include_observed,
      include_occluded=include_occluded,
  )

  agent_x = sampled_points.x
  agent_y = sampled_points.y
  agent_type = sampled_points.agent_type
  agent_valid = sampled_points.valid

  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  batch_size, num_agents, num_steps, points_per_agent = agent_x.shape.as_list()
  # The timestep distance between flow steps.
  waypoint_size = config.num_future_steps // config.num_waypoints
  num_flow_steps = num_steps - waypoint_size
  topdown_shape = [
      batch_size, config.grid_height_cells, config.grid_width_cells,
      num_flow_steps
  ]

  # Transform from world coordinates to topdown image coordinates.
  # All 3 have shape: [batch, num_agents, num_steps, points_per_agent]
  agent_x, agent_y, point_is_in_fov = _transform_to_image_coordinates(
      points_x=agent_x,
      points_y=agent_y,
      config=config,
  )
  assert_shapes([(point_is_in_fov,
                  [batch_size, num_agents, num_steps, points_per_agent])])

  # Filter out points from invalid objects.
  agent_valid = tf.cast(agent_valid, tf.bool)

  # Backward Flow.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  dx = agent_x[:, :, :-waypoint_size, :] - agent_x[:, :, waypoint_size:, :]
  dy = agent_y[:, :, :-waypoint_size, :] - agent_y[:, :, waypoint_size:, :]
  assert_shapes([
      (dx, [batch_size, num_agents, num_flow_steps, points_per_agent]),
      (dy, [batch_size, num_agents, num_flow_steps, points_per_agent]),
  ])

  # Adjust other fields as well to reduce from num_steps to num_flow_steps.
  # agent_x, agent_y: Use later timesteps since flow vectors go back in time.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  agent_x = agent_x[:, :, waypoint_size:, :]
  agent_y = agent_y[:, :, waypoint_size:, :]
  # agent_type: Use later timesteps since flow vectors go back in time.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  agent_type = agent_type[:, :, waypoint_size:, :]
  # point_is_in_fov: Use later timesteps since flow vectors go back in time.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  point_is_in_fov = point_is_in_fov[:, :, waypoint_size:, :]
  # agent_valid: And the two timesteps.  They both need to be valid.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  agent_valid = tf.logical_and(agent_valid[:, :, waypoint_size:, :],
                               agent_valid[:, :, :-waypoint_size, :])

  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  point_is_in_fov_and_valid = tf.logical_and(point_is_in_fov, agent_valid)

  flows = {}
  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # Collect points for each agent type, i.e., pedestrians and vehicles.
    agent_type_matches = tf.equal(agent_type, object_type)
    should_render_point = tf.logical_and(point_is_in_fov_and_valid,
                                         agent_type_matches)
    assert_shapes([
        (should_render_point,
         [batch_size, num_agents, num_flow_steps, points_per_agent]),
    ])

    # [batch_size, height, width, num_flow_steps, 2]
    flow = _render_flow_points_for_one_agent_type(
        agent_x=agent_x,
        agent_y=agent_y,
        dx=dx,
        dy=dy,
        should_render_point=should_render_point,
        topdown_shape=topdown_shape,
    )
    flows[object_type] = flow

  return occupancy_flow_data.AgentGrids(
      vehicles=flows[_ObjectType.TYPE_VEHICLE],
      pedestrians=flows[_ObjectType.TYPE_PEDESTRIAN],
      cyclists=flows[_ObjectType.TYPE_CYCLIST],
  )


def _render_flow_points_for_one_agent_type(
    agent_x: tf.Tensor,
    agent_y: tf.Tensor,
    dx: tf.Tensor,
    dy: tf.Tensor,
    should_render_point: tf.Tensor,
    topdown_shape: List[int],
) -> tf.Tensor:
  """Renders topdown (dx, dy) flow for given agent points.

  Args:
    agent_x: [batch_size, num_agents, num_steps, points_per_agent].
    agent_y: [batch_size, num_agents, num_steps, points_per_agent].
    dx: [batch_size, num_agents, num_steps, points_per_agent].
    dy: [batch_size, num_agents, num_steps, points_per_agent].
    should_render_point: [batch_size, num_agents, num_steps, points_per_agent].
    topdown_shape: Shape of the output flow field.

  Returns:
    Rendered flow as [batch_size, height, width, num_flow_steps, 2] float32
      tensor.
  """
  assert_shapes = tf.debugging.assert_shapes

  # Scatter points across topdown maps for each timestep.  The tensor
  # `point_indices` holds the indices where `should_render_point` is True.
  # It is a 2-D tensor with shape [n, 4], where n is the number of valid
  # agent points inside FOV.  Each row in this tensor contains indices over
  # the following 4 dimensions: (batch, agent, timestep, point).

  # [num_points_to_render, 4]
  point_indices = tf.cast(tf.where(should_render_point), tf.int32)
  # [num_points_to_render, 1]
  x_img_coord = tf.gather_nd(agent_x, point_indices)[..., tf.newaxis]
  y_img_coord = tf.gather_nd(agent_y, point_indices)[..., tf.newaxis]

  num_points_to_render = point_indices.shape.as_list()[0]
  assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                 (y_img_coord, [num_points_to_render, 1])])

  # [num_points_to_render, 4]
  xy_img_coord = tf.concat(
      [
          point_indices[:, :1],
          tf.cast(y_img_coord, tf.int32),
          tf.cast(x_img_coord, tf.int32),
          point_indices[:, 2:3],
      ],
      axis=1,
  )
  # [num_points_to_render]
  gt_values_dx = tf.gather_nd(dx, point_indices)
  gt_values_dy = tf.gather_nd(dy, point_indices)

  # tf.scatter_nd() accumulates values when there are repeated indices.
  # Keep track of number of indices writing to the same pixel so we can
  # account for accumulated values.
  # [num_points_to_render]
  gt_values = tf.squeeze(tf.ones_like(x_img_coord, dtype=tf.float32), axis=-1)

  # [batch_size, grid_height_cells, grid_width_cells, num_flow_steps]
  flow_x = tf.scatter_nd(xy_img_coord, gt_values_dx, topdown_shape)
  flow_y = tf.scatter_nd(xy_img_coord, gt_values_dy, topdown_shape)
  num_values_per_pixel = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
  assert_shapes([
      (flow_x, topdown_shape),
      (flow_y, topdown_shape),
      (num_values_per_pixel, topdown_shape),
  ])

  # Undo the accumulation effect of tf.scatter_nd() for repeated indices.
  flow_x = tf.math.divide_no_nan(flow_x, num_values_per_pixel)
  flow_y = tf.math.divide_no_nan(flow_y, num_values_per_pixel)

  # [batch_size, grid_height_cells, grid_width_cells, num_flow_steps, 2]
  flow = tf.stack([flow_x, flow_y], axis=-1)
  assert_shapes([(flow, topdown_shape + [2])])
  return flow


def render_roadgraph_from_inputs(
    inputs: Mapping[str, tf.Tensor],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> tf.Tensor:
  """Creates a topdown render of roadgraph points.

  This function is mostly useful for visualization.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    Rendered roadgraph as [batch_size, height, width, 1] float32 tensor
      containing zeros and ones.
  """
  grid_height_cells = config.grid_height_cells
  grid_width_cells = config.grid_width_cells

  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  batch_size, num_rg_points, _ = (
      inputs['roadgraph_samples/xyz'].shape.as_list())
  topdown_shape = [batch_size, grid_height_cells, grid_width_cells, 1]

  # Translate the roadgraph points so that the autonomous vehicle is at the
  # origin.
  sdc_xyz = tf.concat(
      [
          inputs['sdc/current/x'],
          inputs['sdc/current/y'],
          inputs['sdc/current/z'],
      ],
      axis=1,
  )
  # [batch_size, 1, 3]
  sdc_xyz = sdc_xyz[:, tf.newaxis, :]
  # [batch_size, num_rg_points, 3]
  rg_points = inputs['roadgraph_samples/xyz'] - sdc_xyz

  # [batch_size, num_rg_points, 1]
  rg_valid = inputs['roadgraph_samples/valid']
  assert_shapes([(rg_points, [batch_size, num_rg_points, 3]),
                 (rg_valid, [batch_size, num_rg_points, 1])])
  # [batch_size, num_rg_points]
  rg_x, rg_y, _ = tf.unstack(rg_points, axis=-1)
  assert_shapes([(rg_x, [batch_size, num_rg_points]),
                 (rg_y, [batch_size, num_rg_points])])

  if config.normalize_sdc_yaw:
    angle = math.pi / 2 - inputs['sdc/current/bbox_yaw']
    rg_x, rg_y = rotate_points_around_origin(rg_x, rg_y, angle)

  # Transform from world coordinates to topdown image coordinates.
  # All 3 have shape: [batch, num_rg_points]
  rg_x, rg_y, point_is_in_fov = _transform_to_image_coordinates(
      points_x=rg_x,
      points_y=rg_y,
      config=config,
  )
  assert_shapes([(point_is_in_fov, [batch_size, num_rg_points])])

  # Filter out invalid points.
  point_is_valid = tf.cast(rg_valid[..., 0], tf.bool)
  # [batch, num_rg_points]
  should_render_point = tf.logical_and(point_is_in_fov, point_is_valid)

  # Scatter points across a topdown map.  The tensor `point_indices` holds the
  # indices where `should_render_point` is True.  It is a 2-D tensor with shape
  # [n, 2], where n is the number of valid roadgraph points inside FOV.  Each
  # row in this tensor contains indices over the following 2 dimensions:
  # (batch, point).

  # [num_points_to_render, 2] holding (batch index, point index).
  point_indices = tf.cast(tf.where(should_render_point), tf.int32)
  # [num_points_to_render, 1]
  x_img_coord = tf.gather_nd(rg_x, point_indices)[..., tf.newaxis]
  y_img_coord = tf.gather_nd(rg_y, point_indices)[..., tf.newaxis]

  num_points_to_render = point_indices.shape.as_list()[0]
  assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                 (y_img_coord, [num_points_to_render, 1])])

  # [num_points_to_render, 3]
  xy_img_coord = tf.concat(
      [
          point_indices[:, :1],
          tf.cast(y_img_coord, tf.int32),
          tf.cast(x_img_coord, tf.int32),
      ],
      axis=1,
  )
  # Set pixels with roadgraph points to 1.0, leave others at 0.0.
  # [num_points_to_render, 1]
  gt_values = tf.ones_like(x_img_coord, dtype=tf.float32)

  # [batch_size, grid_height_cells, grid_width_cells, 1]
  rg_viz = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
  assert_shapes([(rg_viz, topdown_shape)])

  # scatter_nd() accumulates values if there are repeated indices.  Clip the
  # final values to handle cases where two roadgraph points coincide.
  rg_viz = tf.clip_by_value(rg_viz, 0.0, 1.0)
  return rg_viz


def _transform_to_image_coordinates(
    points_x: tf.Tensor,
    points_y: tf.Tensor,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Returns transformed points and a mask indicating whether point is in image.

  Args:
    points_x: Tensor of any shape containing x values in world coordinates
      centered on the autonomous vehicle (see translate_sdc_to_origin).
    points_y: Tensor with same shape as points_x containing y values in world
      coordinates centered on the autonomous vehicle.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    Tuple containing the following tensors:
      - Transformed points_x.
      - Transformed points_y.
      - tf.bool tensor with same shape as points_x indicating which points are
        inside the FOV of the image after transformation.
  """
  pixels_per_meter = config.pixels_per_meter
  points_x = tf.round(points_x * pixels_per_meter) + config.sdc_x_in_grid
  points_y = tf.round(-points_y * pixels_per_meter) + config.sdc_y_in_grid

  # Filter out points that are located outside the FOV of topdown map.
  point_is_in_fov = tf.logical_and(
      tf.logical_and(
          tf.greater_equal(points_x, 0), tf.greater_equal(points_y, 0)),
      tf.logical_and(
          tf.less(points_x, config.grid_width_cells),
          tf.less(points_y, config.grid_height_cells)))

  return points_x, points_y, point_is_in_fov


def _sample_and_filter_agent_points(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    include_observed: bool,
    include_occluded: bool,
) -> _SampledPoints:
  """Samples points and filters them according to current visibility of agents.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    config: OccupancyFlowTaskConfig proto message.
    include_observed: Whether to include currently-observed agents.
    include_occluded: Whether to include currently-occluded agents.

  Returns:
    _SampledPoints: containing x, y, z coordinates, type, and valid bits.
  """
  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  batch_size, num_agents, _ = (inputs['state/current/x'].shape.as_list())
  num_steps = _get_num_steps_from_times(times, config)
  points_per_agent = (
      config.agent_points_per_side_length * config.agent_points_per_side_width)

  # Sample points from agent boxes over specified time frames.
  # All fields have shape [batch_size, num_agents, num_steps, points_per_agent].
  sampled_points = _sample_agent_points(
      inputs,
      times=times,
      points_per_side_length=config.agent_points_per_side_length,
      points_per_side_width=config.agent_points_per_side_width,
      translate_sdc_to_origin=True,
      normalize_sdc_yaw=config.normalize_sdc_yaw,
  )

  field_shape = [batch_size, num_agents, num_steps, points_per_agent]
  assert_shapes([
      (sampled_points.x, field_shape),
      (sampled_points.y, field_shape),
      (sampled_points.z, field_shape),
      (sampled_points.valid, field_shape),
      (sampled_points.agent_type, field_shape),
  ])

  agent_valid = tf.cast(sampled_points.valid, tf.bool)
  # 1. If all agents are requested, no additional filtering is necessary.
  # 2. Filter observed/occluded agents for future only.
  include_all = include_observed and include_occluded
  if not include_all and 'future' in times:
    history_times = ['past', 'current']
    # [batch_size, num_agents, num_history_steps, 1]
    agent_is_observed = _stack_field(inputs, history_times, 'valid')
    # [batch_size, num_agents, 1, 1]
    agent_is_observed = tf.reduce_max(agent_is_observed, axis=2, keepdims=True)
    agent_is_observed = tf.cast(agent_is_observed, tf.bool)

    if include_observed:
      agent_filter = agent_is_observed
    elif include_occluded:
      agent_filter = tf.logical_not(agent_is_observed)
    else:  # Both observed and occluded are off.
      raise ValueError('Either observed or occluded agents must be requested.')

    assert_shapes([
        (agent_filter, [batch_size, num_agents, 1, 1]),
    ])

    agent_valid = tf.logical_and(agent_valid, agent_filter)

  return _SampledPoints(
      x=sampled_points.x,
      y=sampled_points.y,
      z=sampled_points.z,
      agent_type=sampled_points.agent_type,
      valid=agent_valid,
  )


def _sample_agent_points(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    points_per_side_length: int,
    points_per_side_width: int,
    translate_sdc_to_origin: bool,
    normalize_sdc_yaw: bool,
) -> _SampledPoints:
  """Creates a set of points to represent agents in the scene.

  For each timestep in `times`, samples the interior of each agent bounding box
  on a uniform grid to create a set of points representing the agent.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    points_per_side_length: The number of points along the length of the agent.
    points_per_side_width: The number of points along the width of the agent.
    translate_sdc_to_origin: If true, translate the points such that the
      autonomous vehicle is at the origin.
    normalize_sdc_yaw: If true, transform the scene such that the autonomous
      vehicle is heading up at the current time.

  Returns:
    _SampledPoints object.
  """
  if normalize_sdc_yaw and not translate_sdc_to_origin:
    raise ValueError('normalize_sdc_yaw requires translate_sdc_to_origin.')

  # All fields: [batch_size, num_agents, num_steps, 1].
  x = _stack_field(inputs, times, 'x')
  y = _stack_field(inputs, times, 'y')
  z = _stack_field(inputs, times, 'z')
  bbox_yaw = _stack_field(inputs, times, 'bbox_yaw')
  length = _stack_field(inputs, times, 'length')
  width = _stack_field(inputs, times, 'width')
  agent_type = _stack_field(inputs, times, 'type')
  valid = _stack_field(inputs, times, 'valid')
  shape = ['batch_size', 'num_agents', 'num_steps', 1]
  tf.debugging.assert_shapes([
      (x, shape),
      (y, shape),
      (z, shape),
      (bbox_yaw, shape),
      (length, shape),
      (width, shape),
      (valid, shape),
  ])

  # Translate all agent coordinates such that the autonomous vehicle is at the
  # origin.
  if translate_sdc_to_origin:
    sdc_x = inputs['sdc/current/x'][:, tf.newaxis, tf.newaxis, :]
    sdc_y = inputs['sdc/current/y'][:, tf.newaxis, tf.newaxis, :]
    sdc_z = inputs['sdc/current/z'][:, tf.newaxis, tf.newaxis, :]
    x = x - sdc_x
    y = y - sdc_y
    z = z - sdc_z

  if normalize_sdc_yaw:
    angle = math.pi / 2 - inputs['sdc/current/bbox_yaw'][:, tf.newaxis,
                                                         tf.newaxis, :]
    x, y = rotate_points_around_origin(x, y, angle)
    bbox_yaw = bbox_yaw + angle

  return _sample_points_from_agent_boxes(
      x=x,
      y=y,
      z=z,
      bbox_yaw=bbox_yaw,
      width=width,
      length=length,
      agent_type=agent_type,
      valid=valid,
      points_per_side_length=points_per_side_length,
      points_per_side_width=points_per_side_width,
  )


def _sample_points_from_agent_boxes(
    x: tf.Tensor,
    y: tf.Tensor,
    z: tf.Tensor,
    bbox_yaw: tf.Tensor,
    width: tf.Tensor,
    length: tf.Tensor,
    agent_type: tf.Tensor,
    valid: tf.Tensor,
    points_per_side_length: int,
    points_per_side_width: int,
) -> _SampledPoints:
  """Create a set of 3D points by sampling the interior of agent boxes.

  For each state in the inputs, a set of points_per_side_length *
  points_per_side_width points are generated by sampling within each box.

  Args:
    x: Centers of agent boxes X [..., 1] (any shape with last dim 1).
    y: Centers of agent boxes Y [..., 1] (same shape as x).
    z: Centers of agent boxes Z [..., 1] (same shape as x).
    bbox_yaw: Agent box orientations [..., 1] (same shape as x).
    width : Widths of agent boxes [..., 1] (same shape as x).
    length: Lengths of agent boxes [..., 1] (same shape as x).
    agent_type: Types of agents [..., 1] (same shape as x).
    valid: Agent state valid flag [..., 1] (same shape as x).
    points_per_side_length: The number of points along the length of the agent.
    points_per_side_width: The number of points along the width of the agent.

  Returns:
    _SampledPoints object.
  """
  assert_shapes = tf.debugging.assert_shapes
  assert_shapes([(x, [..., 1])])
  x_shape = x.get_shape().as_list()
  assert_shapes([(y, x_shape), (z, x_shape), (bbox_yaw, x_shape),
                 (width, x_shape), (length, x_shape), (valid, x_shape)])
  if points_per_side_length < 1:
    raise ValueError('points_per_side_length must be >= 1')
  if points_per_side_width < 1:
    raise ValueError('points_per_side_width must be >= 1')

  # Create sample points on a unit square or boundary depending on flag.
  if points_per_side_length == 1:
    step_x = 0.0
  else:
    step_x = 1.0 / (points_per_side_length - 1)
  if points_per_side_width == 1:
    step_y = 0.0
  else:
    step_y = 1.0 / (points_per_side_width - 1)
  unit_x = []
  unit_y = []
  for xi in range(points_per_side_length):
    for yi in range(points_per_side_width):
      unit_x.append(xi * step_x - 0.5)
      unit_y.append(yi * step_y - 0.5)

  # Center unit_x and unit_y if there was only 1 point on those dimensions.
  if points_per_side_length == 1:
    unit_x = np.array(unit_x) + 0.5
  if points_per_side_width == 1:
    unit_y = np.array(unit_y) + 0.5

  unit_x = tf.convert_to_tensor(unit_x, tf.float32)
  unit_y = tf.convert_to_tensor(unit_y, tf.float32)

  num_points = points_per_side_length * points_per_side_width
  assert_shapes([(unit_x, [num_points]), (unit_y, [num_points])])

  # Transform the unit square points to agent dimensions and coordinate frames.
  sin_yaw = tf.sin(bbox_yaw)
  cos_yaw = tf.cos(bbox_yaw)

  # [..., num_points]
  tx = cos_yaw * length * unit_x - sin_yaw * width * unit_y + x
  ty = sin_yaw * length * unit_x + cos_yaw * width * unit_y + y
  tz = tf.broadcast_to(z, tx.shape)

  points_shape = x_shape[:-1] + [num_points]
  assert_shapes([(tx, points_shape), (ty, points_shape), (tz, points_shape)])
  agent_type = tf.broadcast_to(agent_type, tx.shape)
  valid = tf.broadcast_to(valid, tx.shape)

  return _SampledPoints(x=tx, y=ty, z=tz, agent_type=agent_type, valid=valid)


def rotate_points_around_origin(
    x: tf.Tensor,
    y: tf.Tensor,
    angle: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Rotates points around the origin.

  Args:
    x: Tensor of shape [batch_size, ...].
    y: Tensor of shape [batch_size, ...].
    angle: Tensor of shape [batch_size, ...].

  Returns:
    Rotated x, y, each with shape [batch_size, ...].
  """
  tx = tf.cos(angle) * x - tf.sin(angle) * y
  ty = tf.sin(angle) * x + tf.cos(angle) * y
  return tx, ty


def _stack_field(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    field: str,
) -> tf.Tensor:
  """Stack requested field from all agents over specified time frames.

  NOTE: Always adds a last dimension with size 1.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    field: The field to retrieve.

  Returns:
    A tensor containing the requested field over the requested time frames
    with shape [batch_size, num_agents, num_steps, 1].
  """
  if field == 'type':
    # [batch_size, num_agents]
    fields = inputs['state/type']
    # The `type` field's shape is different from other fields.  Broadcast it
    # to have the same shape as another field.
    x = _stack_field(inputs, times, field='x')
    # [batch_size, num_agents, num_steps, 1]
    fields = tf.broadcast_to(fields[:, :, tf.newaxis, tf.newaxis], x.shape)
  else:
    # [batch_size, num_agents, num_steps]
    fields = tf.concat([inputs[f'state/{t}/{field}'] for t in times], axis=-1)
    # [batch_size, num_agents, num_steps, 1]
    fields = fields[:, :, :, tf.newaxis]
  return fields


def _get_num_steps_from_times(
    times: Sequence[str],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig) -> int:
  """Returns number of timesteps that exist in requested times."""
  num_steps = 0
  if 'past' in times:
    num_steps += config.num_past_steps
  if 'current' in times:
    num_steps += 1
  if 'future' in times:
    num_steps += config.num_future_steps
  return num_steps
