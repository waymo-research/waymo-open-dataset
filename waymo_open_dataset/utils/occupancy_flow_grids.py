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
"""Data structures for holding GT/predicted occupancy grids and flow fields."""

import dataclasses
import functools
from typing import List, Mapping, Optional

import tensorflow as tf

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_renderer

_ObjectType = scenario_pb2.Track.ObjectType


# Holds ground-truth occupancy and flow tensors for each timestep in
# past/current/future for one agent class.
@dataclasses.dataclass
class _TimestepGridsOneType:
  """Occupancy and flow tensors over past/current/future for one agent type."""
  # [batch_size, height, width, 1]
  current_occupancy: Optional[tf.Tensor] = None
  # [batch_size, height, width, num_past_steps]
  past_occupancy: Optional[tf.Tensor] = None
  # [batch_size, height, width, num_future_steps]
  future_observed_occupancy: Optional[tf.Tensor] = None
  # [batch_size, height, width, num_future_steps]
  future_occluded_occupancy: Optional[tf.Tensor] = None
  # Backward flow (dx, dy) for all observed and occluded agents.  Flow is
  # constructed between timesteps `waypoint_size` apart over all timesteps in
  # [past, current, future].  The flow for each timestep `t` contains (dx, dy)
  # occupancy displacements from timestep `t` to timestep `t - waypoints_size`,
  # which is EARLIER than t (hence backward flow).
  # waypoint_size = num_future_steps // num_waypoints
  # num_flow_steps = (num_past_steps + 1 + num_future_steps) - waypoint_size
  # [batch_size, height, width, num_flow_steps, 2]
  all_flow: Optional[tf.Tensor] = None
  # Observed and occluded occupancy over all timesteps.  This is used to
  # generate flow_origin tensors in WaypointGrids.
  # [batch_size, height, width, num_past_steps + 1 + num_future_steps]
  all_occupancy: Optional[tf.Tensor] = None


# Holds ground-truth occupancy and flow tensors for each timestep in
# past/current/future for all agent classes.
@dataclasses.dataclass
class TimestepGrids:
  """Occupancy and flow for vehicles, pedestrians, cyclists."""
  vehicles: _TimestepGridsOneType = dataclasses.field(
      default_factory=_TimestepGridsOneType)
  pedestrians: _TimestepGridsOneType = dataclasses.field(
      default_factory=_TimestepGridsOneType)
  cyclists: _TimestepGridsOneType = dataclasses.field(
      default_factory=_TimestepGridsOneType)

  def view(self, agent_type: str) -> _TimestepGridsOneType:
    """Retrieve occupancy and flow tensors for given agent type."""
    if agent_type == _ObjectType.TYPE_VEHICLE:
      return self.vehicles
    elif agent_type == _ObjectType.TYPE_PEDESTRIAN:
      return self.pedestrians
    elif agent_type == _ObjectType.TYPE_CYCLIST:
      return self.cyclists
    else:
      raise ValueError(f'Unknown agent type:{agent_type}.')


# Holds num_waypoints occupancy and flow tensors for one agent class.
@dataclasses.dataclass
class _WaypointGridsOneType:
  """Sequence of num_waypoints occupancy and flow tensors for one agent type."""
  # num_waypoints tensors shaped [batch_size, height, width, 1]
  observed_occupancy: List[tf.Tensor] = dataclasses.field(default_factory=list)
  # num_waypoints tensors shaped [batch_size, height, width, 1]
  occluded_occupancy: List[tf.Tensor] = dataclasses.field(default_factory=list)
  # num_waypoints tensors shaped [batch_size, height, width, 2]
  flow: List[tf.Tensor] = dataclasses.field(default_factory=list)
  # The origin occupancy for each flow waypoint.  Notice that a flow field
  # transforms some origin occupancy into some destination occupancy.
  # Flow-origin occupancies are the base occupancies for each flow field.
  # num_waypoints tensors shaped [batch_size, height, width, 1]
  flow_origin_occupancy: List[tf.Tensor] = dataclasses.field(
      default_factory=list)


# Holds num_waypoints occupancy and flow tensors for all agent clases.  This is
# used to store both ground-truth and predicted topdowns.
@dataclasses.dataclass
class WaypointGrids:
  """Occupancy and flow sequences for vehicles, pedestrians, cyclists."""
  vehicles: _WaypointGridsOneType = dataclasses.field(
      default_factory=_WaypointGridsOneType)
  pedestrians: _WaypointGridsOneType = dataclasses.field(
      default_factory=_WaypointGridsOneType)
  cyclists: _WaypointGridsOneType = dataclasses.field(
      default_factory=_WaypointGridsOneType)

  def view(self, agent_type: str) -> _WaypointGridsOneType:
    """Retrieve occupancy and flow sequences for given agent type."""
    if agent_type == _ObjectType.TYPE_VEHICLE:
      return self.vehicles
    elif agent_type == _ObjectType.TYPE_PEDESTRIAN:
      return self.pedestrians
    elif agent_type == _ObjectType.TYPE_CYCLIST:
      return self.cyclists
    else:
      raise ValueError(f'Unknown agent type:{agent_type}.')

  def get_observed_occupancy_at_waypoint(
      self, k: int) -> occupancy_flow_data.AgentGrids:
    """Returns occupancies of currently-observed agents at waypoint k."""
    agent_grids = occupancy_flow_data.AgentGrids()
    if self.vehicles.observed_occupancy:
      agent_grids.vehicles = self.vehicles.observed_occupancy[k]
    if self.pedestrians.observed_occupancy:
      agent_grids.pedestrians = self.pedestrians.observed_occupancy[k]
    if self.cyclists.observed_occupancy:
      agent_grids.cyclists = self.cyclists.observed_occupancy[k]
    return agent_grids

  def get_occluded_occupancy_at_waypoint(
      self, k: int) -> occupancy_flow_data.AgentGrids:
    """Returns occupancies of currently-occluded agents at waypoint k."""
    agent_grids = occupancy_flow_data.AgentGrids()
    if self.vehicles.occluded_occupancy:
      agent_grids.vehicles = self.vehicles.occluded_occupancy[k]
    if self.pedestrians.occluded_occupancy:
      agent_grids.pedestrians = self.pedestrians.occluded_occupancy[k]
    if self.cyclists.occluded_occupancy:
      agent_grids.cyclists = self.cyclists.occluded_occupancy[k]
    return agent_grids

  def get_flow_at_waypoint(self, k: int) -> occupancy_flow_data.AgentGrids:
    """Returns flow fields of all agents at waypoint k."""
    agent_grids = occupancy_flow_data.AgentGrids()
    if self.vehicles.flow:
      agent_grids.vehicles = self.vehicles.flow[k]
    if self.pedestrians.flow:
      agent_grids.pedestrians = self.pedestrians.flow[k]
    if self.cyclists.flow:
      agent_grids.cyclists = self.cyclists.flow[k]
    return agent_grids


# Holds topdown renders of scene objects suitable for visualization.
@dataclasses.dataclass
class VisGrids:
  # Roadgraph elements.
  # [batch_size, height, width, 1]
  roadgraph: Optional[tf.Tensor] = None
  # Trail of scene agents over past and current time frames.
  # [batch_size, height, width, 1]
  agent_trails: Optional[tf.Tensor] = None


def create_ground_truth_timestep_grids(
    inputs: Mapping[str, tf.Tensor],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> TimestepGrids:
  """Renders topdown views of agents over past/current/future time frames.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    TimestepGrids object holding topdown renders of agents.
  """
  timestep_grids = TimestepGrids()

  # Occupancy grids.
  render_func = functools.partial(
      occupancy_flow_renderer.render_occupancy_from_inputs,
      inputs=inputs,
      config=config)

  current_occupancy = render_func(
      times=['current'],
      include_observed=True,
      include_occluded=True,
  )
  timestep_grids.vehicles.current_occupancy = current_occupancy.vehicles
  timestep_grids.pedestrians.current_occupancy = current_occupancy.pedestrians
  timestep_grids.cyclists.current_occupancy = current_occupancy.cyclists

  past_occupancy = render_func(
      times=['past'],
      include_observed=True,
      include_occluded=True,
  )
  timestep_grids.vehicles.past_occupancy = past_occupancy.vehicles
  timestep_grids.pedestrians.past_occupancy = past_occupancy.pedestrians
  timestep_grids.cyclists.past_occupancy = past_occupancy.cyclists

  future_obs = render_func(
      times=['future'],
      include_observed=True,
      include_occluded=False,
  )
  timestep_grids.vehicles.future_observed_occupancy = future_obs.vehicles
  timestep_grids.pedestrians.future_observed_occupancy = future_obs.pedestrians
  timestep_grids.cyclists.future_observed_occupancy = future_obs.cyclists

  future_occ = render_func(
      times=['future'],
      include_observed=False,
      include_occluded=True,
  )
  timestep_grids.vehicles.future_occluded_occupancy = future_occ.vehicles
  timestep_grids.pedestrians.future_occluded_occupancy = future_occ.pedestrians
  timestep_grids.cyclists.future_occluded_occupancy = future_occ.cyclists

  # All occupancy for flow_origin_occupancy.
  all_occupancy = render_func(
      times=['past', 'current', 'future'],
      include_observed=True,
      include_occluded=True,
  )
  timestep_grids.vehicles.all_occupancy = all_occupancy.vehicles
  timestep_grids.pedestrians.all_occupancy = all_occupancy.pedestrians
  timestep_grids.cyclists.all_occupancy = all_occupancy.cyclists

  # Flow.
  # NOTE: Since the future flow depends on the current and past timesteps, we
  # need to compute it from [past + current + future] sparse points.
  all_flow = occupancy_flow_renderer.render_flow_from_inputs(
      inputs=inputs,
      times=['past', 'current', 'future'],
      config=config,
      include_observed=True,
      include_occluded=True,
  )
  timestep_grids.vehicles.all_flow = all_flow.vehicles
  timestep_grids.pedestrians.all_flow = all_flow.pedestrians
  timestep_grids.cyclists.all_flow = all_flow.cyclists

  return timestep_grids


def create_ground_truth_waypoint_grids(
    timestep_grids: TimestepGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> WaypointGrids:
  """Subsamples or aggregates future topdowns as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    WaypointGrids object.
  """
  if config.num_future_steps % config.num_waypoints != 0:
    raise ValueError(f'num_future_steps({config.num_future_steps}) must be '
                     f'a multiple of num_waypoints({config.num_waypoints}).')

  true_waypoints = WaypointGrids(
      vehicles=_WaypointGridsOneType(
          observed_occupancy=[], occluded_occupancy=[], flow=[]),
      pedestrians=_WaypointGridsOneType(
          observed_occupancy=[], occluded_occupancy=[], flow=[]),
      cyclists=_WaypointGridsOneType(
          observed_occupancy=[], occluded_occupancy=[], flow=[]),
  )

  # Observed occupancy.
  _add_ground_truth_observed_occupancy_to_waypoint_grids(
      timestep_grids=timestep_grids,
      waypoint_grids=true_waypoints,
      config=config)
  # Occluded occupancy.
  _add_ground_truth_occluded_occupancy_to_waypoint_grids(
      timestep_grids=timestep_grids,
      waypoint_grids=true_waypoints,
      config=config)
  # Flow origin occupancy.
  _add_ground_truth_flow_origin_occupancy_to_waypoint_grids(
      timestep_grids=timestep_grids,
      waypoint_grids=true_waypoints,
      config=config)
  # Flow.
  _add_ground_truth_flow_to_waypoint_grids(
      timestep_grids=timestep_grids,
      waypoint_grids=true_waypoints,
      config=config)

  return true_waypoints


def _add_ground_truth_observed_occupancy_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates future topdowns as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  waypoint_size = config.num_future_steps // config.num_waypoints
  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # [batch_size, height, width, num_future_steps]
    future_obs = timestep_grids.view(object_type).future_observed_occupancy
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size]
        segment = future_obs[..., waypoint_start:waypoint_end]
        # [batch_size, height, width, 1]
        waypoint_occupancy = tf.reduce_max(segment, axis=-1, keepdims=True)
      else:
        # [batch_size, height, width, 1]
        waypoint_occupancy = future_obs[..., waypoint_end - 1:waypoint_end]
      waypoint_grids.view(object_type).observed_occupancy.append(
          waypoint_occupancy)


def _add_ground_truth_occluded_occupancy_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates future topdowns as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  waypoint_size = config.num_future_steps // config.num_waypoints
  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # [batch_size, height, width, num_future_steps]
    future_occ = timestep_grids.view(object_type).future_occluded_occupancy
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size]
        segment = future_occ[..., waypoint_start:waypoint_end]
        # [batch_size, height, width, 1]
        waypoint_occupancy = tf.reduce_max(segment, axis=-1, keepdims=True)
      else:
        # [batch_size, height, width, 1]
        waypoint_occupancy = future_occ[..., waypoint_end - 1:waypoint_end]
      waypoint_grids.view(object_type).occluded_occupancy.append(
          waypoint_occupancy)


def _add_ground_truth_flow_origin_occupancy_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates topdowns as origin occupancies for flow fields.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  waypoint_size = config.num_future_steps // config.num_waypoints
  num_history_steps = config.num_past_steps + 1  # Includes past + current.
  num_future_steps = config.num_future_steps
  if waypoint_size > num_history_steps:
    raise ValueError('If waypoint_size > num_history_steps, we cannot find the '
                     'flow origin occupancy for the first waypoint.')

  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # [batch_size, height, width, num_past_steps + 1 + num_future_steps]
    all_occupancy = timestep_grids.view(object_type).all_occupancy
    # Keep only the section containing flow_origin_occupancy timesteps.
    # First remove `waypoint_size` from the end.  Then keep the tail containing
    # num_future_steps timesteps.
    flow_origin_occupancy = all_occupancy[:, :, :, :-waypoint_size]
    # [batch_size, height, width, num_future_steps]
    flow_origin_occupancy = flow_origin_occupancy[:, :, :, -num_future_steps:]
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size]
        segment = flow_origin_occupancy[..., waypoint_start:waypoint_end]
        # [batch_size, height, width, 1]
        waypoint_flow_origin = tf.reduce_max(segment, axis=-1, keepdims=True)
      else:
        # [batch_size, height, width, 1]
        waypoint_flow_origin = flow_origin_occupancy[..., waypoint_end -
                                                     1:waypoint_end]
      waypoint_grids.view(object_type).flow_origin_occupancy.append(
          waypoint_flow_origin)


def _add_ground_truth_flow_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates future flow fields as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  num_future_steps = config.num_future_steps
  waypoint_size = config.num_future_steps // config.num_waypoints

  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # num_flow_steps = (num_past_steps + num_futures_steps) - waypoint_size
    # [batch_size, height, width, num_flow_steps, 2]
    flow = timestep_grids.view(object_type).all_flow
    # Keep only the flow tail, containing num_future_steps timesteps.
    # [batch_size, height, width, num_future_steps, 2]
    flow = flow[:, :, :, -num_future_steps:, :]
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size, 2]
        segment = flow[:, :, :, waypoint_start:waypoint_end, :]
        # Compute mean flow over the timesteps in this segment by counting
        # the number of pixels with non-zero flow and dividing the flow sum
        # by that number.
        # [batch_size, height, width, waypoint_size, 2]
        occupied_pixels = tf.cast(tf.not_equal(segment, 0.0), tf.float32)
        # [batch_size, height, width, 2]
        num_flow_values = tf.reduce_sum(occupied_pixels, axis=3)
        # [batch_size, height, width, 2]
        segment_sum = tf.reduce_sum(segment, axis=3)
        # [batch_size, height, width, 2]
        mean_flow = tf.math.divide_no_nan(segment_sum, num_flow_values)
        waypoint_flow = mean_flow
      else:
        waypoint_flow = flow[:, :, :, waypoint_end - 1, :]
      waypoint_grids.view(object_type).flow.append(waypoint_flow)


def create_ground_truth_vis_grids(
    inputs: Mapping[str, tf.Tensor],
    timestep_grids: TimestepGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> VisGrids:
  """Creates topdown renders of roadgraph and agent trails for visualization.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    timestep_grids: Holds topdown renders of agents over time.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    VisGrids object holding roadgraph and agent trails over past, current time.
  """
  roadgraph = occupancy_flow_renderer.render_roadgraph_from_inputs(
      inputs, config)
  agent_trails = _create_agent_trails(timestep_grids)

  return VisGrids(
      roadgraph=roadgraph,
      agent_trails=agent_trails,
  )


def _create_agent_trails(
    timestep_grids: TimestepGrids,
    gamma: float = 0.80,
) -> tf.Tensor:
  """Renders trails for all agents over the past and current time frames.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    gamma: Decay for older boxes.

  Returns:
    Agent trails as [batch_size, height, width, 1] float32 tensor.
  """
  agent_trails = 0.0
  num_past = tf.convert_to_tensor(
      timestep_grids.vehicles.past_occupancy).shape.as_list()[-1]
  for i in range(num_past):
    new_agents = (
        timestep_grids.vehicles.past_occupancy[..., i:i + 1] +
        timestep_grids.pedestrians.past_occupancy[..., i:i + 1] +
        timestep_grids.cyclists.past_occupancy[..., i:i + 1])
    agent_trails = tf.clip_by_value(agent_trails * gamma + new_agents, 0, 1)
  new_agents = (
      timestep_grids.vehicles.current_occupancy +
      timestep_grids.pedestrians.current_occupancy +
      timestep_grids.cyclists.current_occupancy)
  agent_trails = tf.clip_by_value(agent_trails * gamma * gamma + new_agents, 0,
                                  1)
  return agent_trails
