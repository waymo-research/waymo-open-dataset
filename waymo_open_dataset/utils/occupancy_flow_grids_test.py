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
"""Tests for occupancy_flow_grids."""

import tensorflow as tf

from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils import occupancy_flow_renderer
from waymo_open_dataset.utils import occupancy_flow_test_util


class OccupancyFlowGridsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 8
    inputs = occupancy_flow_test_util.make_one_data_batch(
        batch_size=self.batch_size)
    self.inputs = occupancy_flow_data.add_sdc_fields(inputs)
    self.config = occupancy_flow_test_util.make_test_config()

  def test_create_ground_truth_timestep_grids(self):
    timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
        inputs=self.inputs, config=self.config)

    batch_size = self.batch_size
    height = self.config.grid_height_cells
    width = self.config.grid_width_cells
    num_past_steps = self.config.num_past_steps
    num_future_steps = self.config.num_future_steps

    waypoint_size = num_future_steps // self.config.num_waypoints
    num_all_steps = occupancy_flow_renderer._get_num_steps_from_times(
        times=['past', 'current', 'future'], config=self.config)
    num_flow_steps = num_all_steps - waypoint_size

    for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
      # Occupancy.
      self.assertEqual(
          timestep_grids.view(object_type).past_occupancy.shape,
          (batch_size, height, width, num_past_steps))
      self.assertEqual(
          timestep_grids.view(object_type).current_occupancy.shape,
          (batch_size, height, width, 1))
      self.assertEqual(
          timestep_grids.view(object_type).future_observed_occupancy.shape,
          (batch_size, height, width, num_future_steps))
      self.assertEqual(
          timestep_grids.view(object_type).future_occluded_occupancy.shape,
          (batch_size, height, width, num_future_steps))
      # All occupancy for flow origin.
      self.assertEqual(
          timestep_grids.view(object_type).all_occupancy.shape,
          (batch_size, height, width, num_all_steps))
      # Flow.
      self.assertEqual(
          timestep_grids.view(object_type).all_flow.shape,
          (batch_size, height, width, num_flow_steps, 2))

      # The test scene contains all agent classes.  Verify some values too.
      current_occupancy = timestep_grids.view(object_type).current_occupancy
      all_flow = timestep_grids.view(object_type).all_flow
      self.assertEqual(tf.reduce_min(current_occupancy), 0)
      self.assertEqual(tf.reduce_max(current_occupancy), 1)
      self.assertLess(tf.reduce_min(all_flow), 0)
      self.assertGreater(tf.reduce_max(all_flow), 0)

  def test_create_ground_truth_waypoint_grids(self):
    timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
        inputs=self.inputs, config=self.config)
    true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
        timestep_grids=timestep_grids, config=self.config)

    batch_size = self.batch_size
    height = self.config.grid_height_cells
    width = self.config.grid_width_cells
    num_waypoints = self.config.num_waypoints

    for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
      self.assertLen(
          true_waypoints.view(object_type).observed_occupancy, num_waypoints)
      self.assertLen(
          true_waypoints.view(object_type).occluded_occupancy, num_waypoints)
      self.assertLen(
          true_waypoints.view(object_type).flow_origin_occupancy, num_waypoints)
      self.assertLen(true_waypoints.view(object_type).flow, num_waypoints)
      self.assertEqual(
          true_waypoints.view(object_type).observed_occupancy[0].shape,
          (batch_size, height, width, 1))
      self.assertEqual(
          true_waypoints.view(object_type).occluded_occupancy[0].shape,
          (batch_size, height, width, 1))
      self.assertEqual(
          true_waypoints.view(object_type).flow_origin_occupancy[0].shape,
          (batch_size, height, width, 1))
      self.assertEqual(
          true_waypoints.view(object_type).flow[0].shape,
          (batch_size, height, width, 2))

  def test_create_ground_truth_vis_grids(self):
    timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
        inputs=self.inputs, config=self.config)
    vis_grids = occupancy_flow_grids.create_ground_truth_vis_grids(
        inputs=self.inputs,
        timestep_grids=timestep_grids,
        config=self.config,
    )

    batch_size = self.batch_size
    height = self.config.grid_height_cells
    width = self.config.grid_width_cells

    self.assertEqual(vis_grids.roadgraph.shape, (batch_size, height, width, 1))
    self.assertEqual(vis_grids.agent_trails.shape,
                     (batch_size, height, width, 1))


if __name__ == '__main__':
  tf.test.main()
