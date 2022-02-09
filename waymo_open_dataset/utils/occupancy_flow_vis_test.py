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
"""Tests for occupancy_flow_vis."""

import tensorflow as tf

from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils import occupancy_flow_test_util
from waymo_open_dataset.utils import occupancy_flow_vis


class OccupancyFlowVisTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 8
    inputs = occupancy_flow_test_util.make_one_data_batch(
        batch_size=self.batch_size)
    self.inputs = occupancy_flow_data.add_sdc_fields(inputs)
    self.config = occupancy_flow_test_util.make_test_config()
    # Create sample grids.
    timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
        inputs=self.inputs, config=self.config)
    self.true_waypoints = (
        occupancy_flow_grids.create_ground_truth_waypoint_grids(
            timestep_grids=timestep_grids, config=self.config))
    self.true_vis_grids = occupancy_flow_grids.create_ground_truth_vis_grids(
        inputs=self.inputs,
        timestep_grids=timestep_grids,
        config=self.config,
    )

  def test_occupancy_rgb_image(self):
    waypoint_1 = self.true_waypoints.get_observed_occupancy_at_waypoint(0)
    occupancy_rgb = occupancy_flow_vis.occupancy_rgb_image(
        agent_grids=waypoint_1,
        roadgraph_image=self.true_vis_grids.roadgraph,
    )

    batch_size = self.batch_size
    height = self.config.grid_height_cells
    width = self.config.grid_width_cells

    self.assertEqual(occupancy_rgb.shape, (batch_size, height, width, 3))

  def test_flow_rgb_image(self):
    waypoint_1 = self.true_waypoints.get_flow_at_waypoint(0)
    flow_rgb = occupancy_flow_vis.flow_rgb_image(
        flow=waypoint_1.vehicles,
        roadgraph_image=self.true_vis_grids.roadgraph,
        agent_trails=self.true_vis_grids.agent_trails,
    )

    batch_size = self.batch_size
    height = self.config.grid_height_cells
    width = self.config.grid_width_cells

    self.assertEqual(flow_rgb.shape, (batch_size, height, width, 3))


if __name__ == '__main__':
  tf.test.main()
