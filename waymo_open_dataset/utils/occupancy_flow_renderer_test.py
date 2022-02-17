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
"""Tests for occupancy_flow_renderer."""

import numpy as np
import tensorflow as tf

from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_renderer
from waymo_open_dataset.utils import occupancy_flow_test_util


class OccupancyFlowRendererTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 8
    inputs = occupancy_flow_test_util.make_one_data_batch(
        batch_size=self.batch_size)
    self.inputs = occupancy_flow_data.add_sdc_fields(inputs)
    self.config = occupancy_flow_test_util.make_test_config()

  def test_render_occupancy_from_inputs(self):
    times = ['past', 'current', 'future']
    agent_grids = occupancy_flow_renderer.render_occupancy_from_inputs(
        inputs=self.inputs,
        times=times,
        config=self.config,
        include_observed=True,
        include_occluded=False,
    )

    batch_size = self.batch_size
    height = self.config.grid_height_cells
    width = self.config.grid_width_cells
    num_steps = occupancy_flow_renderer._get_num_steps_from_times(
        times=times, config=self.config)

    for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
      self.assertEqual(
          agent_grids.view(object_type).shape,
          (batch_size, height, width, num_steps))
      # The test tf example contains all object types, so:
      self.assertEqual(np.min(agent_grids.view(object_type).numpy()), 0.0)
      self.assertEqual(np.max(agent_grids.view(object_type).numpy()), 1.0)

  def test_render_flow_from_inputs(self):
    times = ['past', 'current', 'future']
    agent_grids = occupancy_flow_renderer.render_flow_from_inputs(
        inputs=self.inputs,
        times=times,
        config=self.config,
        include_observed=True,
        include_occluded=False,
    )

    batch_size = self.batch_size
    height = self.config.grid_height_cells
    width = self.config.grid_width_cells
    num_all_steps = occupancy_flow_renderer._get_num_steps_from_times(
        times=times, config=self.config)

    waypoint_size = self.config.num_future_steps // self.config.num_waypoints
    num_flow_steps = num_all_steps - waypoint_size

    for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
      self.assertEqual(
          agent_grids.view(object_type).shape,
          (batch_size, height, width, num_flow_steps, 2))

  def test_render_roadgraph_from_inputs(self):
    roadgraph = occupancy_flow_renderer.render_roadgraph_from_inputs(
        inputs=self.inputs, config=self.config)

    batch_size = self.batch_size
    height = self.config.grid_height_cells
    width = self.config.grid_width_cells

    self.assertEqual(roadgraph.shape, (batch_size, height, width, 1))


if __name__ == '__main__':
  tf.test.main()
