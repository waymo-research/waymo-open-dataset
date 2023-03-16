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
"""Tests for waymo_open_dataset.utils.sim_agents.converters."""

import tensorflow as tf

from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils.sim_agents import converters
from waymo_open_dataset.utils.sim_agents import submission_specs


class ConvertersTest(tf.test.TestCase):

  def test_scenario_to_joint_scene(self):
    scenario = test_utils.get_womd_test_scenario()
    joint_scene = converters.scenario_to_joint_scene(scenario)
    # The test scenario contains 50 objects that are valid at step 10.
    self.assertLen(joint_scene.simulated_trajectories, 50)
    # Check the simulated trajectory length is exactly 80, i.e. 8 seconds at
    # 10Hz.
    self.assertLen(joint_scene.simulated_trajectories[0].center_x, 80)
    # Validate the joint scene.
    submission_specs.validate_joint_scene(joint_scene, scenario)

  def test_joint_scene_to_trajectories(self):
    scenario = test_utils.get_womd_test_scenario()
    joints_scene = converters.scenario_to_joint_scene(scenario)
    trajectories = converters.joint_scene_to_trajectories(
        joints_scene, scenario)
    # Check shapes of the time-series fields. The test scenario contains a
    # total of 50 sim agents.
    time_fields = [
        'x', 'y', 'z', 'heading', 'valid', 'length', 'width', 'height'
    ]
    for field in time_fields:
      self.assertEqual(getattr(trajectories, field).shape, (50, 91))
    # Check shapes of the per-object fields.
    self.assertEqual(trajectories.object_id.shape, (50,))
    self.assertEqual(trajectories.object_type.shape, (50,))


if __name__ == '__main__':
  tf.random.set_seed(42)
  tf.test.main()
