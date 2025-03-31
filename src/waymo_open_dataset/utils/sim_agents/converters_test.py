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
"""Unit tests for waymo_open_dataset.utils.sim_agents.converters."""

from absl.testing import parameterized
import tensorflow as tf

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils.sim_agents import converters
from waymo_open_dataset.utils.sim_agents import submission_specs


class ConvertersTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (submission_specs.ChallengeType.SIM_AGENTS, 80),
      (submission_specs.ChallengeType.SCENARIO_GEN, 91),
  )
  def test_scenario_to_joint_scene(
      self,
      challenge_type: submission_specs.ChallengeType,
      expected_num_steps: int,
  ):
    scenario = test_utils.get_womd_test_scenario()
    joint_scene = converters.scenario_to_joint_scene(scenario, challenge_type)
    # The test scenario contains 50 objects that are valid at step 10.
    self.assertLen(joint_scene.simulated_trajectories, 50)

    self.assertLen(
        joint_scene.simulated_trajectories[0].center_x, expected_num_steps
    )
    self.assertLen(
        joint_scene.simulated_trajectories[0].length, expected_num_steps
    )
    self.assertLen(
        joint_scene.simulated_trajectories[0].valid, expected_num_steps
    )
    # Expects the object type to be `TYPE_VEHICLE`.
    self.assertEqual(
        joint_scene.simulated_trajectories[0].object_type,
        scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
    )
    # Validate the joint scene.
    submission_specs.validate_joint_scene(joint_scene, scenario, challenge_type)

  @parameterized.named_parameters(
      {'testcase_name': 'all_valid', 'use_log_validity': False},
      {'testcase_name': 'log_valid', 'use_log_validity': True},
  )
  def test_joint_scene_to_trajectories(self, use_log_validity: bool):
    scenario = test_utils.get_womd_test_scenario()
    challenge_type = submission_specs.ChallengeType.SIM_AGENTS
    joint_scenes = converters.scenario_to_joint_scene(scenario, challenge_type)
    trajectories = converters.joint_scene_to_trajectories(
        joint_scenes,
        scenario,
        use_log_validity=use_log_validity,
    )
    # Check shapes of the time-series fields. The test scenario contains a
    # total of 50 sim agents.
    time_fields = [
        'x',
        'y',
        'z',
        'heading',
        'valid',
        'length',
        'width',
        'height',
    ]
    for field in time_fields:
      self.assertEqual(
          getattr(trajectories, field).shape,
          (50, 91),
          msg=f'field name: {field}',
      )
    # Check shapes of the per-object fields.
    self.assertEqual(trajectories.object_id.shape, (50,))
    self.assertEqual(trajectories.object_type.shape, (50,))
    config = submission_specs.get_submission_config(challenge_type)
    # If `use_log_validity=True`, we want to check that the invalid objects
    # in the test scenario are propagated as invalid. Otherwise, if
    # `use_log_validity=False` we want to verify that all the objects are
    # assigned a valid state.
    self.assertEqual(
        tf.reduce_all(trajectories.valid[:, config.current_time_index :]),
        not use_log_validity,
    )

  def test_simulated_scenariogen_to_trajectories(self):
    scenario = test_utils.get_womd_test_scenario()
    challenge_type = submission_specs.ChallengeType.SCENARIO_GEN
    joint_scenes = converters.scenario_to_joint_scene(scenario, challenge_type)
    trajectories = converters.simulated_scenariogen_to_trajectories(
        joint_scenes,
        scenario,
        challenge_type,
    )
    # Check shapes of the time-series fields. The test scenario contains a
    # total of 50 sim agents.
    time_fields = [
        'x',
        'y',
        'z',
        'heading',
        'valid',
        'length',
        'width',
        'height',
    ]
    # Only contains what is provided in the joint scene. Thus the shape is
    # (50, 91)
    for field in time_fields:
      self.assertEqual(
          getattr(trajectories, field).shape,
          (50, 91),
          msg=f'field name: {field}',
      )
    # Check shapes of the per-object fields.
    self.assertEqual(trajectories.object_id.shape, (50,))
    self.assertEqual(trajectories.object_type.shape, (50,))


if __name__ == '__main__':
  tf.random.set_seed(42)
  tf.test.main()
