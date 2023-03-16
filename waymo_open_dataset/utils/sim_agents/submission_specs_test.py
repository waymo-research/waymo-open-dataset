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
"""Tests for waymo_open_dataset.utils.sim_agents.submission_specs."""

import tensorflow as tf

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils


class SubmissionSpecsTest(tf.test.TestCase):

  def test_is_valid_sim_agent(self):
    scenario = test_utils.get_womd_test_scenario()
    # Track 0 is a valid sim agent because it's valid at step 11 (1-indexed).
    valid_track = scenario.tracks[0]
    self.assertTrue(submission_specs.is_valid_sim_agent(valid_track))
    # Track 64 is an invalid sim agent because it's invalid at step 11.
    invalid_track = scenario.tracks[64]
    self.assertFalse(submission_specs.is_valid_sim_agent(invalid_track))
    valid_mask = [
        submission_specs.is_valid_sim_agent(track) for track in scenario.tracks
        ]
    expected_n_objects = 83
    expected_n_sim_agents = 50
    self.assertLen(valid_mask, expected_n_objects)
    self.assertEqual(sum(valid_mask), expected_n_sim_agents)

  def test_get_sim_agent_ids(self):
    scenario = test_utils.get_womd_test_scenario()
    sim_agent_ids = submission_specs.get_sim_agent_ids(scenario)
    expected_n_sim_agents = 50
    self.assertLen(sim_agent_ids, expected_n_sim_agents)

  def test_get_evaluation_sim_agent_ids(self):
    scenario = test_utils.get_womd_test_scenario()
    eval_sim_agent_ids = submission_specs.get_evaluation_sim_agent_ids(
        scenario)
    expected_n_eval_sim_agents = 4
    self.assertLen(eval_sim_agent_ids, expected_n_eval_sim_agents)

  def test_get_evaluation_sim_agent_ids_no_repetitions(self):
    scenario = test_utils.get_womd_test_scenario()
    # Let's modify the scenario such that the AV is inside `tracks_to_predict`
    # (as this happens inconsistently in the dataset).
    scenario.tracks_to_predict.append(scenario_pb2.RequiredPrediction(
        track_index=scenario.sdc_track_index))
    eval_sim_agent_ids = submission_specs.get_evaluation_sim_agent_ids(
        scenario)
    self.assertLen(eval_sim_agent_ids, len(set(eval_sim_agent_ids)))

  def test_validate_joint_scene_success(self):
    scenario = test_utils.get_womd_test_scenario()
    # Create a joint scene with the correct objects and the correct length.
    trajectories = sim_agents_test_utils.get_test_simulated_trajectories(
        scenario, num_sim_steps=submission_specs.N_SIMULATION_STEPS)
    joint_scene = sim_agents_submission_pb2.JointScene(
        simulated_trajectories=trajectories)
    submission_specs.validate_joint_scene(joint_scene, scenario)

  def test_validate_joint_scene_fails_on_wrong_length(self):
    scenario = test_utils.get_womd_test_scenario()
    # Create a joint scene with the correct objects and the wrong length.
    trajectories = sim_agents_test_utils.get_test_simulated_trajectories(
        scenario, num_sim_steps=42)
    joint_scene = sim_agents_submission_pb2.JointScene(
        simulated_trajectories=trajectories)
    with self.assertRaises(ValueError):
      submission_specs.validate_joint_scene(joint_scene, scenario)

  def test_validate_joint_scene_fails_on_missing_objects(self):
    scenario = test_utils.get_womd_test_scenario()
    # Create a joint scene with the correct length but one missing object.
    trajectories = sim_agents_test_utils.get_test_simulated_trajectories(
        scenario, num_sim_steps=submission_specs.N_SIMULATION_STEPS)
    joint_scene = sim_agents_submission_pb2.JointScene(
        simulated_trajectories=trajectories[:-1])
    with self.assertRaises(ValueError):
      submission_specs.validate_joint_scene(joint_scene, scenario)

  def test_validate_joint_scene_fails_on_wrong_objects(self):
    scenario = test_utils.get_womd_test_scenario()
    # Create a joint scene with the correct length but one additional object.
    trajectories = sim_agents_test_utils.get_test_simulated_trajectories(
        scenario, num_sim_steps=submission_specs.N_SIMULATION_STEPS)
    trajectories.append(
        sim_agents_submission_pb2.SimulatedTrajectory(
            center_x=[0.0] * submission_specs.N_SIMULATION_STEPS,
            center_y=[0.0] * submission_specs.N_SIMULATION_STEPS,
            center_z=[0.0] * submission_specs.N_SIMULATION_STEPS,
            heading=[0.0] * submission_specs.N_SIMULATION_STEPS,
            object_id=0))
    joint_scene = sim_agents_submission_pb2.JointScene(
        simulated_trajectories=trajectories)
    with self.assertRaises(ValueError):
      submission_specs.validate_joint_scene(joint_scene, scenario)

  def test_validate_scenario_rollouts_correct_n_simulations(self):
    scenario = test_utils.get_womd_test_scenario()
    # Create a joint scene with the correct objects and the correct length.
    trajectories = sim_agents_test_utils.get_test_simulated_trajectories(
        scenario, num_sim_steps=submission_specs.N_SIMULATION_STEPS)
    joint_scene = sim_agents_submission_pb2.JointScene(
        simulated_trajectories=trajectories)
    scenario_rollouts = sim_agents_submission_pb2.ScenarioRollouts(
        joint_scenes=[joint_scene]*submission_specs.N_ROLLOUTS,
        scenario_id=scenario.scenario_id)
    submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)

  def test_validate_scenario_rollouts_missing_id(self):
    scenario = test_utils.get_womd_test_scenario()
    # Create a joint scene with the correct objects and the correct length.
    trajectories = sim_agents_test_utils.get_test_simulated_trajectories(
        scenario, num_sim_steps=submission_specs.N_SIMULATION_STEPS)
    joint_scene = sim_agents_submission_pb2.JointScene(
        simulated_trajectories=trajectories)
    scenario_rollouts = sim_agents_submission_pb2.ScenarioRollouts(
        joint_scenes=[joint_scene]*submission_specs.N_ROLLOUTS)
    with self.assertRaises(ValueError):
      submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)

  def test_validate_scenario_rollouts_wrong_n_simulations(self):
    scenario = test_utils.get_womd_test_scenario()
    # Create a joint scene with the correct objects and the correct length.
    trajectories = sim_agents_test_utils.get_test_simulated_trajectories(
        scenario, num_sim_steps=submission_specs.N_SIMULATION_STEPS)
    joint_scene = sim_agents_submission_pb2.JointScene(
        simulated_trajectories=trajectories)
    scenario_rollouts = sim_agents_submission_pb2.ScenarioRollouts(
        joint_scenes=[joint_scene]*24, scenario_id=scenario.scenario_id)
    with self.assertRaises(ValueError):
      submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)

  def test_validate_scenario_with_test_submission(self):
    scenario = test_utils.get_womd_test_scenario()
    test_submission = sim_agents_test_utils.load_test_submission()
    scenario_rollouts = test_submission.scenario_rollouts[0]
    submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)


if __name__ == '__main__':
  tf.random.set_seed(42)
  tf.test.main()
