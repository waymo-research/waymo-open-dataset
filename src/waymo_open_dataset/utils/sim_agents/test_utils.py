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
"""Common utils for Sim Agents related tests."""

from google.protobuf import text_format

# copybara removed file resource import

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs


def get_test_simulated_trajectories(
    scenario: scenario_pb2.Scenario, num_sim_steps: int = 80):
  """Generates zero-valued trajectories of the expected sim duration."""
  trajectories = []
  for track in scenario.tracks:
    if submission_specs.is_valid_sim_agent(track):
      trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
          center_x=[0.0] * num_sim_steps, center_y=[0.0] * num_sim_steps,
          center_z=[0.0] * num_sim_steps, heading=[0.0] * num_sim_steps,
          object_id=track.id
      ))
  return trajectories


def load_test_submission(
) -> sim_agents_submission_pb2.SimAgentsChallengeSubmission:
  """Loads the test submission binproto inside the testdata directory.

  This submission was generated starting from the same Scenario included in the
  utils/testdata directory (ID: 637f20cafde22ff8) simulated with a linear
  extrapolation (constant speed) policy for all the agents. There are a total
  of 32 parallel simulations inside this proto.

  This single `ScenarioRollouts` is packaged into a
  `SimAgentsChallengeSubmission` proto with test values for the other fields.

  Returns:
    A `SimAgentsChallengeSubmission` extracted from the binproto file inside
    the testdata directory.
  """
  # pylint: disable=line-too-long
  # pyformat: disable
  test_data_path = '{pyglib_resource}waymo_open_dataset/utils/sim_agents/testdata/test_submission.binproto'.format(pyglib_resource='')
  with open(test_data_path, 'rb') as f:
    return sim_agents_submission_pb2.SimAgentsChallengeSubmission.FromString(
        f.read())


def load_test_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
  """Loads a test `SimAgentMetricsConfig` inside the testdata directory."""
  # pylint: disable=line-too-long
  # pyformat: disable
  test_config_path = '{pyglib_resource}waymo_open_dataset/utils/sim_agents/testdata/test_config.textproto'.format(pyglib_resource='')
  with open(test_config_path, 'r') as f:
    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    text_format.Parse(f.read(), config)
  return config


def load_identity_function_test_metrics_config() -> (
    sim_agents_metrics_pb2.SimAgentMetricsConfig
):
  """Loads config producing likelihoods of 1 for logged v. logged comparison."""
  # pylint: disable=line-too-long
  # pyformat: disable
  test_config_path = '{pyglib_resource}waymo_open_dataset/utils/sim_agents/testdata/test_config_dependent_timesteps.textproto'.format(pyglib_resource='')
  with open(test_config_path, 'r') as f:
    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    text_format.Parse(f.read(), config)
  return config
