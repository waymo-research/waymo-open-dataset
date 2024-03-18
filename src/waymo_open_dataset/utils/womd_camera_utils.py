# Copyright 2024 The Waymo Open Dataset Authors.
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
"""Waymo Motion Open Dataset (WOMD) utils to process camera data."""

import numpy as np

from waymo_open_dataset.protos import scenario_pb2


def add_camera_tokens_to_scenario(
    scenario: scenario_pb2.Scenario, camera_data: scenario_pb2.Scenario
) -> scenario_pb2.Scenario:
  """Augments the scenario with camera tokens for the first 1 second.

  Args:
    scenario: the WOMD scenario proto containing motion data.
    camera_data: A WOMD scenario proto which only contains a non-empty
      `frame_camera_tokens` field. This field is merged into the original WOMD
      scenario's `frame_camera_tokens`.

  Returns:
    scenario_augmented: the augmented WOMD scenario proto.
  """

  scenario_augmented = scenario_pb2.Scenario()
  scenario_augmented.CopyFrom(scenario)
  scenario_augmented.frame_camera_tokens.extend(camera_data.frame_camera_tokens)
  return scenario_augmented


def get_camera_embedding_from_codebook(
    codebook: np.ndarray, input_tokens: np.ndarray
) -> np.ndarray:
  """Gets the camera embedding from tokens.

  Args:
    codebook: A 2D numpy array with shape [#embeddings (8192), #features (32)]
      from a pre-trained VQ-GAN model storing all embedding vectors.
    input_tokens: A 1D numpy array. Each element is an integer referring to the
      index of the embedding in the codebook.

  Returns:
    A 2D numpy array with embedding vectors for each token in the input_tokens.
  """
  max_index = codebook.shape[0] - 1
  if np.min(input_tokens) < 0 or np.max(input_tokens) > max_index:
    raise ValueError(f'Input tokens must be in the range [0, {max_index}].')
  return codebook[input_tokens]
