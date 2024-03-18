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
"""Tests for waymo_open_dataset.utils.womd_camera_utils."""

import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import camera_tokens_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import womd_camera_utils


class WomdCameraUtilsTest(tf.test.TestCase):

  def test_augment_womd_scenario_with_camera_tokens(self):
    """Test of augment_womd_scenario_with_camera_tokens."""
    input_scenario = scenario_pb2.Scenario()
    camera_scenario = scenario_pb2.Scenario(
        frame_camera_tokens=[camera_tokens_pb2.FrameCameraTokens()] * 11
    )
    merged_scenario = (
        womd_camera_utils.add_camera_tokens_to_scenario(
            input_scenario, camera_scenario
        )
    )
    self.assertLen(merged_scenario.frame_camera_tokens, 11)

  def test_raises_exception_given_invalid_indices(self):
    """Test raise error given invalid indices for codebook."""
    codebook = np.array([[1.0, -2.0, 0.5], [1.2, 2.4, 3.6]], dtype=float)
    indices = np.array([-1, 3, 0])
    with self.assertRaisesRegex(ValueError, 'Input tokens must be'):
      womd_camera_utils.get_camera_embedding_from_codebook(codebook, indices)

  def test_get_camera_vqgan_embedding(self):
    """Test of get_camera_vqgan_embedding_from_codebook."""
    codebook = np.array([[1.0, -2.0, 0.5], [1.2, 2.4, 3.6]], dtype=float)
    indices = np.array([0, 1, 0])
    embedding = womd_camera_utils.get_camera_embedding_from_codebook(
        codebook, indices
    )
    expect_result = np.array(
        [[1.0, -2.0, 0.5], [1.2, 2.4, 3.6], [1.0, -2.0, 0.5]], dtype=float
    )
    self.assertAllClose(embedding, expect_result, rtol=1e-3)


if __name__ == '__main__':
  tf.test.main()
