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
# ==============================================================================

from absl.testing import absltest
import numpy as np

from waymo_open_dataset.utils.compression import delta_encoder


class DeltaEncoderTest(absltest.TestCase):

  def setUp(self):
    """Sets up parameters."""
    super().setUp()
    self._test_num_rows = 64
    self._test_num_cols = 2650

  def test_compress_decompress_single_prec(self):
    test_num_channels = 8
    ri_range = np.random.uniform(
        low=0.0,
        high=75.0,
        size=(self._test_num_rows, self._test_num_cols, test_num_channels),
    )
    ri_range_round = np.round(ri_range, 1)
    compressed_ri_range = delta_encoder.compress(ri_range, quant_precision=0.1)
    decoded_ri_range = delta_encoder.decompress(compressed_ri_range)
    np.testing.assert_allclose(decoded_ri_range, ri_range_round, atol=1e-5)

  def test_compress_decompress_multiple_precs(self):
    test_num_channels = 3
    quant_precision = [0.1, 0.01, 0.001]
    ri_range = np.random.uniform(
        low=0.0,
        high=75.0,
        size=(self._test_num_rows, self._test_num_cols, test_num_channels),
    )
    ri_range_round = ri_range.copy()
    ri_range_round[..., 0] = np.round(ri_range_round[..., 0], 1)
    ri_range_round[..., 1] = np.round(ri_range_round[..., 1], 2)
    ri_range_round[..., 2] = np.round(ri_range_round[..., 2], 3)
    compressed_ri_range = delta_encoder.compress(
        ri_range, quant_precision=quant_precision
    )
    decoded_ri_range = delta_encoder.decompress(compressed_ri_range)
    np.testing.assert_allclose(decoded_ri_range, ri_range_round, atol=1e-5)

  def test_compress_decompress_negative(self):
    test_num_channels = 1
    ri_range = np.random.uniform(
        low=-10.0,
        high=10.0,
        size=(self._test_num_rows, self._test_num_cols, test_num_channels),
    )
    ri_range_round = np.round(ri_range, 1)
    compressed_ri_range = delta_encoder.compress(ri_range, quant_precision=0.1)
    decoded_ri_range = delta_encoder.decompress(compressed_ri_range)
    np.testing.assert_allclose(decoded_ri_range, ri_range_round, atol=1e-5)

  def test_compress_decompress_zeros(self):
    test_num_channels = 3
    ri_range = np.zeros(
        (self._test_num_rows, self._test_num_cols, test_num_channels)
    )
    ri_range_round = np.round(ri_range, 1)
    compressed_ri_range = delta_encoder.compress(ri_range, quant_precision=0.1)
    decoded_ri_range = delta_encoder.decompress(compressed_ri_range)
    np.testing.assert_allclose(decoded_ri_range, ri_range_round, atol=1e-5)

  def test_run_len_encode_start_zero(self):
    # We only test 0s and 1s, since the rle will only be used for encoding mask
    input_array = [0.0, 1.0, 0.0, 1.0, 1.0]
    rle_encoded = delta_encoder._rlencode(input_array)
    expected_encoded = np.array([0, 1, 1, 1, 2])
    np.testing.assert_equal(rle_encoded, expected_encoded)
    rle_decoded = delta_encoder._rldecode(rle_encoded)
    np.testing.assert_allclose(input_array, rle_decoded, atol=1e-5)

  def test_run_len_encode_start_one(self):
    input_array = [1, 0, 0, 1, 0, 1, 1]
    rle_encoded = delta_encoder._rlencode(input_array)
    expected_encoded = np.array([1, 2, 1, 1, 2])
    np.testing.assert_equal(rle_encoded, expected_encoded)
    rle_decoded = delta_encoder._rldecode(rle_encoded)
    np.testing.assert_allclose(input_array, rle_decoded, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
