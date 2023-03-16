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
"""Python package for compressing range images using delta encoding.

Paper link: https://arxiv.org/abs/2206.01738
"""
from typing import Union
import zlib

import numpy as np

from waymo_open_dataset.protos import compressed_lidar_pb2


def compress(
    range_image: np.ndarray, quant_precision: Union[float, list[float]]
) -> bytes:
  """Compresses a range image using delta encoding.

  Args:
    range_image: A (num_rows, num_cols, num_channel) size numpy array.
    quant_precision: quantization scale for each channel, expect one or a list
      of positive numbers. If quant_precision is a single number, we will apply
      this quantization scale for every channel. If it is a list of quantization
      scales whose length equals to the number of channels, we will use
      different quantization scale for each channel as specified.

  Returns:
    compressed_bytes: Compressed bytes of range image.
  """
  num_channel = range_image.shape[-1]
  if isinstance(quant_precision, (int, float)):
    if quant_precision <= 0:
      raise ValueError(
          "invalid quantization precision detected. Expect one or a list of"
          " positive numbers"
      )
    quant_precision = [quant_precision] * num_channel
  else:
    if len(quant_precision) != num_channel:
      raise ValueError(
          "length of quant_precision should equal to the channel size"
      )
    for precision in quant_precision:
      if precision <= 0:
        raise ValueError(
            "invalid quantization precision detected. Expect one or a list of"
            " positive numbers"
        )

  range_image_round = range_image
  for i in range(num_channel):
    range_image_round[..., i] = np.round(
        range_image[..., i] / quant_precision[i]
    )
  range_image_round = range_image_round.astype(int)
  lidar_proto = compressed_lidar_pb2.DeltaEncodedData(
      residual=_compute_residual(range_image_round),
      mask=_compute_mask(range_image_round),
      metadata=_compute_metadata(quant_precision, range_image.shape),
  )
  return zlib.compress(lidar_proto.SerializeToString())


def decompress(compressed_range_image: bytes) -> np.ndarray:
  """Decodes a delta encoded range image.

  Args:
    compressed_range_image: Compressed bytes of range image.

  Returns:
    decoded_range_image: A decoded range image.
  """
  lidar_proto = compressed_lidar_pb2.DeltaEncodedData.FromString(
      zlib.decompress(compressed_range_image)
  )

  shape = np.array(lidar_proto.metadata.shape).astype(int)
  precision = np.array(lidar_proto.metadata.quant_precision).astype(float)
  decoded_mask = _decode_mask(np.array(lidar_proto.mask).astype(int))
  residuals = np.array(lidar_proto.residual).astype(int)
  decoded_residuals = np.cumsum(residuals)
  decoded_mask[decoded_mask > 0] = decoded_residuals
  decoded_range_image = np.transpose(
      np.reshape(decoded_mask, (shape[2], shape[0], shape[1])), [1, 2, 0]
  ).astype(float)
  for c in range(shape[2]):
    decoded_range_image[..., c] = decoded_range_image[..., c] * precision[c]
  return decoded_range_image


def _compute_metadata(
    quant_precision: list[float], shape: tuple[int, ...]
) -> compressed_lidar_pb2.Metadata:
  """Computes shape, mask len, quant precision as meta data bytes.

  Args:
    quant_precision: A list of quantization precisions of a range image.
    shape: The shape of range image.

  Returns:
    total_meta_info: A metadata proto message.
  """
  for dim in shape:
    if dim > (2**16 - 1):
      raise ValueError("invalid shape detected. Expect in range of uint16")
  return compressed_lidar_pb2.Metadata(
      shape=shape, quant_precision=quant_precision
  )


def _decode_mask(raw_mask: np.ndarray) -> np.ndarray:
  """Decode the mask from mask varints.

  Args:
    raw_mask: An encoded mask array.

  Returns:
    decoded_mask: A decoded mask array.
  """
  return _rldecode(raw_mask)


def _rlencode(x: np.ndarray) -> np.ndarray:
  """Encodes an array using run-length encoding.

  The run length encoding method works only for bool values (e.g., mask values)
  because it stores only length and not actual values.

  Args:
    x: An array to encode.

  Returns:
    lengths: A run length encoded array.
  """
  x = np.asarray(x)
  # Find the starting index of each repeated/running slice
  starts = np.r_[0, np.where(np.not_equal(x[1:], x[:-1]))[0] + 1]
  # Run length equal to the diff between starting indices.
  lengths = np.diff(np.r_[starts, len(x)])
  # We implicitly set the rule that the first key in rle is 1,
  # thus if the leading entry in the mask array is 0, we need to
  # append 0 at the start of rle.
  if x[0] == 0:
    lengths = np.r_[0, lengths]
  return lengths


def _rldecode(lengths: np.ndarray) -> np.ndarray:
  """Decodes an array using run-length decoding.

  The run length decoding method works only for bool values (e.g., mask values)
  because it stores only length and not actual values.

  Args:
    lengths: An array to decode.

  Returns:
    x: A run length decoded array.
  """
  x = []
  cur = 1
  for l in lengths:
    x.extend([cur] * l)
    cur = 1 - cur
  return np.array(x)


def _compute_mask(range_image: np.ndarray) -> list[bool]:
  """Computes a run-length encoded mask array from a rounded range image.

  Args:
    range_image: A quantized range image.

  Returns:
    encoded_mask: An encoded mask which is a list of integers.
  """

  range_image = np.transpose(range_image, [2, 0, 1])
  range_image_mask = (range_image.reshape(-1) != 0).astype(np.uint32)
  encoded_mask = _rlencode(range_image_mask)
  return encoded_mask.tolist()


def _compute_residual(range_image_round: np.ndarray) -> list[int]:
  """Computes the residual from a rounded range image.

  Args:
    range_image_round: A quantized range image.

  Returns:
    residuals: A list of encoded residual integers.
  """
  # compute residuals w.r.t. left valid pixel
  range_image_round = np.transpose(range_image_round, [2, 0, 1])
  range_image_round_flat = range_image_round.reshape(-1)
  nonzero_entries = range_image_round_flat[range_image_round_flat != 0]
  nonzero_entries[1:] = nonzero_entries[1:] - nonzero_entries[:-1]
  residuals = nonzero_entries
  return residuals.tolist()
