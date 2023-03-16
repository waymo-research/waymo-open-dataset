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
"""Segmentation label components."""


import dataclasses

import pyarrow as pa
import tensorflow as tf

from waymo_open_dataset.v2 import component
from waymo_open_dataset.v2.perception import base


_column = component.create_column


@dataclasses.dataclass(frozen=True)
class InstanceIdToGlobalIdMapping:
  """Local instance ID to global unique ID mapping.

  A mapping between each panoptic label with an instance_id and a globally
  unique ID across all frames within the same sequence. This can be used to
  match instances across cameras and over time. i.e. instances belonging to the
  same object will map to the same global ID across all frames in the same
  sequence.
  NOTE: These unique IDs are not consistent with other IDs in the dataset, e.g.
  the bounding box IDs.

  Attributes:
    local_instance_ids: A list of local instance IDs. They may not be unique
      across all frames within the same sequence.
    global_instance_ids: A list of globally unique instance IDs.
    is_tracked: If false, the corresponding instance will not have consistent
      global IDs between frames.
  """
  local_instance_ids: list[int] = _column(arrow_type=pa.list_(pa.int32()))
  global_instance_ids: list[int] = _column(arrow_type=pa.list_(pa.int32()))
  is_tracked: list[bool] = _column(arrow_type=pa.list_(pa.bool_()))


@dataclasses.dataclass(frozen=True)
class CameraSegmentationLabelComponent(base.CameraComponent):
  """Panoptic (instance + semantic) segmentation labels for a camera image.

  Associations can also be provided between each instance ID and a globally
  unique ID across all frames.

  Attributes:
    panoptic_label_divisor: The value used to separate instance_ids from
      different semantic classes. See the panoptic_label field for how this is
      used. Must be set to be greater than the maximum instance_id.
    panoptic_label: A uint16 png encoded image, with the same resolution as the
      corresponding camera image. Each pixel contains a panoptic segmentation
      label, which is computed as:
        semantic_class_id * panoptic_label_divisor + instance_id.
      We set instance_id = 0 for pixels for which there is no instance_id.
      NOTE: Instance IDs in this label are only consistent within this camera
      image. Use instance_id_to_global_id_mapping to get cross-camera consistent
      instance IDs.
    instance_id_to_global_id_mapping: A mapping between each panoptic label with
      an instance_id and a globally unique ID across all frames within the same
      sequence. This can be used to match instances across cameras and over
      time. i.e. instances belonging to the same object will map to the same
      global ID across all frames in the same sequence.
      NOTE: These unique IDs are not consistent with other IDs in the dataset,
      e.g. the bounding box IDs.
    sequence_id: The sequence id for this label. The above
      instance_id_to_global_id_mapping is only valid with other labels with the
      same sequence id.
    num_cameras_covered: A uint8 png encoded image, with the same resolution as
      the corresponding camera image. The value on each pixel indicates the
      number of cameras that overlap with this pixel. Used for the weighted
      Segmentation and Tracking Quality (wSTQ) metric.
  """
  panoptic_label_divisor: int = _column(arrow_type=pa.int32())
  panoptic_label: bytes = _column(arrow_type=pa.binary())
  instance_id_to_global_id_mapping: InstanceIdToGlobalIdMapping = _column()
  sequence_id: str = _column(arrow_type=pa.string())
  num_cameras_covered: bytes = _column(arrow_type=pa.binary())


@dataclasses.dataclass(frozen=True)
class LiDARSegmentationRangeImage:
  """A [H, W, 2] tensor containing the 3D segmentation label of each pixel.

  Inner dimensions are [instance_id, semantic_class].

  NOTE:
  1. Only the top LiDAR has segmentation labels.
  2. Not every frame has segmentation labels.
  3. There can be points missing segmentation labels within a labeled frame.
     Their labels are set to TYPE_NOT_LABELED when that happens.
  """
  values: list[int] = _column(arrow_type=pa.list_(pa.int32()))
  shape: list[int] = _column(arrow_type=pa.list_(pa.int32(), 3))

  @property
  def tensor(self) -> tf.Tensor:
    return tf.reshape(tf.convert_to_tensor(self.values), self.shape)


@dataclasses.dataclass(frozen=True)
class LiDARSegmentationLabelComponent(base.LaserComponent):
  """A dataset component for 3D segmentation labels."""
  range_image_return1: LiDARSegmentationRangeImage = _column()
  range_image_return2: LiDARSegmentationRangeImage = _column()

  @property
  def range_image_returns(self) -> list[LiDARSegmentationRangeImage]:
    return [self.range_image_return1, self.range_image_return2]
