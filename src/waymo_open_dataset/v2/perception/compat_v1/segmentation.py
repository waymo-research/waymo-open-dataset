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
"""Tools to convert segmentation labels v2 to and from v1 Frame protos."""
from typing import Iterable

from waymo_open_dataset.v2.perception import base as _v2_base
from waymo_open_dataset.v2.perception import segmentation as _v2_segmentation
from waymo_open_dataset.v2.perception.compat_v1 import interfaces
from waymo_open_dataset.v2.perception.utils import lidar_utils


class CameraSegmentationLabelComponentExtractor(
    interfaces.CameraImageComponentExtractor
):
  """Extracts camera segmentation label component of a given camera image."""

  def __call__(
      self, src: interfaces.CameraImageComponentSrc
  ) -> Iterable[_v2_segmentation.CameraSegmentationLabelComponent]:
    if not src.camera_image.HasField('camera_segmentation_label'):
      return None
    key = _v2_base.CameraKey(
        segment_context_name=src.frame.context.name,
        frame_timestamp_micros=src.frame.timestamp_micros,
        camera_name=src.camera_image.name,
    )
    label_buffer = src.camera_image.camera_segmentation_label
    local_instance_ids, global_instance_ids, is_tracked = [], [], []
    for mapping in label_buffer.instance_id_to_global_id_mapping:
      local_instance_ids.append(mapping.local_instance_id)
      global_instance_ids.append(mapping.global_instance_id)
      is_tracked.append(mapping.is_tracked)
    yield _v2_segmentation.CameraSegmentationLabelComponent(
        key=key,
        panoptic_label_divisor=label_buffer.panoptic_label_divisor,
        panoptic_label=label_buffer.panoptic_label,
        instance_id_to_global_id_mapping=(
            _v2_segmentation.InstanceIdToGlobalIdMapping(
                local_instance_ids=local_instance_ids,
                global_instance_ids=global_instance_ids,
                is_tracked=is_tracked,
            )
        ),
        sequence_id=label_buffer.sequence_id,
        num_cameras_covered=label_buffer.num_cameras_covered,
    )


class LiDARSegmentationLabelComponentExtractor(
    interfaces.LiDARComponentExtractor
):
  """Extracts 3D segmentation label component of a given frame."""

  def __call__(
      self, src: interfaces.LiDARComponentSrc
  ) -> Iterable[_v2_segmentation.LiDARSegmentationLabelComponent]:
    range_image_return1 = lidar_utils.parse_range_image(
        src.laser.ri_return1.segmentation_label_compressed,
        _v2_segmentation.LiDARSegmentationRangeImage,
    )
    range_image_return2 = lidar_utils.parse_range_image(
        src.laser.ri_return2.segmentation_label_compressed,
        _v2_segmentation.LiDARSegmentationRangeImage,
    )
    if not (range_image_return1 is None and range_image_return2 is None):
      yield _v2_segmentation.LiDARSegmentationLabelComponent(
          key=lidar_utils.get_laser_key(src),
          range_image_return1=range_image_return1,
          range_image_return2=range_image_return2,
      )
