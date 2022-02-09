# Lint as: python3
# Copyright 2022 The Waymo Open Dataset Authors. All Rights Reserved.
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
# ==============================================================================*/
"""A simple example to generate a file that contains serialized SegmentationFrameList proto."""

from waymo_open_dataset import label_pb2
import waymo_open_dataset.dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_metrics_pb2


def _create_single_frame_seg_pd_file_example():
  """Create a dummy prediction file."""
  frames = segmentation_metrics_pb2.SegmentationFrameList()
  frame = segmentation_metrics_pb2.SegmentationFrame()
  pd = open_dataset.MatrixInt32()
  for _ in range(50):
    for _ in range(1000):
      pd.data.append(label_pb2.Segmentation.TYPE_VEHICLE)
  pd.shape.dims.append(50)
  pd.shape.dims.append(1000)
  pd_str = pd.SerializeToString()
  frame.segmentation_labels.segmentation_label_compressed = pd_str
  frames.frames.append(frame)
  # Write frames to a file.
  f = open('fake_segmentation_predictions.bin', 'wb')
  f.write(frames.SerializeToString())
  f.close()


def main():
  _create_single_frame_seg_pd_file_example()


if __name__ == '__main__':
  main()
