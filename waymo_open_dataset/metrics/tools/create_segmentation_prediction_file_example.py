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
"""A simple example to generate a file that contains a serialized SegmentationFrameList proto."""

import zlib
import waymo_open_dataset.dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2

NUM_CONTEXT = 2
NUM_FRAMES = 3


def _create_single_frame_seg_pd_file_example():
  """Create a dummy prediction file."""
  submission = segmentation_submission_pb2.SemanticSegmentationSubmission()
  frames = segmentation_metrics_pb2.SegmentationFrameList()
  for c in range(NUM_CONTEXT):
    for f in range(NUM_FRAMES):
      frame = segmentation_metrics_pb2.SegmentationFrame()
      segmentation_label = open_dataset.Laser()
      segmentation_label.name = open_dataset.LaserName.TOP
      # Add prediction for the first return image.
      pd = open_dataset.MatrixInt32()
      for _ in range(50):
        for _ in range(1000):
          pd.data.append(segmentation_pb2.Segmentation.TYPE_CAR)
      pd.shape.dims.append(50)
      pd.shape.dims.append(1000)
      pd_str = zlib.compress(pd.SerializeToString())
      segmentation_label.ri_return1.segmentation_label_compressed = pd_str
      # Add prediction for the second return image.
      pd = open_dataset.MatrixInt32()
      for _ in range(50):
        for _ in range(1000):
          pd.data.append(segmentation_pb2.Segmentation.TYPE_PEDESTRIAN)
      pd.shape.dims.append(50)
      pd.shape.dims.append(1000)
      pd_str = zlib.compress(pd.SerializeToString())
      segmentation_label.ri_return2.segmentation_label_compressed = pd_str
      frame.segmentation_labels.append(segmentation_label)
      frame.context_name = f"dummy_context_name_{c}"
      frame.frame_timestamp_micros = f * 1000000
      frames.frames.append(frame)
  submission.account_name = "joe@gmail.com"
  submission.unique_method_name = "JoeNet"
  submission.authors.append("Joe Smith")
  submission.authors.append("Joe Smith")
  submission.affiliation = "Smith Inc."
  submission.description = "A dummy method by Joe."
  submission.method_link = "NA"
  submission.sensor_type = segmentation_submission_pb2.SemanticSegmentationSubmission.LIDAR_ALL
  submission.number_past_frames_exclude_current = 2
  submission.number_future_frames_exclude_current = 0
  submission.inference_results.CopyFrom(frames)
  # Write frames to a file.
  f = open("fake_segmentation_predictions.bin", "wb")
  f.write(frames.SerializeToString())
  f.close()
  # Write submissions to a file.
  f = open("fake_segmentation_submissions.bin", "wb")
  f.write(submission.SerializeToString())
  f.close()


def main():
  _create_single_frame_seg_pd_file_example()


if __name__ == "__main__":
  main()
