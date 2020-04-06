# Lint as: python3
# Copyright 2020 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""A simple example to generate a file that contains serialized Objects proto."""

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


def _create_pd_file_example():
  """Creates a prediction objects file."""
  objects = metrics_pb2.Objects()

  o = metrics_pb2.Object()
  # The following 3 fields are used to uniquely identify a frame a prediction
  # is predicted at. Make sure you set them to values exactly the same as what
  # we provided in the raw data. Otherwise your prediction is considered as a
  # false negative.
  o.context_name = ('context_name for the prediction. See Frame::context::name '
                    'in  dataset.proto.')
  # The frame timestamp for the prediction. See Frame::timestamp_micros in
  # dataset.proto.
  invalid_ts = -1
  o.frame_timestamp_micros = invalid_ts
  # This is only needed for 2D detection or tracking tasks.
  # Set it to the camera name the prediction is for.
  o.camera_name = dataset_pb2.CameraName.FRONT

  # Populating box and score.
  box = label_pb2.Label.Box()
  box.center_x = 0
  box.center_y = 0
  box.center_z = 0
  box.length = 0
  box.width = 0
  box.height = 0
  box.heading = 0
  o.object.box.CopyFrom(box)
  # This must be within [0.0, 1.0]. It is better to filter those boxes with
  # small scores to speed up metrics computation.
  o.score = 0.5
  # For tracking, this must be set and it must be unique for each tracked
  # sequence.
  o.object.id = 'unique object tracking ID'
  # Use correct type.
  o.object.type = label_pb2.Label.TYPE_PEDESTRIAN

  objects.objects.append(o)

  # Add more objects. Note that a reasonable detector should limit its maximum
  # number of boxes predicted per frame. A reasonable value is around 400. A
  # huge number of boxes can slow down metrics computation.

  # Write objects to a file.
  f = open('/tmp/your_preds.bin', 'wb')
  f.write(objects.SerializeToString())
  f.close()


def main():
  _create_pd_file_example()


if __name__ == '__main__':
  main()
