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
"""Tests for waymo_open_dataset/v2/perception/compat_v1/camera_image.py."""

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

# copybara removed file resource import
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.v2.perception import camera_image as _v2_camera_image
from waymo_open_dataset.v2.perception.compat_v1 import camera_image
from waymo_open_dataset.v2.perception.compat_v1 import interfaces

# pyformat: disable
_TEST_DATA_PATH = '{pyglib_resource}waymo_open_dataset/v2/perception/compat_v1/testdata/two_frame.tfrecord'.format(pyglib_resource='')  # pylint: disable=line-too-long
# pyformat: enable


def _get_camera_image_by_name(
    frame: dataset_pb2.Frame,
    camera_name: _v2_camera_image.CameraName,
) -> dataset_pb2.CameraImage:
  for image in frame.images:
    if image.name == camera_name.value:
      return image
  raise ValueError(f'{camera_name} not found.')


class CameraImageTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.frames = self._load_segment_data()

  def _load_segment_data(self):
    dataset = tf.data.TFRecordDataset(_TEST_DATA_PATH)
    dataset_iter = dataset.as_numpy_iterator()
    frames = []
    for _ in range(2):
      frames.append(dataset_pb2.Frame.FromString(next(dataset_iter)))
    return frames

  @parameterized.named_parameters(
      ('front_camera', _v2_camera_image.CameraName.FRONT),
      ('front_left_camera', _v2_camera_image.CameraName.FRONT_LEFT),
      ('front_right_camera', _v2_camera_image.CameraName.FRONT_RIGHT),
      ('side_left_camera', _v2_camera_image.CameraName.SIDE_LEFT),
      ('side_right_camera', _v2_camera_image.CameraName.SIDE_RIGHT),
  )
  def test_populates_camera_image(self, camera_name):
    frame = self.frames[0]
    src = interfaces.CameraImageComponentSrc(
        frame=frame,
        camera_image=_get_camera_image_by_name(frame, camera_name),
    )
    extractor = camera_image.CameraImageComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.frame_timestamp_micros, 1543251275738460)
    self.assertEqual(component.key.camera_name, camera_name.value)
    self.assertNotEmpty(component.image, 0)
    self.assertLen(component.pose.transform, 16)


if __name__ == '__main__':
  absltest.main()
