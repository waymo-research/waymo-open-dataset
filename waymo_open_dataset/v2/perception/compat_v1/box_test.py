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
"""Tests for waymo_open_dataset/v2/perception/compat_v1/box.py."""
from absl.testing import absltest
import tensorflow as tf

# copybara removed file resource import
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.v2.perception.compat_v1 import box
from waymo_open_dataset.v2.perception.compat_v1 import interfaces


# pyformat: disable
_TEST_DATA_PATH = '{pyglib_resource}waymo_open_dataset/v2/perception/compat_v1/testdata/two_frame.tfrecord'.format(pyglib_resource='')  # pylint: disable=line-too-long
# pyformat: enable


class BoxTest(absltest.TestCase):

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

  def test_populates_lidar_box(self):
    frame = self.frames[0]
    src = interfaces.LiDARLabelComponentSrc(
        frame=frame,
        lidar_label=frame.laser_labels[0],
    )
    extractor = box.LiDARBoxComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.frame_timestamp_micros, 1543251275738460)
    self.assertEqual(component.key.laser_object_id, '-H7-eIvyHm5I5z_fgZ5IKg')
    self.assertIsNone(component.difficulty_level)
    self.assertEqual(component.type, 2)

  def test_populates_supports_missing_metadata(self):
    frame = self.frames[0]
    frame.laser_labels[0].ClearField('metadata')
    src = interfaces.LiDARLabelComponentSrc(
        frame=frame,
        lidar_label=frame.laser_labels[0],
    )
    extractor = box.LiDARBoxComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )

  def test_sets_missing_metadata_z_to_zero(self):
    frame = self.frames[0]
    frame.laser_labels[0].metadata.ClearField('speed_z')
    frame.laser_labels[0].metadata.ClearField('accel_z')
    src = interfaces.LiDARLabelComponentSrc(
        frame=frame,
        lidar_label=frame.laser_labels[0],
    )
    extractor = box.LiDARBoxComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )

  def test_yield_nothing_if_there_is_no_lidar_box(self):
    frame = self.frames[0]
    frame.laser_labels[0].ClearField('box')
    src = interfaces.LiDARLabelComponentSrc(
        frame=frame,
        lidar_label=frame.laser_labels[0],
    )
    extractor = box.LiDARBoxComponentExtractor()
    self.assertRaises(StopIteration, extractor(src).__next__)

  def test_populates_lidar_camera_synced_box(self):
    frame = self.frames[1]
    src = interfaces.LiDARLabelComponentSrc(
        frame=frame,
        lidar_label=frame.laser_labels[1],
    )
    extractor = box.LiDARCameraSyncedBoxComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.frame_timestamp_micros, 1543251275838479)
    self.assertEqual(component.key.laser_object_id, '-Xri4lKAIXSXuIDSOqor1A')
    self.assertEqual(component.most_visible_camera_name, 2)

  def test_yield_nothing_if_there_is_no_lidar_camera_synced_box(self):
    frame = self.frames[0]
    frame.laser_labels[0].ClearField('camera_synced_box')
    src = interfaces.LiDARLabelComponentSrc(
        frame=frame,
        lidar_label=frame.laser_labels[0],
    )
    extractor = box.LiDARCameraSyncedBoxComponentExtractor()
    self.assertRaises(StopIteration, extractor(src).__next__)

  def test_yield_nothing_if_most_visible_camera_name_is_empty(self):
    frame = self.frames[1]
    frame.laser_labels[0].most_visible_camera_name = ''
    src = interfaces.LiDARLabelComponentSrc(
        frame=frame,
        lidar_label=frame.laser_labels[0],
    )
    extractor = box.LiDARCameraSyncedBoxComponentExtractor()
    self.assertRaises(StopIteration, extractor(src).__next__)

  def test_populates_projected_lidar_box(self):
    frame = self.frames[1]
    src = interfaces.CameraLabelComponentSrc(
        frame=frame,
        camera_labels=frame.projected_lidar_labels[0],
        label=frame.projected_lidar_labels[0].labels[0],
    )
    extractor = box.ProjectedLiDARBoxComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.frame_timestamp_micros, 1543251275838479)
    self.assertEqual(component.key.camera_name, 1)
    self.assertEqual(component.key.laser_object_id, '-H7-eIvyHm5I5z_fgZ5IKg')
    self.assertEqual(component.type, 2)

  def test_yield_nothing_if_there_is_no_projected_lidar_box(self):
    frame = self.frames[0]
    frame.projected_lidar_labels[0].labels[0].ClearField('box')
    src = interfaces.CameraLabelComponentSrc(
        frame=frame,
        camera_labels=frame.projected_lidar_labels[0],
        label=frame.projected_lidar_labels[0].labels[0],
    )
    extractor = box.ProjectedLiDARBoxComponentExtractor()
    self.assertRaises(StopIteration, extractor(src).__next__)

  def test_populates_camera_box(self):
    frame = self.frames[1]
    src = interfaces.CameraLabelComponentSrc(
        frame=frame,
        camera_labels=frame.camera_labels[0],
        label=frame.camera_labels[0].labels[0],
    )
    extractor = box.CameraBoxComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.frame_timestamp_micros, 1543251275838479)
    self.assertEqual(component.key.camera_name, 1)
    self.assertEqual(
        component.key.camera_object_id, '0171c02b-9522-4d34-a223-860f9db52084'
    )
    self.assertEqual(component.type, 2)

  def test_yield_nothing_if_there_is_no_camera_box(self):
    frame = self.frames[0]
    frame.camera_labels[0].labels[0].ClearField('box')
    src = interfaces.CameraLabelComponentSrc(
        frame=frame,
        camera_labels=frame.camera_labels[0],
        label=frame.camera_labels[0].labels[0],
    )
    extractor = box.CameraBoxComponentExtractor()
    self.assertRaises(StopIteration, extractor(src).__next__)

  def test_populates_camera_to_lidar_box_association(self):
    frame = self.frames[1]
    src = interfaces.CameraLabelComponentSrc(
        frame=frame,
        camera_labels=frame.camera_labels[0],
        label=frame.camera_labels[0].labels[0],
    )
    extractor = box.CameraToLiDARBoxAssociationComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.frame_timestamp_micros, 1543251275838479)
    self.assertEqual(component.key.camera_name, 1)
    self.assertEqual(
        component.key.camera_object_id, '0171c02b-9522-4d34-a223-860f9db52084'
    )
    self.assertEqual(
        component.key.laser_object_id,
        'SBYvHrmX8kfxyTZ2ZPZYkw',
    )

  def test_yield_nothing_if_there_is_no_camera_to_lidar_association(self):
    frame = self.frames[0]
    frame.camera_labels[0].labels[0].ClearField('association')
    src = interfaces.CameraLabelComponentSrc(
        frame=frame,
        camera_labels=frame.camera_labels[0],
        label=frame.camera_labels[0].labels[0],
    )
    extractor = box.CameraToLiDARBoxAssociationComponentExtractor()
    self.assertRaises(StopIteration, extractor(src).__next__)


if __name__ == '__main__':
  absltest.main()
