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
# =============================================================================
"""Tests for keypoint_draw."""

from matplotlib import collections as mc
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.utils import keypoint_data as _data
from waymo_open_dataset.utils import keypoint_draw as _lib
from waymo_open_dataset.utils import keypoint_test_util as _util

_NOSE = keypoint_pb2.KeypointType.KEYPOINT_TYPE_NOSE
_LEFT_SHOULDER = keypoint_pb2.KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER
_RIGHT_SHOULDER = keypoint_pb2.KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER


def _get_collection(ax, col_type):
  cols = [c for c in ax.collections if isinstance(c, col_type)]
  assert len(cols) == 1, f'Required a single {col_type}, got {cols}'
  return cols[0]


def _vec2d_as_array(location_px: keypoint_pb2.Vec2d) -> np.ndarray:
  return np.array([location_px.x, location_px.y])


class EdgeTest(tf.test.TestCase):

  def test_simple_edge_between_two_points_creates_single_line(self):
    coords = {
        _LEFT_SHOULDER: _data.Point([0, 0]),
        _RIGHT_SHOULDER: _data.Point([1, 1])
    }
    colors = {_LEFT_SHOULDER: '#AAAAAA', _RIGHT_SHOULDER: '#BBBBBB'}

    edge = _lib.SolidLineEdge(_LEFT_SHOULDER, _RIGHT_SHOULDER, 1)
    lines = edge.create_lines(coords, colors)

    self.assertLen(lines, 1)
    # Line has the same color as the start point.
    self.assertEqual(lines[0].color, '#AAAAAA')

  def test_bicolored_edge_between_two_points_creates_two_lines(self):
    coords = {
        _LEFT_SHOULDER: _data.Point([0, 0]),
        _RIGHT_SHOULDER: _data.Point([1, 1])
    }
    colors = {_LEFT_SHOULDER: '#AAAAAA', _RIGHT_SHOULDER: '#BBBBBB'}

    edge = _lib.BicoloredEdge(_LEFT_SHOULDER, _RIGHT_SHOULDER, 1)
    lines = edge.create_lines(coords, colors)

    self.assertLen(lines, 2)
    self.assertEqual(lines[0].color, '#AAAAAA')
    self.assertEqual(lines[1].color, '#BBBBBB')

  def test_multipoint_edge_averages_locations_to_get_ends(self):
    coords = {
        _LEFT_SHOULDER: _data.Point([2, 0]),
        _RIGHT_SHOULDER: _data.Point([0, 2]),
        _NOSE: _data.Point([3, 3])
    }
    colors = {
        _LEFT_SHOULDER: '#AAAAAA',
        _RIGHT_SHOULDER: '#BBBBBB',
        _NOSE: '#CCCCCC'
    }

    edge = _lib.MultipointEdge([_LEFT_SHOULDER, _RIGHT_SHOULDER], [_NOSE], 1)
    lines = edge.create_lines(coords, colors)

    self.assertLen(lines, 2)
    np.testing.assert_allclose(lines[0].start, [1, 1])
    np.testing.assert_allclose(lines[0].end, [2, 2])
    self.assertEqual(lines[0].color, _lib.DEFAULT_COLOR)
    np.testing.assert_allclose(lines[1].start, [2, 2])
    np.testing.assert_allclose(lines[1].end, [3, 3])
    self.assertEqual(lines[1].color, '#CCCCCC')


class WireframeTest(tf.test.TestCase):

  def test_build_camera_wireframe_dot_and_lines_have_correct_attributes(self):
    camera_keypoints = [
        _util.camera_keypoint(_NOSE, location_px=(2, 1), is_occluded=True),
        _util.camera_keypoint(_LEFT_SHOULDER, location_px=(1, 0)),
        _util.camera_keypoint(_RIGHT_SHOULDER, location_px=(3, 0))
    ]
    config = _lib.WireframeConfig(
        edges=[_lib.BicoloredEdge(_LEFT_SHOULDER, _RIGHT_SHOULDER, width=2)],
        point_colors={
            _NOSE: '#FFFFFF',
            _LEFT_SHOULDER: '#AAAAAA',
            _RIGHT_SHOULDER: '#BBBBBB'
        },
        point_sizes={
            _NOSE: 1,
            _LEFT_SHOULDER: 2,
            _RIGHT_SHOULDER: 3
        })

    wireframe = _lib.build_camera_wireframe(camera_keypoints, config)

    with self.subTest(name='CorrectNumberOfLines'):
      # Two lines per edge, e.g. 2 for the edge between shoulders.
      self.assertLen(wireframe.lines, 2)
    with self.subTest(name='PopulatesLineColors'):
      self.assertEqual([l.color for l in wireframe.lines],
                       ['#AAAAAA', '#BBBBBB'])
    with self.subTest(name='PopulatesLineWidth'):
      self.assertEqual([l.width for l in wireframe.lines], [2, 2])
    with self.subTest(name='CorrectNumberOfDots'):
      # One dot per keypoint.
      self.assertLen(wireframe.dots, 3)
    with self.subTest(name='PopulatesDotColors'):
      self.assertEqual([d.color for d in wireframe.dots],
                       ['#FFFFFF', '#AAAAAA', '#BBBBBB'])
    with self.subTest(name='PopulatesDotSizes'):
      self.assertEqual([d.size for d in wireframe.dots], [1, 2, 3])
    with self.subTest(name='PopulatesBoarderColorForOccludedPoints'):
      self.assertEqual([d.border_color for d in wireframe.dots],
                       [_lib.OCCLUDED_COLOR, None, None])
      self.assertEqual([d.actual_border_color for d in wireframe.dots],
                       [_lib.OCCLUDED_COLOR, '#AAAAAA', '#BBBBBB'])

  def test_build_laser_wireframe_dot_and_lines_have_correct_attributes(self):
    laser_keypoints = [
        _util.laser_keypoint(_NOSE, location_m=(2, 1, 1), is_occluded=True),
        _util.laser_keypoint(_LEFT_SHOULDER, location_m=(1, 0, 1)),
        _util.laser_keypoint(_RIGHT_SHOULDER, location_m=(3, 0, 1))
    ]
    config = _lib.WireframeConfig(
        edges=[_lib.BicoloredEdge(_LEFT_SHOULDER, _RIGHT_SHOULDER, width=2)],
        point_colors={
            _NOSE: '#FFFFFF',
            _LEFT_SHOULDER: '#AAAAAA',
            _RIGHT_SHOULDER: '#BBBBBB'
        },
        point_sizes={
            _NOSE: 1,
            _LEFT_SHOULDER: 2,
            _RIGHT_SHOULDER: 3
        })

    wireframe = _lib.build_laser_wireframe(laser_keypoints, config)

    with self.subTest(name='CorrectNumberOfLines'):
      # Two lines per edge, e.g. 2 for the edge between shoulders.
      self.assertLen(wireframe.lines, 2)
    with self.subTest(name='PopulatesLineColors'):
      self.assertEqual([l.color for l in wireframe.lines],
                       ['#AAAAAA', '#BBBBBB'])
    with self.subTest(name='PopulatesLineWidth'):
      self.assertEqual([l.width for l in wireframe.lines], [2, 2])
    with self.subTest(name='CorrectNumberOfDots'):
      # One dot per keypoint.
      self.assertLen(wireframe.dots, 3)
    with self.subTest(name='PopulatesDotColors'):
      self.assertEqual([d.color for d in wireframe.dots],
                       ['#FFFFFF', '#AAAAAA', '#BBBBBB'])
    with self.subTest(name='PopulatesDotSizes'):
      self.assertEqual([d.size for d in wireframe.dots], [1, 2, 3])
    with self.subTest(name='PopulatesDotNames'):
      self.assertEqual([d.name for d in wireframe.dots],
                       ['NOSE?', 'LEFT_SHOULDER', 'RIGHT_SHOULDER'])
    with self.subTest(name='PopulatesBoarderColorForOccludedPoints'):
      self.assertEqual([d.border_color for d in wireframe.dots],
                       [_lib.OCCLUDED_COLOR, None, None])
      self.assertEqual([d.actual_border_color for d in wireframe.dots],
                       [_lib.OCCLUDED_COLOR, '#AAAAAA', '#BBBBBB'])

  def test_build_camera_wireframe_factors_modify_sizes(self):
    camera_keypoints = [
        _util.camera_keypoint(_NOSE, location_px=(2, 1)),
        _util.camera_keypoint(_LEFT_SHOULDER, location_px=(1, 0)),
        _util.camera_keypoint(_RIGHT_SHOULDER, location_px=(3, 0))
    ]
    config = _lib.WireframeConfig(
        edges=[_lib.BicoloredEdge(_LEFT_SHOULDER, _RIGHT_SHOULDER, width=2)],
        point_colors={
            _NOSE: '#FFFFFF',
            _LEFT_SHOULDER: '#AAAAAA',
            _RIGHT_SHOULDER: '#BBBBBB'
        },
        point_sizes={
            _NOSE: 1,
            _LEFT_SHOULDER: 2,
            _RIGHT_SHOULDER: 3
        },
        dot_size_factor=2.0,
        line_width_factor=3.0)

    wireframe = _lib.build_camera_wireframe(camera_keypoints, config)

    with self.subTest(name='ModfiesLineWidth'):
      self.assertEqual([l.width for l in wireframe.lines], [6, 6])
    with self.subTest(name='ModifiesDotSizes'):
      self.assertEqual([l.size for l in wireframe.dots], [2, 4, 6])


class DrawTest(tf.test.TestCase):

  def test_draw_camera_wireframe_adds_collections_to_the_axis(self):
    wireframe = _lib.Wireframe(
        lines=[
            _lib.Line(start=[0, 0], end=[1, 1], color='#AAAAAA', width=1),
            _lib.Line(start=[1, 1], end=[2, 2], color='#BBBBBB', width=1),
        ],
        dots=[
            _lib.Dot(location=[0, 0], color='#FFFFFF', size=1, name='A'),
            _lib.Dot(location=[1, 1], color='#FFFFFF', size=2, name='B'),
            _lib.Dot(location=[2, 2], color='#FFFFFF', size=3, name='C')
        ])

    _, ax = plt.subplots()
    _lib.draw_camera_wireframe(ax, wireframe)

    with self.subTest(name='CorrectNumberOfLineSegments'):
      line_collection = _get_collection(ax, mc.LineCollection)
      self.assertLen(line_collection.get_segments(), 2)
    with self.subTest(name='CorrectNumberOfDots'):
      el_collection = _get_collection(ax, mc.EllipseCollection)
      self.assertEqual(el_collection.get_offsets().shape, (3, 2))

  def test_draw_laser_wireframe_adds_traces_to_the_figure(self):
    wireframe = _lib.Wireframe(
        lines=[
            _lib.Line(start=[0, 0, 0], end=[1, 1, 1], color='#AAAAAA', width=1),
            _lib.Line(start=[1, 1, 1], end=[2, 2, 2], color='#BBBBBB', width=1),
        ],
        dots=[
            _lib.Dot(location=[0, 0, 0], color='#FFFFFF', size=1, name='A'),
            _lib.Dot(location=[1, 1, 1], color='#FFFFFF', size=2, name='B'),
            _lib.Dot(location=[2, 2, 2], color='#FFFFFF', size=3, name='C')
        ])

    fig = go.Figure()
    _lib.draw_laser_wireframe(fig, wireframe)

    with self.subTest(name='CorrectNumberOfLineSegments'):
      line_traces = list(fig.select_traces({'mode': 'lines'}))
      self.assertLen(line_traces, 2)
    with self.subTest(name='CorrectNumberOfDots'):
      dot_traces = list(fig.select_traces({'mode': 'markers'}))
      self.assertLen(dot_traces, 1)
      self.assertLen(dot_traces[0].x, 3)

  def test_crop_camera_keypoints_returns_updated_keypoints_and_image(self):
    image = np.zeros((450, 250, 3))
    camera_keypoints = [
        _util.camera_keypoint(location_px=(100, 200)),
        _util.camera_keypoint(location_px=(150, 200)),
    ]
    # NOTE: Box.length is along OX, Box.width is along OY
    box = label_pb2.Label.Box(center_x=100, center_y=200, length=100, width=200)

    new_image, new_keypoints = _lib.crop_camera_keypoints(
        image, camera_keypoints, box)

    with self.subTest(name='CroppedImageSizeIsCorrect'):
      self.assertEqual(new_image.shape, (200, 100, 3))
    with self.subTest(name='CroppedKeypointsShifted'):
      np.testing.assert_allclose(
          _vec2d_as_array(new_keypoints[0].keypoint_2d.location_px), [50, 100])
      np.testing.assert_allclose(
          _vec2d_as_array(new_keypoints[1].keypoint_2d.location_px), [100, 100])


if __name__ == '__main__':
  tf.test.main()
