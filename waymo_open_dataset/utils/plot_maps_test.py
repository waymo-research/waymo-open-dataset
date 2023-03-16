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
"""Tests for open dataset map plotting functions."""

from absl.testing import absltest

from google.protobuf import text_format
from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils import plot_maps


class PlotMapsTest(absltest.TestCase):

  def test_draw_lane(self):
    feature_str = """
      id: 1
      lane: {
        speed_limit_mph : 0.0
        type    : TYPE_SURFACE_STREET
        polyline: {
          x: 1
          y: 0
          z: 0
        }
        polyline: {
          x: 1
          y: 1
          z: 0
        }
        polyline: {
          x: 1
          y: 2
          z: 0
        }
      }
    """
    feature = text_format.Parse(feature_str, map_pb2.MapFeature())
    fig = plot_maps.plot_map_features([feature])
    line_traces = list(fig.select_traces({'mode': 'lines'}))
    self.assertLen(line_traces, 1)
    trace = line_traces[0]
    self.assertEqual(trace.line['color'], 'royalblue')
    self.assertEqual(trace.line['dash'], 'solid')
    self.assertLen(trace.x, 3)
    self.assertEqual(trace.mode, 'lines')
    self.assertTupleEqual(trace['x'], (1.0, 1.0, 1.0))
    self.assertTupleEqual(trace['y'], (0.0, 1.0, 2.0))
    self.assertTupleEqual(trace['z'], (0.0, 0.0, 0.0))

  def test_draw_road_line(self):
    feature_str = """
      id: 1
      road_line: {
        type    : TYPE_BROKEN_SINGLE_YELLOW
        polyline: {
          x: 1
          y: 0
          z: 0
        }
        polyline: {
          x: 1
          y: 1
          z: 0
        }
        polyline: {
          x: 1
          y: 2
          z: 0
        }
      }
    """
    feature = text_format.Parse(feature_str, map_pb2.MapFeature())
    fig = plot_maps.plot_map_features([feature])
    line_traces = list(fig.select_traces({'mode': 'lines'}))
    self.assertLen(line_traces, 1)
    trace = line_traces[0]
    self.assertEqual(trace.line['color'], 'yellow')
    self.assertEqual(trace.line['dash'], 'dash')
    self.assertLen(trace.x, 3)
    self.assertEqual(trace.mode, 'lines')
    self.assertTupleEqual(trace['x'], (1.0, 1.0, 1.0))
    self.assertTupleEqual(trace['y'], (0.0, 1.0, 2.0))
    self.assertTupleEqual(trace['z'], (0.0, 0.0, 0.0))

  def test_draw_road_edge(self):
    feature_str = """
      id: 1
      road_edge: {
        type    : TYPE_ROAD_EDGE_BOUNDARY
        polyline: {
          x: 1
          y: 0
          z: 0
        }
        polyline: {
          x: 1
          y: 1
          z: 0
        }
        polyline: {
          x: 1
          y: 2
          z: 0
        }
      }
    """
    feature = text_format.Parse(feature_str, map_pb2.MapFeature())
    fig = plot_maps.plot_map_features([feature])
    line_traces = list(fig.select_traces({'mode': 'lines'}))
    self.assertLen(line_traces, 1)
    trace = line_traces[0]
    self.assertEqual(trace.line['color'], 'green')
    self.assertEqual(trace.line['dash'], 'solid')
    self.assertLen(trace.x, 3)
    self.assertEqual(trace.mode, 'lines')
    self.assertTupleEqual(trace['x'], (1.0, 1.0, 1.0))
    self.assertTupleEqual(trace['y'], (0.0, 1.0, 2.0))
    self.assertTupleEqual(trace['z'], (0.0, 0.0, 0.0))

  def test_draw_stop_sign(self):
    feature_str = """
      id: 1
      stop_sign: {
        lane    : 0
        position: {
          x: 1
          y: 0
          z: 0
        }
      }
    """
    feature = text_format.Parse(feature_str, map_pb2.MapFeature())
    fig = plot_maps.plot_map_features([feature])
    line_traces = list(fig.select_traces({'mode': 'lines'}))
    self.assertLen(line_traces, 1)
    trace = line_traces[0]
    self.assertEqual(trace.line['color'], 'red')
    self.assertEqual(trace.line['dash'], 'solid')
    self.assertLen(trace.x, 1)
    self.assertEqual(trace.mode, 'lines')
    self.assertTupleEqual(trace['x'], (1.0,))
    self.assertTupleEqual(trace['y'], (0.0,))
    self.assertTupleEqual(trace['z'], (0.0,))

  def test_draw_crosswalk(self):
    feature_str = """
      id: 1
      crosswalk: {
        polygon: {
          x: 1
          y: 0
          z: 0
        }
        polygon: {
          x: 1
          y: 1
          z: 0
        }
        polygon: {
          x: 1
          y: 2
          z: 0
        }
      }
    """
    feature = text_format.Parse(feature_str, map_pb2.MapFeature())
    fig = plot_maps.plot_map_features([feature])
    line_traces = list(fig.select_traces({'mode': 'lines'}))
    self.assertLen(line_traces, 1)
    trace = line_traces[0]
    self.assertEqual(trace.line['color'], 'orange')
    self.assertEqual(trace.line['dash'], 'solid')
    self.assertLen(trace.x, 4)
    self.assertEqual(trace.mode, 'lines')
    self.assertTupleEqual(trace['x'], (1.0, 1.0, 1.0, 1.0))
    self.assertTupleEqual(trace['y'], (0.0, 1.0, 2.0, 0.0))
    self.assertTupleEqual(trace['z'], (0.0, 0.0, 0.0, 0.0))

  def test_draw_speed_bump(self):
    feature_str = """
      id: 1
      speed_bump: {
        polygon: {
          x: 1
          y: 0
          z: 0
        }
        polygon: {
          x: 1
          y: 1
          z: 0
        }
        polygon: {
          x: 1
          y: 2
          z: 0
        }
      }
    """
    feature = text_format.Parse(feature_str, map_pb2.MapFeature())
    fig = plot_maps.plot_map_features([feature])
    line_traces = list(fig.select_traces({'mode': 'lines'}))
    self.assertLen(line_traces, 1)
    trace = line_traces[0]
    self.assertEqual(trace.line['color'], 'cyan')
    self.assertEqual(trace.line['dash'], 'solid')
    self.assertLen(trace.x, 4)
    self.assertEqual(trace.mode, 'lines')
    self.assertTupleEqual(trace['x'], (1.0, 1.0, 1.0, 1.0))
    self.assertTupleEqual(trace['y'], (0.0, 1.0, 2.0, 0.0))
    self.assertTupleEqual(trace['z'], (0.0, 0.0, 0.0, 0.0))

  def test_draw_driveway(self):
    feature_str = """
      id: 0
      driveway: {
        polygon: {
          x: 1
          y: 0
          z: 0
        }
        polygon: {
          x: 1
          y: 1
          z: 0
        }
        polygon: {
          x: 1
          y: 2
          z: 0
        }
      }
    """
    feature = text_format.Parse(feature_str, map_pb2.MapFeature())
    fig = plot_maps.plot_map_features([feature])
    line_traces = list(fig.select_traces({'mode': 'lines'}))
    self.assertLen(line_traces, 1)
    trace = line_traces[0]
    self.assertEqual(trace.line['color'], 'blue')
    self.assertEqual(trace.line['dash'], 'solid')
    self.assertLen(trace.x, 4)
    self.assertEqual(trace.mode, 'lines')
    self.assertTupleEqual(trace['x'], (1.0, 1.0, 1.0, 1.0))
    self.assertTupleEqual(trace['y'], (0.0, 1.0, 2.0, 0.0))
    self.assertTupleEqual(trace['z'], (0.0, 0.0, 0.0, 0.0))


if __name__ == '__main__':
  absltest.main()
