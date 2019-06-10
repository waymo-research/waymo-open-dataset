# Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""Tests for waymo_open_dataset.metrics.python.config_util_py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from google.protobuf import text_format
from waymo_open_dataset.metrics.python import config_util
from waymo_open_dataset.metrics.python import config_util_py
from waymo_open_dataset.protos import metrics_pb2


class ConfigUtilTest(unittest.TestCase):

  def _build_config(self):
    config = metrics_pb2.Config()
    config_text = """
    num_desired_score_cutoffs: 11
    breakdown_generator_ids: OBJECT_TYPE
    breakdown_generator_ids: ONE_SHARD
    breakdown_generator_ids: RANGE
    breakdown_generator_ids: RANGE
    difficulties {
    }
    difficulties {
      levels: LEVEL_1
    }
    difficulties {
      levels: LEVEL_2
      levels: LEVEL_1
    }
    difficulties {
      levels: LEVEL_1
    }
    matcher_type: TYPE_HUNGARIAN
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    box_type: TYPE_3D
    """
    text_format.Merge(config_text, config)
    return config

  def test_get_breakdown_names_from_config(self):
    config = self._build_config()
    clif_names = config_util.GetBreakdownNamesFromConfig(config)
    py_names = config_util_py.get_breakdown_names_from_config(config)
    self.assertEqual(len(clif_names), len(py_names))
    for a, b in zip(clif_names, py_names):
      self.assertEqual(a, b)


if __name__ == '__main__':
  unittest.main()
