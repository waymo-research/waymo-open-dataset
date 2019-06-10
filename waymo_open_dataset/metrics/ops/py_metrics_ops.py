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
"""Waymo Open Dataset tensorflow ops python interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

detection_metrics_module = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile('detection_metrics_ops.so'))
detection_metrics = detection_metrics_module.detection_metrics
