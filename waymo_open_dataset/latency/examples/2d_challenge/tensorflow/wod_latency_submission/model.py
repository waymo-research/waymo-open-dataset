# Copyright 2021 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""Module to load and run an EfficientDet model."""
import os

import numpy as np
from object_detection.builders import model_builder
from object_detection.utils import config_util
import tensorflow as tf


model = None
DATA_FIELDS = ['FRONT_IMAGE']


def initialize_model():
  """Initialize the global model variable to the pretrained EfficientDet.

  This assumes that the EfficientDet model has already been downloaded to a
  specific path, as done in the Dockerfile for this example.
  """
  model_dir = '/models/research/object_detection/test_data/efficientdet_d5_coco17_tpu-32'
  configs = config_util.get_configs_from_pipeline_file(
      os.path.join(model_dir, 'pipeline.config'))
  model_config = configs['model']
  global model
  model = model_builder.build(model_config=model_config, is_training=False)
  ckpt = tf.train.Checkpoint(model=model)
  ckpt.restore(os.path.join(model_dir, 'checkpoint', 'ckpt-0'))


def translate_label_to_wod(label):
  """Translate a single COCO class to its corresponding WOD class.

  Note: Returns -1 if this COCO class has no corresponding class in WOD.

  Args:
    label: int COCO class label

  Returns:
    Int WOD class label, or -1.
  """
  label_conversion_map = {
      1: 2,   # Person is ped
      2: 4,   # Bicycle is bicycle
      3: 1,   # Car is vehicle
      4: 1,   # Motorcycle is vehicle
      6: 1,   # Bus is vehicle
      8: 1,   # Truck is vehicle
      13: 3,  # Stop sign is sign
  }
  return label_conversion_map.get(label, -1)


# BEGIN-INTERNAL
# pylint: disable=invalid-name
# END-INTERNAL
def run_model(FRONT_IMAGE):
  """Run the model on the RGB image.

  Args:
    FRONT_IMAGE: H x W x 3 numpy ndarray.

  Returns:
    Dict from string to numpy ndarray.
  """
  # Run the model.
  inp_tensor = tf.convert_to_tensor(
      np.expand_dims(FRONT_IMAGE, 0), dtype=tf.float32)
  image, shapes = model.preprocess(inp_tensor)
  pred_dict = model.predict(image, shapes)
  detections = model.postprocess(pred_dict, shapes)
  corners = detections['detection_boxes'][0, ...].numpy()
  scores = detections['detection_scores'][0, ...].numpy()
  coco_classes = detections['detection_classes'][0, ...].numpy()
  coco_classes = coco_classes.astype(np.uint8)

  # Convert the classes from COCO classes to WOD classes, and only keep
  # detections that belong to a WOD class.
  wod_classes = np.vectorize(translate_label_to_wod)(coco_classes)
  corners = corners[wod_classes > 0]
  scores = scores[wod_classes > 0]
  classes = wod_classes[wod_classes > 0]

  # Note: the boxes returned by the TF OD API's pretrained models returns boxes
  # in the format (ymin, xmin, ymax, xmax), normalized to [0, 1]. Thus, this
  # function needs to convert them to the format expected by WOD, namely
  # (center_x, center_y, width, height) in pixels.
  boxes = np.zeros_like(corners)
  boxes[:, 0] = (corners[:, 3] + corners[:, 1]) * FRONT_IMAGE.shape[0] / 2.0
  boxes[:, 1] = (corners[:, 2] + corners[:, 0]) * FRONT_IMAGE.shape[1] / 2.0
  boxes[:, 2] = (corners[:, 3] - corners[:, 1]) * FRONT_IMAGE.shape[0]
  boxes[:, 3] = (corners[:, 2] - corners[:, 0]) * FRONT_IMAGE.shape[1]

  return {
      'boxes': boxes,
      'scores': scores,
      'classes': classes,
  }
# BEGIN-INTERNAL
# pylint: disable=invalid-name
# END-INTERNAL
