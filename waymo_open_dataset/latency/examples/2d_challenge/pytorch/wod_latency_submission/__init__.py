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
"""Module to load and run a Faster R-CNN model."""
import numpy as np
import torch
import torchvision

model = None
DATA_FIELDS = ['FRONT_IMAGE']


def initialize_model():
  global model
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  model.eval()


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
  # Convert the image to a PyTorch-friendly format by casting it from uint8 to
  # float32 (and dividing by 255 to take it from [0, 255] to [0, 1]) and
  # transposing it from H x W x C to C x H x W.
  transposed_float_img = np.transpose(
      FRONT_IMAGE.astype(np.float32) / 255.0, [2, 0, 1])

  outputs = model([torch.from_numpy(transposed_float_img)])
  corners = outputs[0]['boxes'][0, ...].detach().numpy()
  scores = outputs[0]['scores'][0, ...].detach().numpy()
  coco_classes = outputs[0]['labels'][0, ...].detach().numpy()

  # Convert the classes from COCO classes to WOD classes, and only keep
  # detections that belong to a WOD class.
  wod_classes = np.vectorize(translate_label_to_wod)(coco_classes)
  corners = corners[wod_classes > 0]
  scores = scores[wod_classes > 0]
  classes = wod_classes[wod_classes > 0]

  # Note: Torchvision's pretrained models returns boxes in the format
  # (ymin, xmin, ymax, xmax). Thus, this function needs to convert them to the
  # format expected by WOD, namely (center_x, center_y, width, height).
  boxes = np.zeros_like(corners)
  boxes[:, 0] = (corners[:, 3] + corners[:, 1]) / 2.0
  boxes[:, 1] = (corners[:, 2] + corners[:, 0]) / 2.0
  boxes[:, 2] = (corners[:, 3] - corners[:, 1])
  boxes[:, 3] = (corners[:, 2] - corners[:, 0])

  return {
      'boxes': boxes,
      'scores': scores,
      'classes': classes,
  }
# BEGIN-INTERNAL
# pylint: disable=invalid-name
# END-INTERNAL
