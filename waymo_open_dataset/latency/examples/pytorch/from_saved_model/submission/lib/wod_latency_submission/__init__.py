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
"""Module that contains the methods required in the evaluation script."""
import numpy as np
from pcdet.config import cfg
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import torch

# These locations are where the model weights and other misc files are stored.
MODEL_CFG_PATH = '/code/submission/OpenPCDet/tools/cfgs/waymo_models/pv_rcnn.yaml'
MODEL_WEIGHTS = '/code/submission/lib/wod_latency_submission/WAYMO_MODEL_WEIGHTS.pth'

# The names of the lidars and input fields that users might want to use for
# detection.
LIDAR_NAMES = ['TOP', 'REAR', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT']
LIDAR_FIELD = 'RANGE_IMAGE_FIRST_RETURN'

# The data fields requested from the evaluation script should be specified in
# this field in the module.
DATA_FIELDS = [lidar_name + '_' + LIDAR_FIELD for lidar_name in LIDAR_NAMES]

# Global variables that hold the models and configurations.
model_cfg = cfg_from_yaml_file(MODEL_CFG_PATH, cfg)
logger = common_utils.create_logger()
dataset_processor = DatasetTemplate(
    dataset_cfg=model_cfg.DATA_CONFIG,
    class_names=model_cfg.CLASS_NAMES,
    training=False,
    root_path=None,
    logger=logger)
model = None


def initialize_model():
  """Method that will be called by the evaluation script to load the model and weights.
  """
  global model
  model = build_network(
      model_cfg=model_cfg.MODEL, num_class=len(model_cfg.CLASS_NAMES),
      dataset=dataset_processor)
  model.load_params_from_file(MODEL_WEIGHTS, logger=logger, to_cpu=False)
  model.cuda()
  model.eval()


def _process_inputs(input_sensor_dict):
  """Converts raw input evaluation data into a dictionary that can be fed into the OpenPCDet model.

  Args:
    input_sensor_dict: dictionary mapping string input data field name to a
    numpy array of the data.

  Returns:
    Dict with pre-processed input that can be passed to the model.
  """
  points = []
  for field in DATA_FIELDS:
    # H x W x 6
    lidar_range_image = input_sensor_dict[field]
    # Flatten and select all points with positive range.
    lidar_range_image = lidar_range_image[lidar_range_image[..., 0] > 0, :]
    # Also transform last dimension from
    #   (range, intensity, elongation, x, y, z) ->
    #   (x, y, z, intensity, elongation)
    points.append(lidar_range_image[:, [3, 4, 5, 1, 2]])

  # Concatenate points from all lidars.
  points_all = np.concatenate(points, axis=0).astype(np.float32)
  points_all[:, 3] = np.tanh(points_all[:, 3])

  return dataset_processor.prepare_data(data_dict={'points': points_all})


def run_model(**kwargs):
  """Run inference on the pre-loaded OpenPCDet library model.

  Args:
    **kwargs: One keyword argument per input data field from the evaluation
    script.

  Returns:
    Dict from string to numpy ndarray.
  """
  data_dict = _process_inputs(kwargs)

  with torch.no_grad():
    data_dict = dataset_processor.collate_batch([data_dict])
    load_data_to_gpu(data_dict)
    pred_dicts, _ = model.forward(data_dict)
    # Rename the outputs to the format expected by the evaluation script.
    return {'boxes': pred_dicts[0]['pred_boxes'].cpu().numpy(),
            'scores': pred_dicts[0]['pred_scores'].cpu().numpy(),
            'classes': pred_dicts[0]['pred_labels'].cpu().numpy()}
