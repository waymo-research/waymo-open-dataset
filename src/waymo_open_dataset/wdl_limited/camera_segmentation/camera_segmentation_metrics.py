# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================
"""Camera segmentation metrics."""

import copy
from typing import Sequence, Union

import numpy as np

from deeplab2.data import dataset
from deeplab2.evaluation import segmentation_and_tracking_quality
from waymo_open_dataset.protos import camera_segmentation_metrics_pb2


def get_eval_config() -> dataset.DatasetDescriptor:
  """Gets waymo specific config file for evaluation.

  Returns:
    An instance of the `dataset.DatasetDescriptor`.
  """
  dataset_name = (
      dataset.WOD_PVPS_IMAGE_PANOPTIC_SEG_MULTICAM_DATASET.dataset_name
  )
  deeplab_config = dataset.MAP_NAME_TO_DATASET_INFO[dataset_name]
  return deeplab_config


def get_metric_object_by_sequence(
    true_panoptic_labels: Sequence[np.ndarray],
    pred_panoptic_labels: Sequence[np.ndarray],
    num_cameras_covered: Sequence[np.ndarray],
    is_tracked_masks: Sequence[np.ndarray],
    sequence_id: Union[str, int] = 0,
    offset: int = 256 * 256 * 256,
) -> segmentation_and_tracking_quality.STQuality:
  """Creates a metric object and updates state for a single sequence.

  Args:
    true_panoptic_labels: A sequence of the ground-truth panoptic labels for a
      particular video.
    pred_panoptic_labels: A sequence of the predicted panoptic labels, where
      each element shares the same shape with the corresponding ground-truth.
    num_cameras_covered: A sequence of the integer array, where each array
      represents the number of cameras that overlaped, sharing the same shape
      with the corresponding ground-truth array.
    is_tracked_masks: A sequence of the boolean array, where each array
      indicates whether there is consistent global IDs between frames, sharing
      the same shape with the corresponding ground-truth array.
    sequence_id: An integer or a string, indicates the ID of the sequence the
      frames belong to (default: 0).
    offset: An integer, indicates the maximum number of unique labels.

  Returns:
    An instance of the STQuality metric object, containing the intermediate
      state which can be further decoded into the wSTQ, wAQ, and mIoU metrics
      for a single sequence.

  Raises:
    `ValueError' if either the length of the prediction sequence, the
      number of cameras sequence, or is tracked masks does not match the length
      of the ground-truth sequence.

  """
  deeplab_config = get_eval_config()

  num_frames = len(true_panoptic_labels)
  if num_frames != len(pred_panoptic_labels):
    raise ValueError('Inconsistent number of predicted labels.')
  if num_frames != len(num_cameras_covered):
    raise ValueError('Inconsistent number of camera coverage masks.')
  if num_frames != len(is_tracked_masks):
    raise ValueError('Inconsistent number of is tracked masks.')

  panoptic_label_divisor = deeplab_config.panoptic_label_divisor
  metric_obj = segmentation_and_tracking_quality.STQuality(
      num_classes=deeplab_config.num_classes,
      things_list=deeplab_config.class_has_instances_list,
      ignore_label=deeplab_config.ignore_label,
      max_instances_per_category=panoptic_label_divisor,
      offset=offset,
  )

  for y_true, y_pred, coverage, is_tracked in zip(
      true_panoptic_labels,
      pred_panoptic_labels,
      num_cameras_covered,
      is_tracked_masks,
  ):
    y_true_augmented = y_true.copy()
    # Set untracked instances to instance id 0 (crowd).
    y_true_augmented[is_tracked == 0] = (
        y_true_augmented[is_tracked == 0] // panoptic_label_divisor
    ) * panoptic_label_divisor
    weights = 1.0 / np.maximum(
        coverage.astype(np.float32), 1e-8
    )
    metric_obj.update_state(y_true_augmented, y_pred, sequence_id, weights)
  return metric_obj


def aggregate_metrics(
    metric_objects: Sequence[segmentation_and_tracking_quality.STQuality],
) -> camera_segmentation_metrics_pb2.CameraSegmentationMetrics:
  """Aggregates camera segmentation metrics.

  Args:
    metric_objects: A sequence of STQuality metric objects.

  Returns:
    A metric proto with the fields `wstq', `waq', and `miou' computed.
      - wstq: weighted Segmentation and Tracking Quality.
      - waq: weighted Association Quality.
      - miou: mean Intersection over Union.
  """
  aggregated_metric_obj = copy.deepcopy(metric_objects[0])
  aggregated_metric_obj.reset_states()
  aggregated_metric_obj.merge_state(metric_objects)
  result = aggregated_metric_obj.result()
  metrics = camera_segmentation_metrics_pb2.CameraSegmentationMetrics(
      wstq=result['STQ'],
      waq=result['AQ'],
      miou=result['IoU'],
  )

  return metrics
