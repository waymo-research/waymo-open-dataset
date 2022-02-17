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
# ==============================================================================
"""Quality metrics for human keypoints."""
import dataclasses
from typing import Collection, Dict, List, Optional, Tuple

import tensorflow as tf


@dataclasses.dataclass
class KeypointsTensors:
  """Tensors to represent 2D or 3D keypoints.

  Shape descriptions below use the following notation:
    B - number of samples in the batch (aka batch_size).
    N - number of keypoints.
    D - number of dimensions (e.g. 2 or 3).

  Attributes:
    location: a float tensor with shape [B, N, D] or [N, D].
    visibility: a float tensor with shape [B, N] or [N], with values: 0 -
      corresponding point is missing (not labeled or not detected), 1 - present,
      but marked as occluded, 2 - marked as visible or not occluded.
    has_batch_dimension: True if location and visibility have batch dimensions.
  """
  location: tf.Tensor
  visibility: tf.Tensor

  def __post_init__(self):
    if self.location.shape.rank == 2:
      self.visibility.shape.assert_has_rank(1)
      self.visibility.shape.assert_is_compatible_with(self.location.shape[:1])
    elif self.location.shape.rank == 3:
      self.visibility.shape.assert_has_rank(2)
      self.visibility.shape.assert_is_compatible_with(self.location.shape[:2])
    else:
      raise ValueError('Rank of the location has to be 2 or 3, '
                       f'got {self.location.shape.rank}')
    if self.location.shape[-1] not in (2, 3):
      raise ValueError(
          f'Support only 2 or 3 dimensions: got {self.location.shape[-1]}')

  @property
  def has_batch_dimension(self) -> bool:
    return self.location.shape.rank == 3


def stack_keypoints(values: List[KeypointsTensors],
                    axis: int = 0) -> KeypointsTensors:
  """Dispatch method to support tf.stack for `BoundingBoxTensors`."""
  return KeypointsTensors(
      location=tf.stack([v.location for v in values], axis),
      visibility=tf.stack([v.visibility for v in values], axis))


@dataclasses.dataclass
class BoundingBoxTensors:
  """Tensors to represent 2D or 3D bounding boxes.

  Shape descriptions below use the following notation:
    B - number of samples in the batch (aka batch_size).
    D - number of dimensions (e.g. 2 or 3).

  Attributes:
    center: a float tensor with shape [B, D] or [D].
    size: a float tensor with shape [B, D] or [D].
    scale: a float tensor with shape [B] or [], which means square root of the
      box's area in 2D and cubic root the volume in 3D.
    min_corner: corner with smallest coordinates, e.g. top left corner for a 2D
      box.
    max_corner: corner with largest coordinates, e.g. bottom right corner for a
      2D box.
    has_batch_dimension: True if location and visibility have batch dimensions.
  """
  center: tf.Tensor
  size: tf.Tensor

  @property
  def scale(self) -> tf.Tensor:
    return tf.pow(tf.reduce_prod(self.size, axis=-1), 1.0 / self.size.shape[-1])

  @property
  def min_corner(self) -> tf.Tensor:
    return self.center - self.size / 2

  @property
  def max_corner(self) -> tf.Tensor:
    return self.center + self.size / 2

  def __post_init__(self):
    if self.center.shape.rank not in (1, 2):
      raise ValueError(
          f'Rank of the center has to be 1 or 2, got {self.center.shape.rank}')
    if self.center.shape[-1] not in (2, 3):
      raise ValueError(
          f'Support only 2 or 3 dimensions: got {self.center.shape[-1]}')
    self.center.shape.assert_is_compatible_with(self.size.shape)

  @property
  def has_batch_dimension(self) -> bool:
    return self.center.shape.rank == 2


def stack_boxes(values: List[BoundingBoxTensors],
                axis: int = 0) -> BoundingBoxTensors:
  """Dispatch method to support tf.stack for `BoundingBoxTensors`."""
  return BoundingBoxTensors(
      center=tf.stack([v.center for v in values], axis),
      size=tf.stack([v.size for v in values], axis))


def _oks_per_point(diff: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
  squared_distance = tf.reduce_sum(diff * diff, axis=-1)
  return tf.exp(tf.math.divide_no_nan(-squared_distance, 2. * scale * scale))


def box_displacement(location: tf.Tensor, box: BoundingBoxTensors) -> tf.Tensor:
  """Computes shift required to move the point into the box."""
  clipped = tf.clip_by_value(
      location,
      clip_value_min=box.min_corner[..., tf.newaxis, :],
      clip_value_max=box.max_corner[..., tf.newaxis, :])
  return clipped - location


def _mean(values: tf.Tensor,
          weights: Optional[tf.Tensor] = None,
          axis: int = -1) -> tf.Tensor:
  if weights is None:
    return tf.math.reduce_mean(values, axis=axis)
  else:
    return tf.math.divide_no_nan(
        tf.reduce_sum(values * weights, axis=axis),
        tf.reduce_sum(weights, axis=axis))


def object_keypoint_similarity(
    gt: KeypointsTensors,
    pr: KeypointsTensors,
    box: BoundingBoxTensors,
    per_type_scales: Collection[float],
    sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
  """Computes Object Keypoint Similarity scores.

  If ground truth keypoints are available (visibility > 0) it computes OKS with
  respect to ground truth keypoints and takes into account number of
  available keypoints. If a sample has no keypoints available (e.g. ground
  truth box has no keypoints labeled) then it computes a surrogate OKS using
  distances between keypoints and a box 3X larger than the ground trurh.

  For the detailed description refer to
  https://cocodataset.org/#keypoints-eval
  and to pycocotools for implementation details
  https://github.com/matteorr/coco-analyze/blob/9eb8a0a9e57ad1e592661efc2b8964864c0e6f28/pycocotools/cocoeval.py#L203

  Args:
    gt: ground truth keypoints.
    pr: predicted keypoints.
    box: ground truth box. Used to compute object scale (area or volume) and
      bound predicted keypoints too far away from the ground truth.
    per_type_scales: a list of num_points floats, with scale values for each
      keypoint type in the same order as keypoints stored in `gt` and `pr`.
    sample_weight: an optional weights for keypoints, a float tensor with shape
      [batch_size, num_points].

  Returns:
    a float tensor with shape [batch_size].
  """
  if gt.has_batch_dimension != pr.has_batch_dimension:
    raise ValueError(f'Batch dimension doesn\'t match: {gt} vs {pr}')
  if gt.has_batch_dimension != box.has_batch_dimension:
    raise ValueError(f'Batch dimension doesn\'t match: {gt} vs {box}')
  if gt.has_batch_dimension:
    scale = tf.constant([per_type_scales]) * tf.expand_dims(box.scale, axis=-1)
  else:
    scale = tf.constant(per_type_scales) * box.scale
  gt_diff = gt.location - pr.location
  # Enlarge the ground truth bounding box in size 3x to not penalize keypoints
  # predicted close enough to the ground truth box.
  no_penalty_box = BoundingBoxTensors(center=box.center, size=3 * box.size)
  box_diff = box_displacement(pr.location, box=no_penalty_box)
  mask = tf.cast(gt.visibility > 0, tf.float32)
  gt_available = (tf.math.reduce_sum(mask, axis=-1) > 0)
  diff = tf.where(gt_available[..., tf.newaxis, tf.newaxis], gt_diff, box_diff)
  per_point_scores = _oks_per_point(diff, scale)
  total_num_points = tf.constant(
      len(per_type_scales), dtype=tf.float32, shape=gt_available.shape)
  box_similarity = tf.math.divide_no_nan(
      tf.reduce_sum(per_point_scores, axis=-1), total_num_points)
  if sample_weight is None:
    weights = mask
  else:
    weights = mask * sample_weight
  gt_similarity = _mean(per_point_scores, weights=weights, axis=-1)
  return tf.where(gt_available, gt_similarity, box_similarity)


def _object_weights(per_keypoint_weights):
  if per_keypoint_weights is None:
    return None
  has_keypoints = tf.math.reduce_sum(per_keypoint_weights, axis=-1) > 0
  return tf.cast(has_keypoints, tf.float32)


class AveragePrecisionAtOKS(tf.keras.metrics.Metric):
  """Computes average precision at specified thresholds for OKS scores."""

  def __init__(self,
               thresholds: Collection[float],
               per_type_scales: Collection[float],
               precision_format: str = '{} P @ OKS>={}',
               average_precision_format: str = '{} AP',
               name: Optional[str] = None):
    """Creates all sub metrics.

    Args:
      thresholds: a list of thresholds for OKS to compute precision values.
        Usually generated using `tf.range(start=0.5, limit=1.0, delta=0.05)`
      per_type_scales: a list of scale values for each type of keypoint.
      precision_format: a format string for names for precision sub metrics.
        Meant to be used with `.format(name, threshold)`.
      average_precision_format: a format string for the averal precision
        metrics. Meant to be with `.format(name)`.
      name: an optional name for the metrics, which is also used as a part of
        all submetrics.
    """
    super().__init__(name=name)
    self._thresholds = thresholds
    self._per_type_scales = per_type_scales
    self._precision_format = precision_format
    self._average_precision_format = average_precision_format
    self._submetrics = self._create_metrics(self._thresholds)

  def _create_metrics(
      self,
      thresholds: Collection[float]) -> Dict[str, tf.keras.metrics.Metric]:
    metrics = {}
    for th in thresholds:
      name = self._precision_name(th)
      metrics[name] = tf.keras.metrics.Mean(name=f'{name}_precision')
    name = self._average_precision_name()
    metrics[name] = tf.keras.metrics.Mean(name=f'{name}_average_precision')
    return metrics

  def _precision_name(self, threshold: float) -> str:
    return self._precision_format.format(self.name, threshold)

  def _average_precision_name(self) -> str:
    return self._average_precision_format.format(self.name)

  def precision(self, threshold: float) -> tf.keras.metrics.Metric:
    """Returns a precision metric for the threshold."""
    return self._submetrics[self._precision_name(threshold)]

  def average_precision(self) -> tf.keras.metrics.Metric:
    """Returns the average precision metric."""
    return self._submetrics[self._average_precision_name()]

  def update_state(
      self,
      inputs: Tuple[KeypointsTensors, KeypointsTensors, BoundingBoxTensors],
      sample_weight: Optional[tf.Tensor] = None) -> List[tf.Operation]:
    """Updates all sub metrics using new inputs.

    Args:
      inputs: a tuple (ground truth keypoints, predicted keypoints, ground truth
        bounding box).
      sample_weight: an optional weights for keypoints, a float tensor with
        shape [batch_size, num_points].

    Returns:
      a list of update ops.
    """
    gt, pr, box = inputs
    oks = object_keypoint_similarity(
        gt, pr, box, self._per_type_scales, sample_weight=sample_weight)
    matches = []
    update_ops = []
    weight = _object_weights(sample_weight)
    for t in self._thresholds:
      m = tf.cast(oks > t, tf.float32)
      matches.append(m)
      update_ops.append(self.precision(t).update_state(m, sample_weight=weight))
    update_ops.append(self.average_precision().update_state(
        [_mean(m, weights=weight) for m in matches]))
    return update_ops

  def result(self) -> Dict[str, tf.Tensor]:
    """Returns a dictionary with scalar metric values.

    Returns:
      a dictionary with all sub metrics, where keys are metric names (formated
      using specified `precision_format` and `average_precision_format`) and
      values - scalar tensors.
    """
    return {n: m.result() for n, m in self._submetrics.items()}
