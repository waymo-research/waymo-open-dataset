# Copyright 2022 The Waymo Open Dataset Authors.
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

import abc
import dataclasses
from typing import Callable, Collection, Dict, List, Mapping, Optional, Tuple

import immutabledict
import tensorflow as tf

from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.python import matcher
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.utils import keypoint_data as _data


KeypointType = keypoint_pb2.KeypointType


def _oks_per_point(diff: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
  squared_distance = tf.reduce_sum(diff * diff, axis=-1)
  return tf.exp(tf.math.divide_no_nan(-squared_distance, 2. * scale * scale))


def box_displacement(location: tf.Tensor,
                     box: _data.BoundingBoxTensors) -> tf.Tensor:
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


def _reshape_weights(weights):
  """Optionally reshapes the weights tensor to have rank=2."""
  if weights.shape.rank == 1:
    return tf.expand_dims(weights, 1)
  elif weights.shape.rank == 2:
    return weights
  else:
    raise ValueError(f'Support only rank 1 or 2, got {weights.shape.rank}')


def _masked_weights(keypoint_mask: tf.Tensor,
                    sample_weight: Optional[tf.Tensor]) -> tf.Tensor:
  if sample_weight is None:
    return keypoint_mask
  return keypoint_mask * _reshape_weights(sample_weight)


def object_keypoint_similarity(
    gt: _data.KeypointsTensors,
    pr: _data.KeypointsTensors,
    box: _data.BoundingBoxTensors,
    per_type_scales: Collection[float],
    sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
  """Computes Object Keypoint Similarity scores.

  If ground truth keypoints are available (visibility > 0) it computes OKS with
  respect to ground truth keypoints and takes into account number of
  available keypoints. If a sample has no keypoints available (e.g. ground
  truth box has no keypoints labeled) then it computes a surrogate OKS using
  distances between keypoints and a box 3X larger than the ground truth.

  For the detailed description refer to
  https://cocodataset.org/#keypoints-eval
  and to pycocotools for implementation details
  https://github.com/matteorr/coco-analyze/blob/9eb8a0a9e57ad1e592661efc2b8964864c0e6f28/pycocotools/cocoeval.py#L203

  Args:
    gt: ground truth keypoints.
    pr: predicted keypoints.
    box: ground truth box. Used to compute object scale (area or volume) and
      bound predicted keypoints too far away from the ground truth (e.g.
      distance from a predicted keypoint is larger than 3x of the  object size,
      see the comment above).
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
  no_penalty_box = _data.BoundingBoxTensors(
      center=box.center, size=3 * box.size)
  box_diff = box_displacement(pr.location, box=no_penalty_box)
  mask = gt.mask
  gt_available = (tf.math.reduce_sum(mask, axis=-1) > 0)
  diff = tf.where(gt_available[..., tf.newaxis, tf.newaxis], gt_diff, box_diff)
  per_point_scores = _oks_per_point(diff, scale)
  total_num_points = tf.constant(
      len(per_type_scales), dtype=tf.float32, shape=gt_available.shape)
  box_similarity = tf.math.divide_no_nan(
      tf.reduce_sum(per_point_scores, axis=-1), total_num_points)
  weights = _masked_weights(mask, sample_weight)
  gt_similarity = _mean(per_point_scores, weights=weights, axis=-1)
  return tf.where(gt_available, gt_similarity, box_similarity)


def _object_weights(per_keypoint_weights):
  if per_keypoint_weights is None:
    return None
  has_keypoints = tf.math.reduce_sum(per_keypoint_weights, axis=-1) > 0
  return tf.cast(has_keypoints, tf.float32)


class KeypointsMetric(tf.keras.metrics.Metric, metaclass=abc.ABCMeta):
  """Interface for all keypoint metrics."""

  @abc.abstractmethod
  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None) -> List[tf.Operation]:
    """Updates all sub metrics using new inputs.

    Args:
      gt: ground truth keypoints.
      pr: predicted keypoints.
      box: ground truth bounding box.
      sample_weight: an optional weights for keypoints, a float tensor with
        shape [batch_size, num_points].

    Returns:
      a list of update ops.
    """

  @abc.abstractmethod
  def _result(self) -> Dict[str, tf.Tensor]:
    """Returns a dictionary with scalar metric values.

    Returns:
      a dictionary with all sub metrics, where keys are metric names (formatted
      using specified `precision_format` and `average_precision_format`) and
      values - scalar tensors.
    """

  # @final (uncomment after dropping support for PY<3.8)
  def update_state(
      self,
      inputs: Tuple[_data.KeypointsTensors, _data.KeypointsTensors,
                    _data.BoundingBoxTensors],
      sample_weight: Optional[tf.Tensor] = None) -> List[tf.Operation]:
    """Accumulates statistics for the metric.

    Shape descriptions below use the following notation:
      B - number of samples in the batch (aka batch_size).
      N - number of keypoints.
      D - number of dimensions (e.g. 2 or 3).

    Args:
      inputs: A tuple of 3 tensors (ground_truth, prediction, boxes), see their
        type definition for details.
      sample_weight: an optional tensor with shape [batch_size], which per
        object weights.

    Returns:
      A list of update ops.
    """
    gt, pr, box = inputs
    gt.location.shape.assert_is_compatible_with(pr.location.shape)
    gt.visibility.shape.assert_is_compatible_with(pr.visibility.shape)
    if box is not None:
      if gt.dims != box.dims:
        raise ValueError(f'Keypoints and box has diffferent dimensions: '
                         f'{gt.dims} != {box.dims}')
    return self._update_state(gt, pr, box, sample_weight)

  # @final (uncomment after dropping support for PY<3.8)
  def result(self) -> Dict[str, tf.Tensor]:
    """Returns a dictionary with scalar metric values.

    Returns:
      a dictionary with all sub metrics, where keys are metric names (formatted
      using specified `precision_format` and `average_precision_format`) and
      values - scalar tensors.
    """
    return self._result()


class _LayerDictWrapper:
  """A convenience wrapper for a collection of keras layers."""

  def __init__(self, layer_class, keys: Collection[float], name_format: str,
               name_format_args: Dict[str, str]):
    self.by_name = {}
    self.by_key = {}
    for key in keys:
      name = name_format.format(key=key, **name_format_args)
      layer = layer_class(name=name)
      self.by_name[name] = layer
      self.by_key[key] = layer


DEFAULT_OKS_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


class AveragePrecisionAtOKS(KeypointsMetric):
  """Computes average precision at specified thresholds for OKS scores."""

  def __init__(self,
               per_type_scales: Collection[float],
               thresholds: Collection[float] = DEFAULT_OKS_THRESHOLDS,
               precision_format: str = '{name} P @ {threshold:.2f}',
               average_precision_format: str = '{name} AP',
               name: Optional[str] = None):
    """Creates all sub metrics.

    Args:
      per_type_scales: a list of scale values for each type of keypoint.
      thresholds: a list of thresholds for OKS to compute precision values.
        Usually generated using `tf.range(start=0.5, limit=1.0, delta=0.05)`
      precision_format: a format string of precision sub metrics.
        Meant to be used with kwargs: name=<str>, threshold=<float>.
      average_precision_format: a format string for the average precision
        metrics. Meant to be used with kwargs: name=<str>.
      name: an optional name for the metrics, which is also used as a part of
        all submetrics.
    """
    super().__init__(name=name)
    self._thresholds = thresholds
    self._per_type_scales = per_type_scales
    self._precision_wrapper = _LayerDictWrapper(
        layer_class=tf.keras.metrics.Mean,
        keys=self._thresholds,
        name_format=precision_format.replace('threshold', 'key'),
        name_format_args={'name': self.name})
    # Assign wrapped metrics to a member attribute to enable attribute tracking
    # by Keras.
    self._precision_means = self._precision_wrapper.by_name
    self.average_precision = tf.keras.metrics.Mean(
        name=average_precision_format.format(name=self.name))

  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None) -> List[tf.Operation]:
    """See base class."""
    oks = object_keypoint_similarity(
        gt, pr, box, self._per_type_scales, sample_weight=sample_weight)
    matches = []
    update_ops = []
    weight = _object_weights(sample_weight)
    for threshold in self._thresholds:
      m = tf.cast(oks > threshold, tf.float32)
      matches.append(m)
      precision = self._precision_wrapper.by_key[threshold]
      update_ops.append(precision.update_state(m, sample_weight=weight))
    precision_means = [_mean(m, weights=weight) for m in matches]
    update_ops.append(self.average_precision.update_state(precision_means))
    return update_ops

  def _result(self) -> Dict[str, tf.Tensor]:
    """See base class."""
    metric_by_name = self._precision_wrapper.by_name
    results = {name: metric.result() for name, metric in metric_by_name.items()}
    results[self.average_precision.name] = self.average_precision.result()
    return results


class MeanPerJointPositionError(KeypointsMetric):
  """Computes mean per joint position error (aka MPJPE or MPJE).

  Resulting metric values have the same units as the location of input
  keypoints, e.g. the MPJE will be in pixels for camera keypoints and in meters
  for laser keypoints.
  """

  def __init__(self, name: Optional[str] = None):
    super().__init__(name=name)
    self._mean = tf.keras.metrics.Mean(name=f'{name}_mean')

  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None) -> List[tf.Operation]:
    """See base class."""
    position_error = tf.linalg.norm(gt.location - pr.location, axis=-1)
    weights = _masked_weights(gt.mask, sample_weight)
    update_op = self._mean.update_state(position_error, sample_weight=weights)
    return [update_op]

  def _result(self) -> Dict[str, tf.Tensor]:
    """See base class."""
    return {self.name: self._mean.result()}


class PoseEstimationMetric(KeypointsMetric):
  """Computes Pose Estimation Metric (PEM).

  It is a sum of the Mean Per Joint Position Error (MPJPE) over visible matched
  keypoints and a penalty for mismatched keypoints.

  Predicted and ground truth boxes are supposed to be already matched.
  Boxes and keypoints for missing ground truth or predicted objects have to be
  padded with zeros in order to have identical shapes. If a visible ground truth
  keypoint has a corresponding prediction with visibility set to zero (i.e.
  padded) it will be considered as a false negative keypoint detection. If a
  visible predicted keypoint has corresponding ground truth keypoints with
  visibility set to 0 or 1 (KeypointVisibility.is_occluded=True), it will be
  considered as as false positive.

  The evaluation service for the Pose Estimation challenge uses the Hungarian
  method based on Intersection over Union scores larger than 0.5 (a configurable
  threshold) to do the matching. But any other matching method could be used as
  well to pre-process the inputs for the PEM metric.
  """

  def __init__(self, mismatch_penalty: float, name: Optional[str] = None):
    """Initializes the metric.

    Args:
      mismatch_penalty: a value added for each mismatched keypoints.
      name: name of the metric.
    """
    super().__init__(name=name)
    self._mean = tf.keras.metrics.Mean(name=f'{name}_mean')
    self._mismatch_penalty = mismatch_penalty

  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None,
  ) -> List[tf.Operation]:
    """See base class."""
    error = tf.linalg.norm(gt.location - pr.location, axis=-1)
    gt_or_pr_mask = gt.is_fully_visible | pr.is_fully_visible
    gt_and_pr_mask = gt.is_fully_visible & pr.is_fully_visible
    # Set to zero errors for keypoints where `gt_or_pr_mask` is False.
    # Errors for keypoints with false positives and false negatives will be
    # replaced with the penalty later.
    error = tf.where(gt_or_pr_mask, error, tf.zeros_like(error))
    penalty = tf.fill(error.shape, self._mismatch_penalty)
    # Combine the error and the penalty. True positive keypoints have errors,
    # false positives and false negatives - the penalty.
    error_w_penalty = tf.where(gt_and_pr_mask, error, penalty)
    # The denominator for the metric is a sum of all weights and it will include
    # matched keypoints, false positives and false negatives.
    weights = _masked_weights(tf.cast(gt_or_pr_mask, tf.float32), sample_weight)
    update_op = self._mean.update_state(error_w_penalty, sample_weight=weights)
    return [update_op]

  def _result(self) -> Dict[str, tf.Tensor]:
    """See base class."""
    return {self.name: self._mean.result()}


class KeypointVisibilityPrecision(KeypointsMetric):
  """Computes Precision of keypoints visibility."""

  def __init__(self, name: Optional[str] = None):
    super().__init__(name=name)
    self._precision = tf.keras.metrics.Precision(name=f'{name}_precision')

  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None,
  ) -> List[tf.Operation]:
    """See base class."""
    del box
    # Predicted and ground truth boxes and corresponding keypoints are
    # supposed to be already matched. Keypoints for missing ground truth or
    # predicted objects has to be padded with zeros in order to have identical
    # shapes. The field `is_fully_visible` can have two values - True and False,
    # So precision of the keypoint detection is a precision of the binary
    # classification task.
    return [
        self._precision.update_state(
            gt.is_fully_visible, pr.is_fully_visible, sample_weight
        )
    ]

  def _result(self) -> Dict[str, tf.Tensor]:
    """See base class."""
    return {self.name: self._precision.result()}


class KeypointVisibilityRecall(KeypointsMetric):
  """Computes Recall of keypoints visibility."""

  def __init__(self, name: Optional[str] = None):
    super().__init__(name=name)
    self._recall = tf.keras.metrics.Recall(name=f'{name}_recall')

  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None,
  ) -> List[tf.Operation]:
    """See base class."""
    del box
    # Predicted and ground truth boxes and corresponding keypoints are
    # supposed to be already matched. Keypoints for missing ground truth or
    # predicted objects has to be padded with zeros in order to have identical
    # shapes. The field `is_fully_visible` can have two values - True and False,
    # So recall of the keypoint detection is a recall of the binary
    # classification task.
    return [
        self._recall.update_state(
            gt.is_fully_visible, pr.is_fully_visible, sample_weight
        )
    ]

  def _result(self) -> Dict[str, tf.Tensor]:
    """See base class."""
    return {self.name: self._recall.result()}


DEFAULT_PCK_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


class PercentageOfCorrectKeypoints(KeypointsMetric):
  """Ratio of number of points close to ground truth to total number of points."""

  def __init__(self,
               thresholds: Collection[float] = DEFAULT_PCK_THRESHOLDS,
               per_type_scales: Optional[Collection[float]] = None,
               use_object_scale: bool = False,
               metric_name_format: str = '{name} @ {threshold:.2f}',
               name: Optional[str] = None):
    """Initializes member variables.

    Absolute thresholds are determined in the following way:
      abs_threshold = thresholds [* per_type_scales [* box.scale]]
    where parts in [ ] are optional and depend on values of corresponding
    arguments.

    Args:
      thresholds: a list of float numbers to be used as the threshold, depending
        on the values of the other parameters. It could be an absolute threshold
        in meters or pixels, as well as a value relative to the object's or
        keypoint's scale.
      per_type_scales: an optional list of scale values for each type of
        keypoint. If specified will use keypoint specific scales to determine
        threshold values. The scales are absolute if use_object_scale=False and
        relative to the object's scale if use_object_scale=True.
      use_object_scale: if True, will use the object scale to determine
        threshold values. If it is False, values in `thresholds` and/or
         `per_type_scales` will be used to compute absolute thresholds
          regardless of the object scale. Object scale is defined as the square
          root from the box area for 2D case and the cubic root of the object
          volume in the 3D case.
      metric_name_format: a format string for PCK metric names at different
        thresholds. Meant to be used with kwargs: name=<str>,
        threshold=<float>.
      name: an optional name for the metric's name.
    """
    super().__init__(name=name)
    self._mean = tf.keras.metrics.Mean(name=f'{name}_mean')
    self._thresholds = thresholds
    self._use_object_scale = use_object_scale
    self._per_type_scales = per_type_scales
    self._pck_wrapper = _LayerDictWrapper(
        layer_class=tf.keras.metrics.Mean,
        keys=self._thresholds,
        name_format=metric_name_format.replace('threshold', 'key'),
        name_format_args={'name': self.name})
    # Assign wrapped metrics to a member attribute to enable attribute tracking
    # by Keras.
    self._pck_means = self._pck_wrapper.by_name

  def _keypoint_threshold(self, box: Optional[tf.Tensor],
                          threshold: float) -> tf.Tensor:
    """Returns a tensor with thresholds for all samples and keypoints."""
    if self._use_object_scale:
      if box is None:
        raise ValueError('box argument has to be not None when '
                         'use_object_scale=True')
      thresholds = (threshold * box.scale)[:, tf.newaxis]
    else:
      thresholds = tf.constant(threshold, shape=(1, 1), dtype=tf.float32)
    if self._per_type_scales is not None:
      thresholds *= tf.constant(
          self._per_type_scales,
          shape=(1, len(self._per_type_scales)),
          dtype=tf.float32)
    return thresholds

  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None) -> List[tf.Operation]:
    """See base class."""
    position_error = tf.linalg.norm(gt.location - pr.location, axis=-1)
    update_ops = []
    for threshold in self._thresholds:
      is_correct = tf.cast(
          position_error <= self._keypoint_threshold(box, threshold),  # pytype: disable=wrong-arg-types  # dynamic-method-lookup
          tf.float32)
      weights = _masked_weights(gt.mask, sample_weight)
      pck = self._pck_wrapper.by_key[threshold]
      update_ops.append(pck.update_state(is_correct, sample_weight=weights))
    return update_ops

  def _result(self) -> Dict[str, tf.Tensor]:
    """See base class."""
    metric_by_name = self._pck_wrapper.by_name
    return {name: metric.result() for name, metric in metric_by_name.items()}


def _merge_results(metrics):
  results = {}
  for m in metrics:
    results.update(m.result())
  return results


class MetricForSubsets(KeypointsMetric):
  """A wrapper for a collection of metrics for different subsets of keypoints."""

  def __init__(self,
               src_order: Collection['KeypointType'],
               subsets: Mapping[str, Collection['KeypointType']],
               create_metric_fn: Callable[[str], KeypointsMetric],
               name: Optional[str] = None):
    """Creates a collection of metrics.

    Args:
      src_order: Order of keypoint types in the input tensor.
      subsets: a dictionary with subsets of keypoint types to get finegrained
        metrics, e.g.
          {'HIPS': (KEYPOINT_TYPE_LEFT_HIP, KEYPOINT_TYPE_RIGHT_HIP)} will
            enable all metrics just for these two types. Keys in the dictionary
            are names of subsets - free form strings which could be used by
            `create_metric_fn` for formatting metric names and/or using a subset
            specific configuration settings.
      create_metric_fn: a factory function to create metric for each subset.
      name: an optional name for the metric's name.
    """
    super().__init__(name=name)
    self._src_order = src_order
    self._subsets = subsets
    self._metric_by_subset = {
        subset: create_metric_fn(subset) for subset in subsets.keys()
    }

  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None) -> List[tf.Operation]:
    """See base class."""
    update_ops = []
    for subset, dst_order in self._subsets.items():
      gt_subset = gt.subset(self._src_order, dst_order)
      pr_subset = pr.subset(self._src_order, dst_order)
      metric = self._metric_by_subset[subset]
      # TODO(gorban): support per_type_weights
      update_ops.extend(
          metric.update_state([gt_subset, pr_subset, box],  # pytype: disable=wrong-arg-types  # dynamic-method-lookup
                              sample_weight=sample_weight))
    return update_ops

  def _result(self) -> Dict[str, tf.Tensor]:
    """See base class."""
    return _merge_results(self._metric_by_subset.values())


class CombinedMetric(KeypointsMetric):
  """Merges results from all metrics into a single dictionary."""

  def __init__(self,
               child_metrics: Collection[KeypointsMetric],
               name: Optional[str] = None):
    """Wraps a collection of existing metrics.

    Args:
      child_metrics: a collection of existing metrics.
      name: an optional name for the metric's name.
    """
    super().__init__(name=name)
    self._child_metrics = child_metrics

  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None) -> List[tf.Operation]:
    """See base class."""
    update_ops = []
    for m in self._child_metrics:
      update_ops.extend(
          m.update_state([gt, pr, box], sample_weight=sample_weight))  # pytype: disable=wrong-arg-types  # dynamic-method-lookup
    return update_ops

  def _result(self):
    """See base class."""
    return _merge_results(self._child_metrics)


@dataclasses.dataclass(frozen=True)
class CombinedMetricsConfig:
  """Settings to create a combined metric with all supported keypoint metrics.

  Attributes:
    src_order: Order of keypoint types in the input tensor.
    subsets: a dictionary with subsets of keypoint types to get finegrained
      metrics, e.g. {'HIPS': (KEYPOINT_TYPE_LEFT_HIP, KEYPOINT_TYPE_RIGHT_HIP)}
        will enable all metrics just for these two types. Keys in the dictionary
        are names of subsets - free form strings which are used for formatting
        metric names and using a subset specific configuration settings.
    per_type_scales: a mapping (dictionary) between keypoint types and their
      scale factors.
    oks_thresholds: a collection of thresholds to compute OKS metrics.
    pck_thresholds: a collection of thresholds to compute PCK metrics.
  """
  src_order: Collection['KeypointType']
  subsets: Mapping[str, Collection['KeypointType']]
  per_type_scales: Mapping['KeypointType', float]
  oks_thresholds: Collection[float]
  pck_thresholds: Collection[float]

  def __post_init__(self):
    for i, kp_type in enumerate(self.src_order):
      if kp_type not in self.per_type_scales:
        name = KeypointType.Name(kp_type)
        raise ValueError(f'Keypoint type src_order{i} {name}({kp_type})'
                         f' is not found in {self.per_type_scales}')
    for subset, types in self.subsets.items():
      if not set(types).issubset(self.per_type_scales.keys()):
        raise ValueError(f'Subset {subset} with types {types} is not a subset '
                         f'of {self.per_type_scales.keys()}')

  def select_scales(self,
                    types: Collection['KeypointType']) -> Collection[float]:
    return [self.per_type_scales[t] for t in types]


# We maintain consistency of scales for corresponding types with `computeOks`
# from pycocotools package:
# https://github.com/matteorr/coco-analyze/blob/9eb8a0a9e57ad1e592661efc2b8964864c0e6f28/pycocotools/cocoeval.py#L216
# Note, the hardcoded constant in `computeOks` defines 5x scales in the
# following order:
#   nose, left_eye, right_eye, left_ear, right_ear, left_shoulder,
#   right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip,
#   right_hip, left_knee, right_knee, left_ankle, right_ankle
# https://github.com/matteorr/coco-analyze/blob/9eb8a0a9e57ad1e592661efc2b8964864c0e6f28/pycocotools/cocoanalyze.py#L928
# NOTE: There is no forehead or head center keypoint in COCO, so we used a
# similar method to determine scales for these keypoint types.
# NOTE: These scales are subject to change.
DEFAULT_PER_TYPE_SCALES = immutabledict.immutabledict({
    KeypointType.KEYPOINT_TYPE_NOSE: 0.052,
    KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER: 0.158,
    KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER: 0.158,
    KeypointType.KEYPOINT_TYPE_LEFT_ELBOW: 0.144,
    KeypointType.KEYPOINT_TYPE_RIGHT_ELBOW: 0.144,
    KeypointType.KEYPOINT_TYPE_LEFT_WRIST: 0.124,
    KeypointType.KEYPOINT_TYPE_RIGHT_WRIST: 0.124,
    KeypointType.KEYPOINT_TYPE_LEFT_HIP: 0.214,
    KeypointType.KEYPOINT_TYPE_RIGHT_HIP: 0.214,
    KeypointType.KEYPOINT_TYPE_LEFT_KNEE: 0.174,
    KeypointType.KEYPOINT_TYPE_RIGHT_KNEE: 0.174,
    KeypointType.KEYPOINT_TYPE_LEFT_ANKLE: 0.178,
    KeypointType.KEYPOINT_TYPE_RIGHT_ANKLE: 0.178,
    KeypointType.KEYPOINT_TYPE_FOREHEAD: 0.158,
    KeypointType.KEYPOINT_TYPE_HEAD_CENTER: 0.158
})

# Canonical groups of keypoints to report metrics.
_SHARED_GROUPS = immutabledict.immutabledict({
    'SHOULDERS': (KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER,
                  KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER),
    'ELBOWS': (KeypointType.KEYPOINT_TYPE_LEFT_ELBOW,
               KeypointType.KEYPOINT_TYPE_RIGHT_ELBOW),
    'WRISTS': (KeypointType.KEYPOINT_TYPE_LEFT_WRIST,
               KeypointType.KEYPOINT_TYPE_RIGHT_WRIST),
    'HIPS': (KeypointType.KEYPOINT_TYPE_LEFT_HIP,
             KeypointType.KEYPOINT_TYPE_RIGHT_HIP),
    'KNEES': (KeypointType.KEYPOINT_TYPE_LEFT_KNEE,
              KeypointType.KEYPOINT_TYPE_RIGHT_KNEE),
    'ANKLES': (KeypointType.KEYPOINT_TYPE_LEFT_ANKLE,
               KeypointType.KEYPOINT_TYPE_RIGHT_ANKLE)
})
CANONICAL_GROUPS_CAMERA = immutabledict.immutabledict(
    **_SHARED_GROUPS, **{
        'ALL':
            _data.CANONICAL_ORDER_CAMERA,
        'HEAD': (KeypointType.KEYPOINT_TYPE_NOSE,
                 KeypointType.KEYPOINT_TYPE_FOREHEAD)
    })
CANONICAL_GROUPS_LASER = immutabledict.immutabledict(
    **_SHARED_GROUPS, **{
        'ALL':
            _data.CANONICAL_ORDER_LASER,
        'HEAD': (KeypointType.KEYPOINT_TYPE_NOSE,
                 KeypointType.KEYPOINT_TYPE_HEAD_CENTER)
    })
CANONICAL_GROUPS_ALL = immutabledict.immutabledict(
    **_SHARED_GROUPS, **{
        'ALL':
            _data.CANONICAL_ORDER_ALL,
        'HEAD': (KeypointType.KEYPOINT_TYPE_NOSE,
                 KeypointType.KEYPOINT_TYPE_FOREHEAD,
                 KeypointType.KEYPOINT_TYPE_HEAD_CENTER)
    })

# Default configuration for models which output only camera keypoints.
DEFAULT_CONFIG_CAMERA = CombinedMetricsConfig(
    src_order=_data.CANONICAL_ORDER_CAMERA,
    subsets=CANONICAL_GROUPS_CAMERA,
    per_type_scales=DEFAULT_PER_TYPE_SCALES,
    oks_thresholds=DEFAULT_OKS_THRESHOLDS,
    pck_thresholds=DEFAULT_PCK_THRESHOLDS)

# Default configuration for models which output only laser keypoints.
DEFAULT_CONFIG_LASER = CombinedMetricsConfig(
    src_order=_data.CANONICAL_ORDER_LASER,
    subsets=CANONICAL_GROUPS_LASER,
    per_type_scales=DEFAULT_PER_TYPE_SCALES,
    oks_thresholds=DEFAULT_OKS_THRESHOLDS,
    pck_thresholds=DEFAULT_PCK_THRESHOLDS)

# Default configuration for models which output both camera and laser keypoints.
DEFAULT_CONFIG_ALL = CombinedMetricsConfig(
    src_order=_data.CANONICAL_ORDER_ALL,
    subsets=CANONICAL_GROUPS_ALL,
    per_type_scales=DEFAULT_PER_TYPE_SCALES,
    oks_thresholds=DEFAULT_OKS_THRESHOLDS,
    pck_thresholds=DEFAULT_PCK_THRESHOLDS)


def create_combined_metric(config: CombinedMetricsConfig) -> KeypointsMetric:
  """Creates a combined metric with all keypoint metrics.

  Args:
    config: See `CombinedMetricsConfig` for details. Use `DEFAULT_CONFIG_*`
      constants for default configurations to get a full set of all supported
      metrics.

  Returns:
    a keypoint metric which is a subclass of `tf.keras.metrics.Metric`, see
    `KeypointsMetric` for details.
  """

  def _create_mpjpe(subset: str) -> KeypointsMetric:
    return MeanPerJointPositionError(name=f'MPJPE/{subset}')

  def _create_oks(subset: str) -> KeypointsMetric:
    return AveragePrecisionAtOKS(
        thresholds=config.oks_thresholds,
        per_type_scales=config.select_scales(config.subsets[subset]),
        name=f'OKS/{subset}')

  def _create_pck(subset: str) -> KeypointsMetric:
    return PercentageOfCorrectKeypoints(
        thresholds=config.pck_thresholds,
        per_type_scales=config.select_scales(config.subsets[subset]),
        use_object_scale=True,
        name=f'PCK/{subset}')

  mpjpe = MetricForSubsets(
      src_order=config.src_order,
      subsets=config.subsets,
      create_metric_fn=_create_mpjpe)
  oks = MetricForSubsets(
      src_order=config.src_order,
      subsets=config.subsets,
      create_metric_fn=_create_oks)
  pck = MetricForSubsets(
      src_order=config.src_order,
      subsets=config.subsets,
      create_metric_fn=_create_pck)
  return CombinedMetric([mpjpe, pck, oks])


@dataclasses.dataclass(frozen=True)
class MatchingConfig:
  method: metrics_pb2.MatcherProto.Type = (
      metrics_pb2.MatcherProto.TYPE_HUNGARIAN
  )
  iou_threshold: float = 0.5


def _box_to_tensor(b: _data.BoundingBoxTensors) -> tf.Tensor:
  # Boxes should have the format CENTER_XYZ_DXDYDZ_PHI.
  return tf.concat([b.center, b.size, tf.expand_dims(b.heading, axis=-1)], -1)


def missing_ids(ids: tf.Tensor, count: int) -> tf.Tensor:
  """Returns a list of ids, missing in the input.

  Args:
    ids: input ids which are present, a tensor with shape [K].
    count: total number of ids.

  Returns:
    a tensor with ids, which are not present in the input.
    It has shape [count-K].
  """
  all_ids = tf.range(count, dtype=ids.dtype)
  # Shape of `all_mask` is [K]
  all_mask = tf.ones_like(ids) > 0
  # Shape of `mask` is [count]
  mask = tf.scatter_nd(ids[:, tf.newaxis], all_mask, shape=[count])
  # Shape of the returned tensor is [count - K].
  return tf.boolean_mask(all_ids, tf.logical_not(mask), axis=0)


def _reorder(tensor: tf.Tensor, *args: int | tf.Tensor) -> tf.Tensor:
  """Reorders slices of the tensor with optional paddings with zeros.

  Example:
    _reorder([0, 1, 2, 3, 4], [[1, 3], 2, [2, 4], 3]) will return
    [1, 3, 0, 0, 2, 4, 0, 0, 0]

  Args:
    tensor: input tensor with shape[N, ...].
    *args: a sequence with tensors or integers, where elements could be indices
      of the slices or number of elements to pad.

  Returns:
    a tensor with shape [sum(n_i), ...], where n_i is the length of ids tensor
    or corresponding padding.
  """
  parts = []
  for ids_or_num in args:
    if isinstance(ids_or_num, tf.Tensor):
      ids = ids_or_num
      parts.append(tf.gather(tensor, ids, axis=0))
    elif isinstance(ids_or_num, int):
      num = ids_or_num
      parts.append(tf.zeros((num,) + tensor.shape[1:], dtype=tensor.dtype))
    else:
      raise AssertionError('')
  return tf.concat(parts, axis=0)


def _reorder_objects(
    objects: _data.PoseEstimationTensors, *args: int | tf.Tensor
) -> _data.PoseEstimationTensors:
  """Reorders all tensors in the input dataclass according to the *args."""
  reorder = lambda t: _reorder(t, *args)
  return _data.PoseEstimationTensors(
      keypoints=_data.KeypointsTensors(
          location=reorder(objects.keypoints.location),
          visibility=reorder(objects.keypoints.visibility),
      ),
      box=_data.BoundingBoxTensors(
          center=reorder(objects.box.center),
          size=reorder(objects.box.size),
          heading=reorder(objects.box.heading),
      ),
  )


def match_pose_estimations(
    gt: _data.PoseEstimationTensors,
    pr: _data.PoseEstimationTensors,
    config: MatchingConfig = MatchingConfig(),
) -> Tuple[_data.PoseEstimationTensors, _data.PoseEstimationTensors]:
  """Reorders input tensors by matching bounding boxes of `gt` and `pr`.

  Shapes of output boxes and keypoints are [K, N, ...], where K is a sum of the
  number of matched and mismatched objects.

  Args:
    gt: a ground truth pose estimation tensors for all objects in a frame.
    pr: a predicted pose estimation tensors for all objects in a frame. Number
      of predicted objects is likely to be different from the number of ground
      truth objects.
    config: Configuration for the matching method.

  Returns:
    a 2-tuple (gt, pr) with reordered ground truth and prediction data
    compatible with `KeypointMetric` classes - shapes of all tensors in the
    corresponding dataclasses fields will match between `gt` and `pr`. For
    example: gt.keypoints.location.shape == pr.keypoints.location.shape
    and the 3D coordinate of the j-th keypoint for the i-th ground truth object
    will correspond to pr.keypoints.location[i, j].
  """
  # Run bipartite matching between prediction and groundtruth boxes.
  gt_box = _box_to_tensor(gt.box)
  pr_box = _box_to_tensor(pr.box)
  match_results = matcher.match(
      prediction_boxes=pr_box,
      groundtruth_boxes=gt_box,
      iou=config.iou_threshold,
      box_type=label_pb2.Label.Box.TYPE_3D,
      matcher_type=config.method,
  )
  gt_ids = match_results.groundtruth_ids
  pr_ids = match_results.prediction_ids
  # Indices of false negative ground truth objects.
  fn_ids = missing_ids(gt_ids, count=gt.box.center.shape[0])
  # Indices of false positive predicted objects.
  fp_ids = missing_ids(pr_ids, count=pr.box.center.shape[0])
  # Output ground truth order: matched objects, false negatives, zero padding.
  gt_m = _reorder_objects(gt, gt_ids, fn_ids, fp_ids.shape[0])
  # Output predictions order: matched objects, zero padding, false positives.
  pr_m = _reorder_objects(pr, pr_ids, fn_ids.shape[0], fp_ids)
  return (gt_m, pr_m)
