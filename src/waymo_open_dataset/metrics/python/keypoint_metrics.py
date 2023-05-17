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
"""Quality metrics for human keypoints.

Glossary:
  FN - false negative
  FP - false positive
  GT - ground truth
    GTi - ground truth box without visible keypoints (aka invisible).
    GTv - ground truth box with visible keypoints.
  OKS - Objective Keypoint Similarity
  PEM - Pose Estimation Metric
  PR - prediction
  TN - true negative
  TP - true positive
"""

import abc
import dataclasses
from typing import Callable, Collection, Dict, List, Mapping, Optional, Tuple

import immutabledict
from scipy import optimize
import tensorflow as tf

from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.python import matcher as _cpp_matcher
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.utils import keypoint_data as _data


KeypointType = keypoint_pb2.KeypointType


def _oks_per_point(diff: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
  squared_distance = tf.reduce_sum(diff * diff, axis=-1)
  return tf.exp(tf.math.divide_no_nan(-squared_distance, 2.0 * scale * scale))


def _get_heading(box: _data.BoundingBoxTensors) -> tf.Tensor:
  if box.heading is None:
    if box.has_batch_dimension:
      return tf.zeros([box.center.shape[0]], dtype=box.center.dtype)
    else:
      return tf.zeros([], dtype=box.center.dtype)
  else:
    return box.heading


def _to_box_transform(box: _data.BoundingBoxTensors) -> tf.Tensor:
  """Homogeneous matrices to transform points into the box's coordinate frame.

  We define the box's coordinate frame as following:
    - origin (O) is in the center of the box;
    - OX axis is oriented toward front face of the box;
    - OY oriented toward left side of the box;
    - OZ - up (same as before).
  Which means the box's heading will zero in this frame.

  Args:
    box: Shapes of the tensors for the box are described in the
      `BoundingBoxTensors` docstring. Batch dimension is optional.

  Returns:
   a 3x3 for 2d boxes or 4x4 matrices, a tensor with shape [..., 3, 3] or
   [..., 4, 4] accordingly
  """
  # All tensors below have shape: [batch_size] or []
  heading = _get_heading(box)
  cos = tf.math.cos(heading)
  sin = tf.math.sin(heading)
  zero = tf.zeros_like(heading)
  one = tf.ones_like(heading)
  cols = lambda e: tf.stack(e, axis=-1)
  rows = lambda e: tf.stack(e, axis=-2)
  cx = box.center[..., 0]
  cy = box.center[..., 1]
  tx = cos * cx - sin * cy
  ty = sin * cx + cos * cy
  if box.dims == 2:
    return rows([
        cols([cos, -sin, -tx]),
        cols([sin, cos, -ty]),
        cols([zero, zero, one]),
    ])
  elif box.dims == 3:
    tz = box.center[..., 2]
    return rows([
        cols([cos, -sin, zero, -tx]),
        cols([sin, cos, zero, -ty]),
        cols([zero, zero, one, -tz]),
        cols([zero, zero, zero, one]),
    ])
  else:
    raise AssertionError(f'Only 2D or 3D case is supported: got {box.dims}')


def _apply_transform(transform: tf.Tensor, location: tf.Tensor) -> tf.Tensor:
  """Applies homogenious transformation.

  Args:
    transform: a tensor with shape [batch_size, 3, 3] or [batch_size, 4, 4].
    location: a tensor with shape [batch_size, num_points, 2 or 3]

  Returns:
    a tensor with shape [batch_size, num_points, 2 or 3]
  """
  rotation = transform[..., tf.newaxis, :-1, :-1]
  translation = transform[..., tf.newaxis, :-1, -1]
  return tf.einsum('...ij,...j->...i', rotation, location) + translation


def box_displacement(
    location: tf.Tensor, box: _data.BoundingBoxTensors
) -> tf.Tensor:
  """Computes shift required to move the point into the box.

  For example, if points `a` are outside corresponding boxes `b`, points
  `a + box_displacement(a, b)` will be closest to the them points inside the
  boxes.

  Args:
    location: a tensor with shape [batch_size, num_points, 2 or 3].
    box: Shapes of the tensors for the box are described in the
      `BoundingBoxTensors` docstring.

  Returns:
    a tensor with shape [batch_size, num_points, 2 or 3].
  """
  # Move to the a coordinate frame related to the box to take into account
  # its heading (if specified).
  transform = _to_box_transform(box)
  # The location in the box's coordinate frame.
  p_box_frame = _apply_transform(transform, location)
  half_size = box.size[..., tf.newaxis, :] / 2
  # In the box coordinate frame a closest point inside the box can be found
  # simply by clipping the coordinates.
  p_box_clipped = tf.clip_by_value(p_box_frame, -half_size, half_size)
  # Compute the shift in the original coordinate frame.
  return _apply_transform(tf.linalg.inv(transform), p_box_clipped) - location


def _mean(
    values: tf.Tensor, weights: Optional[tf.Tensor] = None, axis: int = -1
) -> tf.Tensor:
  if weights is None:
    return tf.math.reduce_mean(values, axis=axis)
  else:
    return tf.math.divide_no_nan(
        tf.reduce_sum(values * weights, axis=axis),
        tf.reduce_sum(weights, axis=axis),
    )


def _reshape_weights(weights):
  """Optionally reshapes the weights tensor to have rank=2."""
  if weights.shape.rank == 1:
    return tf.expand_dims(weights, 1)
  elif weights.shape.rank == 2:
    return weights
  else:
    raise ValueError(f'Support only rank 1 or 2, got {weights.shape.rank}')


def _masked_weights(
    keypoint_mask: tf.Tensor, sample_weight: Optional[tf.Tensor]
) -> tf.Tensor:
  if sample_weight is None:
    return keypoint_mask
  return keypoint_mask * _reshape_weights(sample_weight)


def object_keypoint_similarity(
    gt: _data.KeypointsTensors,
    pr: _data.KeypointsTensors,
    box: _data.BoundingBoxTensors,
    per_type_scales: Collection[float],
    sample_weight: Optional[tf.Tensor] = None,
) -> tf.Tensor:
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
    raise ValueError(f"Batch dimension doesn't match: {gt} vs {pr}")
  if gt.has_batch_dimension != box.has_batch_dimension:
    raise ValueError(f"Batch dimension doesn't match: {gt} vs {box}")
  if gt.has_batch_dimension:
    scale = tf.constant([per_type_scales]) * tf.expand_dims(box.scale, axis=-1)
  else:
    scale = tf.constant(per_type_scales) * box.scale
  gt_diff = gt.location - pr.location
  # Enlarge the ground truth bounding box in size 3x to not penalize keypoints
  # predicted close enough to the ground truth box.
  no_penalty_box = _data.BoundingBoxTensors(
      center=box.center, size=3 * box.size
  )
  box_diff = box_displacement(pr.location, box=no_penalty_box)
  mask = gt.mask
  gt_available = tf.math.reduce_sum(mask, axis=-1) > 0
  diff = tf.where(gt_available[..., tf.newaxis, tf.newaxis], gt_diff, box_diff)
  per_point_scores = _oks_per_point(diff, scale)
  total_num_points = tf.constant(
      len(per_type_scales), dtype=tf.float32, shape=gt_available.shape
  )
  box_similarity = tf.math.divide_no_nan(
      tf.reduce_sum(per_point_scores, axis=-1), total_num_points
  )
  weights = _masked_weights(gt.mask * pr.mask, sample_weight)
  gt_similarity = _mean(per_point_scores, weights=weights, axis=-1)
  return tf.where(gt_available, gt_similarity, box_similarity)


def _oks_object_weights(box_weight, per_keypoint_weights):
  """Computes per object weights using per keypoint weights for OKS metric.

  Args:
    box_weight: is a float32 tensor with shape [batch_size].
    per_keypoint_weights: is a float32 tensor with shape [batch_size,
      num_points]. These are unnormalized weights of each keypoints.

  Returns:
    a tensor with shape [batch_size]. Normalized weights of each object for the
    purpose of OKS metric computation.
  """
  if per_keypoint_weights is not None:
    if per_keypoint_weights.shape.rank != 2:
      raise ValueError(
          f'Support only rank 2, got {per_keypoint_weights.shape.rank}'
      )
    # Object weight is proportional to the sum of weights of its keypoints.
    object_weights_abs = tf.reduce_sum(per_keypoint_weights, axis=-1)
    # All object weights add to 1.
    object_weights = tf.math.divide_no_nan(
        object_weights_abs,
        tf.reduce_sum(object_weights_abs, axis=-1, keepdims=True),
    )
    return object_weights * box_weight
  else:
    return box_weight


class KeypointsMetric(tf.keras.metrics.Metric, metaclass=abc.ABCMeta):
  """Interface for all keypoint metrics."""

  @abc.abstractmethod
  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None,
  ) -> List[tf.Operation]:
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
      inputs: Tuple[
          _data.KeypointsTensors,
          _data.KeypointsTensors,
          _data.BoundingBoxTensors,
      ],
      sample_weight: Optional[tf.Tensor] = None,
  ) -> List[tf.Operation]:
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
        raise ValueError(
            'Keypoints and box has diffferent dimensions: '
            f'{gt.dims} != {box.dims}'
        )
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

  def __init__(
      self,
      layer_class,
      keys: Collection[float],
      name_format: str,
      name_format_args: Dict[str, str],
  ):
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

  def __init__(
      self,
      per_type_scales: Collection[float],
      thresholds: Collection[float] = DEFAULT_OKS_THRESHOLDS,
      precision_format: str = '{name} P @ {threshold:.2f}',
      average_precision_format: str = '{name} AP',
      name: Optional[str] = None,
  ):
    """Creates all sub metrics.

    Args:
      per_type_scales: a list of scale values for each type of keypoint.
      thresholds: a list of thresholds for OKS to compute precision values.
        Usually generated using `tf.range(start=0.5, limit=1.0, delta=0.05)`
      precision_format: a format string of precision sub metrics. Meant to be
        used with kwargs: name=<str>, threshold=<float>.
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
        name_format_args={'name': self.name},
    )
    # Assign wrapped metrics to a member attribute to enable attribute tracking
    # by Keras.
    self._precision_means = self._precision_wrapper.by_name
    self.average_precision = tf.keras.metrics.Mean(
        name=average_precision_format.format(name=self.name)
    )

  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None,
  ) -> List[tf.Operation]:
    """See base class."""
    oks = object_keypoint_similarity(
        gt, pr, box, self._per_type_scales, sample_weight=sample_weight
    )
    box_is_valid = tf.reduce_any(box.size > 1e-5, axis=-1)
    oks_valid = tf.where(box_is_valid, oks, tf.zeros_like(oks))
    matches = []
    update_ops = []
    weight = _oks_object_weights(
        tf.cast(box_is_valid, tf.float32), sample_weight
    )
    for threshold in self._thresholds:
      m = tf.cast(oks_valid > threshold, tf.float32)
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
      sample_weight: Optional[tf.Tensor] = None,
  ) -> List[tf.Operation]:
    """See base class."""
    position_error = tf.linalg.norm(gt.location - pr.location, axis=-1)
    weights = _masked_weights(gt.mask * pr.mask, sample_weight)
    update_op = self._mean.update_state(position_error, sample_weight=weights)
    return [update_op]

  def _result(self) -> Dict[str, tf.Tensor]:
    """See base class."""
    return {self.name: self._mean.result()}


def _pose_estimation_error(
    gt: _data.KeypointsTensors,
    pr: _data.KeypointsTensors,
    mismatch_penalty: float,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Computes per-point pose estimation error for `PoseEstimationMetric`.

  Shapes of the tensors for the keypoints (`gt` and `pr`) are described in the
  `KeypointsTensors` docstring. First dimension represents multiple objects
  within a single frame/scene (i.e. `n_obj`, not an actual batch).

  This function assumes that gt[i] object corresponds to pr[i] object and uses
  keypoint visibility to determine if corresponding keypoints are matched. A
  keypoint will be considered as:
   - true positive (tp) if it is visible in gt and pr;
   - true negative (tn) - not visible in gt and pr;
   - false positive (fp) - predicted, but not visible in gt;
   - false negative (fn) - not predicted, but visible in gt.

  Args:
    gt: ground keypoints for `n_obj` objects, each has `n_points` keypoints.
    pr: predicted keypoints for `n_obj` objects, each has `n_points` keypoints.
    mismatch_penalty: a value added for each mismatched keypoints.

  Returns:
    a 2-tuple (error, mask), where both tensors have shape [n_obj, n_points].
    The error - is the per-point pose estimation error, and the mask has True
    values for tp, fp and fn points.
  """
  error = tf.linalg.norm(gt.location - pr.location, axis=-1)
  error = tf.clip_by_value(error, 0, mismatch_penalty)
  tn_mask = tf.logical_not(gt.is_fully_visible | pr.is_fully_visible)
  tp_mask = gt.is_fully_visible & pr.is_fully_visible
  # Set to zero errors for keypoints where `tn_mask` is True.
  # Errors for keypoints with false positives and false negatives will be
  # replaced with the penalty later.
  error = tf.where(tn_mask, tf.zeros_like(error), error)
  penalty = tf.fill(error.shape, mismatch_penalty)
  # Combine the error and the penalty. True positive keypoints have errors,
  # false positives and false negatives - the penalty.
  error_w_penalty = tf.where(tp_mask | tn_mask, error, penalty)
  return error_w_penalty, tf.logical_not(tn_mask)


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
    error_w_penalty, tp_fp_fn_mask = _pose_estimation_error(
        gt, pr, self._mismatch_penalty
    )
    weights = _masked_weights(tf.cast(tp_fp_fn_mask, tf.float32), sample_weight)
    # The denominator for the metric is a sum of all weights and it will include
    # matched keypoints, false positives and false negatives.
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

  def __init__(
      self,
      thresholds: Collection[float] = DEFAULT_PCK_THRESHOLDS,
      per_type_scales: Optional[Collection[float]] = None,
      use_object_scale: bool = False,
      metric_name_format: str = '{name} @ {threshold:.2f}',
      name: Optional[str] = None,
  ):
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
        `per_type_scales` will be used to compute absolute thresholds regardless
        of the object scale. Object scale is defined as the square root from the
        box area for 2D case and the cubic root of the object volume in the 3D
        case.
      metric_name_format: a format string for PCK metric names at different
        thresholds. Meant to be used with kwargs: name=<str>, threshold=<float>.
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
        name_format_args={'name': self.name},
    )
    # Assign wrapped metrics to a member attribute to enable attribute tracking
    # by Keras.
    self._pck_means = self._pck_wrapper.by_name

  def _keypoint_threshold(
      self, box: Optional[tf.Tensor], threshold: float
  ) -> tf.Tensor:
    """Returns a tensor with thresholds for all samples and keypoints."""
    if self._use_object_scale:
      if box is None:
        raise ValueError(
            'box argument has to be not None when use_object_scale=True'
        )
      thresholds = (threshold * box.scale)[:, tf.newaxis]
    else:
      thresholds = tf.constant(threshold, shape=(1, 1), dtype=tf.float32)
    if self._per_type_scales is not None:
      thresholds *= tf.constant(
          self._per_type_scales,
          shape=(1, len(self._per_type_scales)),
          dtype=tf.float32,
      )
    return thresholds

  def _update_state(
      self,
      gt: _data.KeypointsTensors,
      pr: _data.KeypointsTensors,
      box: _data.BoundingBoxTensors,
      sample_weight: Optional[tf.Tensor] = None,
  ) -> List[tf.Operation]:
    """See base class."""
    position_error = tf.linalg.norm(gt.location - pr.location, axis=-1)
    update_ops = []
    for threshold in self._thresholds:
      is_correct = tf.cast(
          position_error <= self._keypoint_threshold(box, threshold),  # pytype: disable=wrong-arg-types  # dynamic-method-lookup
          tf.float32,
      )
      weights = _masked_weights(gt.mask * pr.mask, sample_weight)
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

  def __init__(
      self,
      src_order: Collection['KeypointType'],
      subsets: Mapping[str, Collection['KeypointType']],
      create_metric_fn: Callable[[str], KeypointsMetric],
      name: Optional[str] = None,
  ):
    """Creates a collection of metrics.

    Args:
      src_order: Order of keypoint types in the input tensor.
      subsets: a dictionary with subsets of keypoint types to get finegrained
        metrics, e.g. {'HIPS': (KEYPOINT_TYPE_LEFT_HIP,
        KEYPOINT_TYPE_RIGHT_HIP)} will enable all metrics just for these two
        types. Keys in the dictionary are names of subsets - free form strings
        which could be used by `create_metric_fn` for formatting metric names
        and/or using a subset specific configuration settings.
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
      sample_weight: Optional[tf.Tensor] = None,
  ) -> List[tf.Operation]:
    """See base class."""
    update_ops = []
    for subset, dst_order in self._subsets.items():
      gt_subset = gt.subset(self._src_order, dst_order)
      pr_subset = pr.subset(self._src_order, dst_order)
      metric = self._metric_by_subset[subset]
      # TODO(gorban): support per_type_weights
      update_ops.extend(
          metric.update_state(
              [gt_subset, pr_subset, box],  # pytype: disable=wrong-arg-types  # dynamic-method-lookup
              sample_weight=sample_weight,
          )
      )
    return update_ops

  def _result(self) -> Dict[str, tf.Tensor]:
    """See base class."""
    return _merge_results(self._metric_by_subset.values())


class CombinedMetric(KeypointsMetric):
  """Merges results from all metrics into a single dictionary."""

  def __init__(
      self,
      child_metrics: Collection[KeypointsMetric],
      name: Optional[str] = None,
  ):
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
      sample_weight: Optional[tf.Tensor] = None,
  ) -> List[tf.Operation]:
    """See base class."""
    update_ops = []
    for m in self._child_metrics:
      update_ops.extend(
          m.update_state((gt, pr, box), sample_weight=sample_weight)
      )  # pytype: disable=wrong-arg-types  # dynamic-method-lookup
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
        raise ValueError(
            f'Keypoint type src_order{i} {name}({kp_type})'
            f' is not found in {self.per_type_scales}'
        )
    for subset, types in self.subsets.items():
      if not set(types).issubset(self.per_type_scales.keys()):
        raise ValueError(
            f'Subset {subset} with types {types} is not a subset '
            f'of {self.per_type_scales.keys()}'
        )

  def select_scales(
      self, types: Collection['KeypointType']
  ) -> Collection[float]:
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
    KeypointType.KEYPOINT_TYPE_HEAD_CENTER: 0.158,
})

# Canonical groups of keypoints to report metrics.
_SHARED_GROUPS = immutabledict.immutabledict({
    'SHOULDERS': (
        KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER,
        KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER,
    ),
    'ELBOWS': (
        KeypointType.KEYPOINT_TYPE_LEFT_ELBOW,
        KeypointType.KEYPOINT_TYPE_RIGHT_ELBOW,
    ),
    'WRISTS': (
        KeypointType.KEYPOINT_TYPE_LEFT_WRIST,
        KeypointType.KEYPOINT_TYPE_RIGHT_WRIST,
    ),
    'HIPS': (
        KeypointType.KEYPOINT_TYPE_LEFT_HIP,
        KeypointType.KEYPOINT_TYPE_RIGHT_HIP,
    ),
    'KNEES': (
        KeypointType.KEYPOINT_TYPE_LEFT_KNEE,
        KeypointType.KEYPOINT_TYPE_RIGHT_KNEE,
    ),
    'ANKLES': (
        KeypointType.KEYPOINT_TYPE_LEFT_ANKLE,
        KeypointType.KEYPOINT_TYPE_RIGHT_ANKLE,
    ),
})
CANONICAL_GROUPS_CAMERA = immutabledict.immutabledict(
    **_SHARED_GROUPS,
    **{
        'ALL': _data.CANONICAL_ORDER_CAMERA,
        'HEAD': (
            KeypointType.KEYPOINT_TYPE_NOSE,
            KeypointType.KEYPOINT_TYPE_FOREHEAD,
        ),
    },
)
CANONICAL_GROUPS_LASER = immutabledict.immutabledict(
    **_SHARED_GROUPS,
    **{
        'ALL': _data.CANONICAL_ORDER_LASER,
        'HEAD': (
            KeypointType.KEYPOINT_TYPE_NOSE,
            KeypointType.KEYPOINT_TYPE_HEAD_CENTER,
        ),
    },
)
CANONICAL_GROUPS_ALL = immutabledict.immutabledict(
    **_SHARED_GROUPS,
    **{
        'ALL': _data.CANONICAL_ORDER_ALL,
        'HEAD': (
            KeypointType.KEYPOINT_TYPE_NOSE,
            KeypointType.KEYPOINT_TYPE_FOREHEAD,
            KeypointType.KEYPOINT_TYPE_HEAD_CENTER,
        ),
    },
)

# Default configuration for models which output only camera keypoints.
DEFAULT_CONFIG_CAMERA = CombinedMetricsConfig(
    src_order=_data.CANONICAL_ORDER_CAMERA,
    subsets=CANONICAL_GROUPS_CAMERA,
    per_type_scales=DEFAULT_PER_TYPE_SCALES,
    oks_thresholds=DEFAULT_OKS_THRESHOLDS,
    pck_thresholds=DEFAULT_PCK_THRESHOLDS,
)

# Default configuration for models which output only laser keypoints.
DEFAULT_CONFIG_LASER = CombinedMetricsConfig(
    src_order=_data.CANONICAL_ORDER_LASER,
    subsets=CANONICAL_GROUPS_LASER,
    per_type_scales=DEFAULT_PER_TYPE_SCALES,
    oks_thresholds=DEFAULT_OKS_THRESHOLDS,
    pck_thresholds=DEFAULT_PCK_THRESHOLDS,
)

# Default configuration for models which output both camera and laser keypoints.
DEFAULT_CONFIG_ALL = CombinedMetricsConfig(
    src_order=_data.CANONICAL_ORDER_ALL,
    subsets=CANONICAL_GROUPS_ALL,
    per_type_scales=DEFAULT_PER_TYPE_SCALES,
    oks_thresholds=DEFAULT_OKS_THRESHOLDS,
    pck_thresholds=DEFAULT_PCK_THRESHOLDS,
)


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
        name=f'OKS/{subset}',
    )

  def _create_pck(subset: str) -> KeypointsMetric:
    return PercentageOfCorrectKeypoints(
        thresholds=config.pck_thresholds,
        per_type_scales=config.select_scales(config.subsets[subset]),
        use_object_scale=True,
        name=f'PCK/{subset}',
    )

  mpjpe = MetricForSubsets(
      src_order=config.src_order,
      subsets=config.subsets,
      create_metric_fn=_create_mpjpe,
  )
  oks = MetricForSubsets(
      src_order=config.src_order,
      subsets=config.subsets,
      create_metric_fn=_create_oks,
  )
  pck = MetricForSubsets(
      src_order=config.src_order,
      subsets=config.subsets,
      create_metric_fn=_create_pck,
  )
  return CombinedMetric([mpjpe, pck, oks])


CppMatcherMethod = metrics_pb2.MatcherProto.Type


class MatcherConfig:
  """Base class for all configation dataclasses for matchers.

  It is currently used just for type annotation.
  """


@dataclasses.dataclass(frozen=True)
class CppMatcherConfig(MatcherConfig):
  """Configuration for the `MeanErrorMatcher`.

  Attributes:
    method: type of box matching algorithm implemented by the C++ Op to use.
    iou_threshold: Minimal IoU of ground truth and predicted boxes to consider
      them to be a match.
  """

  method: CppMatcherMethod = CppMatcherMethod.TYPE_HUNGARIAN
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
  keypoints = _data.KeypointsTensors(
      location=reorder(objects.keypoints.location),
      visibility=reorder(objects.keypoints.visibility),
  )
  if objects.box is None:
    box = None
  else:
    box = _data.BoundingBoxTensors(
        center=reorder(objects.box.center),
        size=reorder(objects.box.size),
        heading=reorder(objects.box.heading),
    )
  return _data.PoseEstimationTensors(keypoints=keypoints, box=box)


@dataclasses.dataclass(frozen=True)
class PoseEstimationPair:
  """A pair of ground truth and predicted pose estimation tensors.

  Attributes:
    gt: a ground truth pose estimation tensors for all objects in a frame.
    pr: a predicted pose estimation tensors for all objects in a frame. Number
      of predicted objects is likely to be different from the number of ground
      truth objects.
  """

  gt: _data.PoseEstimationTensors
  pr: _data.PoseEstimationTensors


@dataclasses.dataclass(frozen=True)
class MatchingIdsPairs:
  """A pair of ground truth and predicted ids.

  Attributes:
    gt: 1D tensor with n_matches elements.
    pr: 1D tensor with n_matches elements.
  """

  gt: tf.Tensor
  pr: tf.Tensor


def _indices_of_false(values: tf.Tensor) -> tf.Tensor:
  """Returns indices of False elements in `values`."""
  # Same as tf.squeeze(tf.cast(tf.where(~values), dtype=tf.int32), axis=1)
  # but with less hustle.
  indices = tf.range(values.shape[0], dtype=tf.int32)
  return tf.boolean_mask(indices, ~values)


def _select_ids(ids: MatchingIdsPairs, gt_mask: tf.Tensor) -> MatchingIdsPairs:
  """Returns a subset of ids for which gt_mask is True."""
  gt_ids_mask = tf.gather(gt_mask, ids.gt)
  return MatchingIdsPairs(
      gt=tf.boolean_mask(ids.gt, gt_ids_mask),
      pr=tf.boolean_mask(ids.pr, gt_ids_mask),
  )


def _num_objects(t: _data.PoseEstimationTensors) -> int:
  """Returns number of objects in the `PoseEstimationTensor`."""
  return t.keypoints.location.shape[0]


class BaseMatcher(metaclass=abc.ABCMeta):
  """Abstract base class for matchers."""

  @abc.abstractmethod
  def matching_ids(
      self,
      pose: PoseEstimationPair,
  ) -> MatchingIdsPairs:
    """Returns indexes of the ground truth and corresponding predictions."""

  def reorder(self, pose: PoseEstimationPair) -> PoseEstimationPair:
    """Reorders input `pose` tensors.

    Shapes of output boxes and keypoints are [K, N, ...], where K is a sum of
    the number of matched and mismatched objects.

    Args:
      pose: Input pose estimation pair.

    Returns:
      a PoseEstimationPair with reordered ground truth and prediction data
      compatible with `KeypointMetric` classes - shapes of all tensors in the
      corresponding dataclasses fields will match between `gt` and `pr`. For
      example: gt.keypoints.location.shape == pr.keypoints.location.shape
      and the 3D coordinate of the j-th keypoint for the i-th ground truth
      object
      will correspond to pr.keypoints.location[i, j].
    """
    assert pose.gt.keypoints is not None
    assert pose.gt.keypoints.has_visible is not None
    # Indices of both GTi and GTv matching with PR.
    m_ids = self.matching_ids(pose)
    m_visible_ids = _select_ids(m_ids, pose.gt.keypoints.has_visible)
    # Both matched GTv and GTi keypoints ids.
    matched_or_invisible_gt_ids = tf.concat(
        [m_ids.gt, _indices_of_false(pose.gt.keypoints.has_visible)],
        axis=-1,
    )
    matched_or_invisible_pr_ids = tf.concat(
        [m_ids.pr, _indices_of_false(pose.pr.keypoints.has_visible)],
        axis=-1,
    )
    # Exclude GTi from the list of missing GT.
    fn_ids = missing_ids(
        matched_or_invisible_gt_ids, count=_num_objects(pose.gt)
    )
    # Indices of false positive predicted objects.
    fp_ids = missing_ids(
        matched_or_invisible_pr_ids, count=_num_objects(pose.pr)
    )
    # Output ground truth order: matched objects, false negatives, padding.
    gt_m = _reorder_objects(pose.gt, m_visible_ids.gt, fn_ids, fp_ids.shape[0])
    # Output predictions order: matched objects, padding, false positives.
    pr_m = _reorder_objects(pose.pr, m_visible_ids.pr, fn_ids.shape[0], fp_ids)
    return PoseEstimationPair(gt_m, pr_m)


class CppMatcher(BaseMatcher):
  """A matcher which uses C++ Ops to match bounding boxes."""

  def __init__(self, config: CppMatcherConfig = CppMatcherConfig()):
    self._config = config

  def matching_ids(self, pose: PoseEstimationPair) -> MatchingIdsPairs:
    """Uses IoU of bounding boxes and C++ Op to do the matching."""
    # Run bipartite matching between prediction and groundtruth boxes.
    gt_box = _box_to_tensor(pose.gt.box)
    pr_box = _box_to_tensor(pose.pr.box)
    ids = _cpp_matcher.match(
        prediction_boxes=pr_box,
        groundtruth_boxes=gt_box,
        iou=self._config.iou_threshold,
        box_type=label_pb2.Label.Box.TYPE_3D,
        matcher_type=self._config.method,
    )
    return MatchingIdsPairs(
        gt=ids.groundtruth_ids,
        pr=ids.prediction_ids,
    )


def keypoint_distance(
    boxes: _data.BoundingBoxTensors, keypoints: _data.KeypointsTensors
) -> tf.Tensor:
  """Computes distance between boxes and all keypoints.

  We compute the distance between all boxes and all objects (described as a set
  of keypoints). The distance is defined as the minimal moving distance for the
  box to have at least one keypoint to be inside it.

  Args:
    boxes: tensors for `n_boxes` boxes. Shapes of the tensors for the box are
      described in the `BoundingBoxTensors` docstring. First dimension
      represents multiple boxes within a single frame/scene (i.e. `n_boxes`, not
      an actual batch).
    keypoints: tensors for `n_objs` objects with `n_points` keypoints each.
      Shapes of the tensors for the keypoints are described in the
      `KeypointsTensors` docstring. First dimension represents multiple objects
      within a single frame/scene (i.e. `n_objs`, not an actual batch).

  Returns:
    pairwise keypoint distances between boxes and objects, a tensor with
    shape [n_boxes, n_objs, n_points].
  """
  n_boxes = boxes.center.shape[0]
  n_objs, n_points, _ = keypoints.location.shape
  # Shape: [n_boxes, n_objs * n_points, 3]
  locations = tf.broadcast_to(
      tf.reshape(keypoints.location, [1, n_objs * n_points, 3]),
      (n_boxes, n_objs * n_points, 3),
  )
  dist_shape = (n_boxes, n_objs, n_points)
  # If we add the displacement vector to a point it will move it into the box.
  displacement = box_displacement(locations, boxes)
  # Size of the displacement vector is the distance from a point to a closest
  # side of the box.
  point_dist = tf.reshape(tf.linalg.norm(displacement, axis=-1), dist_shape)
  is_visible = tf.broadcast_to(
      tf.reshape(keypoints.is_fully_visible, [1, n_objs, n_points]), dist_shape
  )
  positive_inf = tf.fill(dist_shape, float('inf'))
  return tf.where(is_visible, point_dist, positive_inf)


def closest_keypoint_distance(
    boxes: _data.BoundingBoxTensors, keypoints: _data.KeypointsTensors
) -> tf.Tensor:
  """Computes distance between boxes and closest keypoints."""
  dist = keypoint_distance(boxes, keypoints)
  return tf.reduce_min(dist, axis=-1)


def _gather_keypoints(
    kp: _data.KeypointsTensors, indices: tf.Tensor, axis: int = 0
) -> _data.KeypointsTensors:
  """Selects slices from tensors in `KeypointsTensors` given their indices."""
  return _data.KeypointsTensors(
      location=tf.gather(kp.location, indices, axis=axis),
      visibility=tf.gather(kp.visibility, indices, axis=axis),
  )


def _hungarian_assignment(
    cost: tf.Tensor,
    maximize: bool,
    filter_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
) -> MatchingIdsPairs:
  """Finds an assignment to minimize or maximize the total cost.

  It's a wrapper for linear_sum_assignment in SciPy.
  NOTE: It is not differentiable.

  Args:
    cost: an [M, N] matrix.
    maximize: If True it will output an assignment to maximize total cost.
    filter_fn: an optional filter function, only assignments for which it
      returns True will be returned. Input parameter for the function is a float
      tensor with shape [K], where K=min(M,N) - assignment cost for each pair.
      Output is a boolean tensor with the same shape.

  Returns:
    a 2-tuple (lhs, rhs), where `lhs_ids' and 'rhs` are indices which minimize
    cost of the assignment. Both are tensors with shape [L], where L is
    min(M,N) if `filter_fn` is None or the number of True values returned by the
    `filter_fn`.
  """
  fn = lambda t: optimize.linear_sum_assignment(t.numpy(), maximize=maximize)
  # Shape of both `lhs` and `rhs` is [K], where K=min(M,N).
  lhs, rhs = tf.py_function(func=fn, inp=[cost], Tout=(tf.int32, tf.int32))
  if filter_fn is not None:
    # shape: [n_matches, 2]
    ids = tf.stack([lhs, rhs], axis=-1)
    # shape: [n_matches]
    assignments_cost = tf.gather_nd(cost, ids)
    mask = filter_fn(assignments_cost)
    lhs = tf.boolean_mask(lhs, mask)
    rhs = tf.boolean_mask(rhs, mask)
  return MatchingIdsPairs(gt=lhs, pr=rhs)


# Some value in meters which should be larger than distance between any actual
# predicted keypoints and ground truth objects.
_VERY_LARGE_ERROR_M = 1e10
# A tiny value in meters to account for numerical issues.
_EPS_M = 1e-5


def mean_error_matching(
    gt: _data.KeypointsTensors,
    pr: _data.KeypointsTensors,
    candidates_mask: tf.Tensor,
    mismatch_penalty: float,
) -> MatchingIdsPairs:
  """Matches objects using Hungarian algorithm to minimize total mean error.

  The error between PR and GT truth keypoints for each candidate match is
  computed using the same method as in `PoseEstimationMetric` (PEM) - it is
  equal to L2 distance between matching 3D keypoints (clipped by `max_distance`)
  and is set to `max_distance` for unmatched keypoints. This way output matches
  will minimize the PEM.

  Shapes of the tensors for `gt` and `pr` keypoints are described in the
  `KeypointsTensors` docstring. First dimension represents multiple objects
  within a single frame/scene (not an actual batch).

  Args:
    gt: ground keypoints for `n_gt_obj` objects, each has `n_points` keypoints.
    pr: predicted keypoints for `n_pr_obj` objects, each has `n_points`
      keypoints.
    candidates_mask: a boolean matrix with shape [n_gt_obj, n_pr_obj]. True
      values correspond to potential matching objects.
    mismatch_penalty: A value in meters used in two ways - as the max distance
      between keypoints will be clipped to and a penalty for mismatched
      keypoints.

  Returns:
    Ids of matching pairs.
  """
  n_pr_obj = pr.visibility.shape[0]
  # Special case, where there is no predicted objects for a frame.
  if n_pr_obj == 0:
    return MatchingIdsPairs(
        gt=tf.zeros([0], dtype=tf.int32), pr=tf.zeros([0], dtype=tf.int32)
    )
  n_gt_obj, n_points = gt.visibility.shape

  gt_ids = tf.range(n_gt_obj, dtype=tf.int32)
  pr_ids_all = tf.range(n_pr_obj, dtype=tf.int32)

  # Compute point wise distances only for candidates defined via the mask.
  # shape: [n_gt_objs, n_pr_objs, 2]
  indices = tf.stack(tf.meshgrid(gt_ids, pr_ids_all, indexing='ij'), axis=-1)
  # shape: [n_candidates, 2]
  cand_ids = tf.boolean_mask(indices, candidates_mask)
  cand_gt = _gather_keypoints(gt, cand_ids[:, 0])
  cand_pr = _gather_keypoints(pr, cand_ids[:, 1])
  # Compute mean distance for all keypoints within each object.
  # shape: [n_candidates, n_points]

  point_error, _ = _pose_estimation_error(cand_gt, cand_pr, mismatch_penalty)
  # shape: [n_gt_objs, n_pr_objs, n_points]
  # Initialize the matrix of pairwise errors with the `_VERY_LARGE_ERROR_M`.
  large_error = tf.fill([n_gt_obj, n_pr_obj, n_points], _VERY_LARGE_ERROR_M)
  # Set match candidates to have actual error values.
  obj_point_error = tf.tensor_scatter_nd_update(
      large_error, cand_ids, point_error
  )
  # NOTE: we compute the mean along the axis=-1 and there could be only two
  # cases with respect to values in a slice `obj_point_error[i, j, :]`
  #  1. all are equal to `_VERY_LARGE_ERROR_M`
  #  2. none of them are equal to `_VERY_LARGE_ERROR_M`
  # which guaranties that we never compute mean over regular and the large
  # placeholder values `_VERY_LARGE_ERROR_M`.
  # shape: [n_gt_objs, n_pr_objs]
  mean_error = tf.clip_by_value(
      tf.reduce_mean(obj_point_error, axis=-1), 0, mismatch_penalty
  )
  # Lists of ground truth and corresponding prediction ids.
  # shape: [n_gt_objs]
  return _hungarian_assignment(
      mean_error,
      maximize=False,
      filter_fn=lambda x: x < mismatch_penalty,
  )


@dataclasses.dataclass(frozen=True)
class MeanErrorMatcherConfig(MatcherConfig):
  """Configuration for the `MeanErrorMatcher`.

  Refer to `MeanErrorMatcher` docstring for details of the algorithm and
  the file's docstrings for meaning of the acronyms used in the description.

  Attributes:
    max_closest_keypoint_distance: is a max distance from any PR keypoint from a
      GT box to be considered as a candidate.
  """

  max_closest_keypoint_distance: float = 0.25


def _ids_to_mask(ids: tf.Tensor, count: int) -> tf.Tensor:
  """Creates a boolean mask with True values corresponding to the ids."""
  return tf.scatter_nd(
      ids[:, tf.newaxis], tf.fill([ids.shape[0]], True), shape=[count]
  )


def _mask_of_used_ids(m: MatchingIdsPairs, shape: tf.TensorShape) -> tf.Tensor:
  """Returns a boolean mask with True values in rows/cols used in gt/pr."""
  n_gt, n_pr = shape
  gt_rows = tf.broadcast_to(_ids_to_mask(m.gt, n_gt)[:, tf.newaxis], shape)
  pr_cols = tf.broadcast_to(_ids_to_mask(m.pr, n_pr)[tf.newaxis, :], shape)
  return pr_cols | gt_rows


def _is_gti_and_not_gtv_close_enough(
    gt: _data.KeypointsTensors,
    pr: _data.KeypointsTensors,
    max_closest_keypoint_distance: float,
) -> tf.Tensor:
  """Returns a [n_gt_obj, n_pr_obj] mask, with True values for candidates."""
  # shape [n_gt_obj, n_pr_obj, n_points]
  per_point_dist = tf.linalg.norm(
      gt.location[:, tf.newaxis, :, :] - pr.location[tf.newaxis, :, :, :],
      axis=-1,
  )
  # shape [n_gt_obj, n_pr_obj, n_points]
  gt_is_near = per_point_dist < max_closest_keypoint_distance
  # shape [n_gt_obj, 1, n_points]
  gtv_is_near = tf.reduce_any(
      gt.is_fully_visible[:, tf.newaxis, :]
      & pr.is_fully_visible[tf.newaxis, :, :]
      & gt_is_near,
      axis=-1,
  )
  # shape: [n_gt_obj, 1]
  is_gti = ~tf.reduce_any(gt.is_fully_visible, axis=-1, keepdims=True)
  return ~tf.reduce_any(gtv_is_near, axis=0, keepdims=True) & is_gti


class MeanErrorMatcher(BaseMatcher):
  """The official matcher for the PEM metric.

  For information on the PEM metric refer to
  https://waymo.com/open/challenges/2023/pose-estimation/.

  It considers all pairs of PR and GT objects, for which at least one PR
  keypoint is within the GT box enlarged by a certain distance. For each
  candidate it computes the error using the same method as used by
  `PoseEstimationMetric` and selects a pair with smallest mean
  error as a match.

  For the purpose of matching GT to PR objects we consider two types of GT
  boxes:
    - GT boxes without any visible keypoint (GTi for short)
    - GT boxes with at least one visible keypoint (GTv for short)

  The matching algorithms has two stages:
  1. Maximize number of PR keypoints in GTi to exclude them from the set of
    candidate matches because we do not penalize PR keypoints for objects for
    which there is no visible GT keypoints.
  2. Find an assignment between PR and GT objects which minimizes the PEM.
  """

  def __init__(self, config: MeanErrorMatcherConfig = MeanErrorMatcherConfig()):
    self._config = config

  def matching_ids(self, pose: PoseEstimationPair) -> MatchingIdsPairs:
    """See base class."""
    assert pose.gt.box is not None
    assert pose.pr.keypoints is not None
    # shape: [n_gt_obj, n_pr_obj, n_points]
    dist = keypoint_distance(pose.gt.box, pose.pr.keypoints)
    # shape: [n_gt_obj, n_pr_obj]
    closest_dist = tf.reduce_min(dist, axis=-1)
    closest_mask = tf.less_equal(
        closest_dist, self._config.max_closest_keypoint_distance
    )
    # shape: [n_gt_obj, n_pr_obj]
    n_points_inside = tf.reduce_sum(
        tf.cast(dist < _EPS_M, dtype=tf.float32), axis=-1
    )
    # Stage 1: Exclude GTi matches.
    # - Count number of PR keypoints inside GTi boxes, excluding GTv boxes which
    #   have at least one GT keypoint closer than the
    #   `max_closest_keypoint_distance`.
    # - Use Hungarian algorithm to find an assignment between the GTi boxes and
    #   PR objects, to maximize the total number of keypoints inside GTi boxes.
    n_point_score = tf.where(
        _is_gti_and_not_gtv_close_enough(
            pose.gt.keypoints,
            pose.pr.keypoints,
            self._config.max_closest_keypoint_distance,
        ),
        n_points_inside,
        tf.zeros_like(n_points_inside, dtype=tf.float32),
    )
    m_all = _hungarian_assignment(
        n_point_score,
        maximize=True,
        filter_fn=lambda n_points: n_points > 0,
    )
    m_all_gtv = tf.gather(pose.gt.keypoints.has_visible, m_all.gt)
    m_stage1 = MatchingIdsPairs(
        gt=tf.boolean_mask(m_all.gt, ~m_all_gtv),
        pr=tf.boolean_mask(m_all.pr, ~m_all_gtv),
    )
    # shape: [n_gt_obj, n_pr_obj]
    gti_excluded_mask = _mask_of_used_ids(m_stage1, shape=dist.shape[:2])
    # NOTE: We use the `max_closest_keypoint_distance` in two ways:
    #  - as a penalty for mismatched keypoints.
    #  - to determine a set of candidate matches between predicted and ground
    #    truth objects. It makes no sense to match objects which are
    #    further away from each other than this distance because all their
    #    keypoints will be marked as mismatched and the mismatched penalty will
    #    be applied.

    # Stage 2: Determine the final assignment.
    m_stage2 = mean_error_matching(
        pose.gt.keypoints,
        pose.pr.keypoints,
        candidates_mask=closest_mask & ~gti_excluded_mask,
        mismatch_penalty=self._config.max_closest_keypoint_distance,
    )
    return MatchingIdsPairs(
        gt=tf.concat([m_stage2.gt, m_stage1.gt], axis=-1),
        pr=tf.concat([m_stage2.pr, m_stage1.pr], axis=-1),
    )


# TODO(gorban): refactor to add a factory function to the config objects.
def create_matcher(config: MatcherConfig) -> BaseMatcher:
  """A factory function to create matchers."""
  if isinstance(config, CppMatcherConfig):
    return CppMatcher(config)
  elif isinstance(config, MeanErrorMatcherConfig):
    return MeanErrorMatcher(config)
  else:
    raise AssertionError(f'Unsupported matcher: {config=}')
