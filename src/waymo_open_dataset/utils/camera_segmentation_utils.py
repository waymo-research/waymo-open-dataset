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
"""Utilities for camera panoptic segmentation labels."""
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import immutabledict
import numpy as np
from skimage import segmentation
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
from waymo_open_dataset.v2.perception import segmentation as v2

LabelProtoOrComponent = Union[
    dataset_pb2.CameraSegmentationLabel, v2.CameraSegmentationLabelComponent
]
InstanceIDToGlobalIDMapping = (
    dataset_pb2.CameraSegmentationLabel.InstanceIDToGlobalIDMapping
)


# RGB colors used to visualize each semantic segmentation class.
SEGMENTATION_COLOR_MAP = immutabledict.immutabledict({
    cs_pb2.CameraSegmentation.TYPE_UNDEFINED: [0, 0, 0],
    cs_pb2.CameraSegmentation.TYPE_EGO_VEHICLE: [102, 102, 102],
    cs_pb2.CameraSegmentation.TYPE_CAR: [0, 0, 142],
    cs_pb2.CameraSegmentation.TYPE_TRUCK: [0, 0, 70],
    cs_pb2.CameraSegmentation.TYPE_BUS: [0, 60, 100],
    cs_pb2.CameraSegmentation.TYPE_OTHER_LARGE_VEHICLE: [61, 133, 198],
    cs_pb2.CameraSegmentation.TYPE_BICYCLE: [119, 11, 32],
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLE: [0, 0, 230],
    cs_pb2.CameraSegmentation.TYPE_TRAILER: [111, 168, 220],
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN: [220, 20, 60],
    cs_pb2.CameraSegmentation.TYPE_CYCLIST: [255, 0, 0],
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLIST: [180, 0, 0],
    cs_pb2.CameraSegmentation.TYPE_BIRD: [127, 96, 0],
    cs_pb2.CameraSegmentation.TYPE_GROUND_ANIMAL: [91, 15, 0],
    cs_pb2.CameraSegmentation.TYPE_CONSTRUCTION_CONE_POLE: [230, 145, 56],
    cs_pb2.CameraSegmentation.TYPE_POLE: [153, 153, 153],
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN_OBJECT: [234, 153, 153],
    cs_pb2.CameraSegmentation.TYPE_SIGN: [246, 178, 107],
    cs_pb2.CameraSegmentation.TYPE_TRAFFIC_LIGHT: [250, 170, 30],
    cs_pb2.CameraSegmentation.TYPE_BUILDING: [70, 70, 70],
    cs_pb2.CameraSegmentation.TYPE_ROAD: [128, 64, 128],
    cs_pb2.CameraSegmentation.TYPE_LANE_MARKER: [234, 209, 220],
    cs_pb2.CameraSegmentation.TYPE_ROAD_MARKER: [217, 210, 233],
    cs_pb2.CameraSegmentation.TYPE_SIDEWALK: [244, 35, 232],
    cs_pb2.CameraSegmentation.TYPE_VEGETATION: [107, 142, 35],
    cs_pb2.CameraSegmentation.TYPE_SKY: [70, 130, 180],
    cs_pb2.CameraSegmentation.TYPE_GROUND: [102, 102, 102],
    cs_pb2.CameraSegmentation.TYPE_DYNAMIC: [102, 102, 102],
    cs_pb2.CameraSegmentation.TYPE_STATIC: [102, 102, 102],
})


def decode_semantic_and_instance_labels_from_panoptic_label(
    panoptic_label: np.ndarray,
    panoptic_label_divisor: int) -> Tuple[np.ndarray, np.ndarray]:
  """Converts a panoptic label into semantic and instance segmentation labels.

  Args:
    panoptic_label: A 2D array where each pixel is encoded as: semantic_label *
      panoptic_label_divisor + instance_label.
    panoptic_label_divisor: an int used to encode the panoptic labels.

  Returns:
    A tuple containing the semantic and instance labels, respectively.
  """
  if panoptic_label_divisor <= 0:
    raise ValueError('panoptic_label_divisor must be > 0.')
  return np.divmod(panoptic_label, panoptic_label_divisor)


def encode_semantic_and_instance_labels_to_panoptic_label(
    semantic_label: np.ndarray, instance_label: np.ndarray,
    panoptic_label_divisor: int) -> np.ndarray:
  """Encodes a semantic and instance label into a panoptic label.

  Encoding uses the equation:
  panoptic_label = semantic_label * panoptic_label_divisor + instance_label

  Note that the output of this function will be cast to uint32 to prevent
  overflow issues.

  Args:
    semantic_label: A 2D numpy array containing the semantic class for each
      pixel.
    instance_label: A 2D numpy array containing the instance ID for each pixel.
    panoptic_label_divisor: An int specifying the separation between semantic
      classes. Must be greater than the maximum instance ID.
  Returns:
    A 2D uint32 numpy array containing the panoptic label for each pixel.
  """
  return np.add(
      np.multiply(np.array(semantic_label).astype(np.uint32),
                  panoptic_label_divisor),
      np.array(instance_label).astype(np.uint32))


def _iterate_over_mapping(
    label: LabelProtoOrComponent,
) -> Iterator[InstanceIDToGlobalIDMapping]:
  """Helper function to replicate v1 iteration over id mappings in v2."""
  if isinstance(label, v2.CameraSegmentationLabelComponent):
    mapping = label.instance_id_to_global_id_mapping
    for local_id, global_id, is_tracked in zip(
        mapping.local_instance_ids,
        mapping.global_instance_ids,
        mapping.is_tracked,
    ):
      yield InstanceIDToGlobalIDMapping(
          local_instance_id=local_id,
          global_instance_id=global_id,
          is_tracked=is_tracked,
      )
  elif isinstance(label, dataset_pb2.CameraSegmentationLabel):
    yield from label.instance_id_to_global_id_mapping
  else:
    raise ValueError('Input label format not supported.')


def _remap_global_ids(
    segmentation_proto_list: Sequence[LabelProtoOrComponent],
    remap_to_sequential: bool = False,
) -> Dict[str, Dict[int, int]]:
  """Remaps the global ids in segmentation_proto_list to consecutive values.

  Given protos from different sequences, this function will remap the global
  ids such that ids are consecutive, while ids from different sequences stay
  different. This can be used to reduce the set of global ids to a minimal set
  of unique instance IDs.

  Args:
    segmentation_proto_list: a sequence of CameraSegmentationLabel protos.
    remap_to_sequential: whether to remap the instance IDs to sequential values.
      This is not recommended for eval.

  Returns:
    A dict of dicts, mapping, for each sequence, from the original global
      instance ids to the remapped ids. i.e. {sequence: {orig_id : remapped id}}
  """
  # Read the set of global ids for each sequence.
  global_instance_ids = {}
  for label in segmentation_proto_list:
    if not label.instance_id_to_global_id_mapping:
      continue
    if label.sequence_id not in global_instance_ids:
      global_instance_ids[label.sequence_id] = set()

    frame_global_instance_ids = [
        mapping.global_instance_id for mapping in _iterate_over_mapping(label)
    ]
    global_instance_ids[label.sequence_id].update(frame_global_instance_ids)

  if not global_instance_ids:
    return {}

  # Offset the global ids in each sequence so that each sequence has a different
  # set of ids.
  cumulative_max_instance_id = 0
  global_instance_ids_offset = {}
  for sequence in global_instance_ids:
    max_instance_id = max(global_instance_ids[sequence])
    global_instance_ids_offset[sequence] = [
        inst_id + cumulative_max_instance_id
        for inst_id in global_instance_ids[sequence]
    ]
    cumulative_max_instance_id += max_instance_id

  # Remap the global ids to consecutive values and return as a dict.
  all_global_ids_offset = sum(global_instance_ids_offset.values(), [])
  if remap_to_sequential:
    global_ids_remapped, _, _ = segmentation.relabel_sequential(
        np.array(all_global_ids_offset))
  else:
    global_ids_remapped = all_global_ids_offset

  all_global_ids = []
  all_sequences = []
  for sequence in global_instance_ids:
    all_global_ids.extend(list(global_instance_ids[sequence]))
    all_sequences.extend([sequence for _ in global_instance_ids[sequence]])

  # A dict of: {sequence: {original_global_id: remapped_global_id}}
  global_id_remapping = {}
  for i in range(len(all_sequences)):
    if all_sequences[i] not in global_id_remapping:
      global_id_remapping[all_sequences[i]] = {}
    global_id_remapping[all_sequences[i]][all_global_ids[i]] = (
        global_ids_remapped[i])
  return global_id_remapping


def decode_single_panoptic_label_from_proto(
    segmentation_proto: LabelProtoOrComponent,
) -> np.ndarray:
  """Decodes a panoptic label from a CameraSegmentationLabel.

  Args:
    segmentation_proto: a CameraSegmentationLabel to be decoded.

  Returns:
    A 2D numpy array containing the per-pixel panoptic segmentation label.
  """
  return tf.io.decode_png(
      segmentation_proto.panoptic_label, dtype=tf.uint16).numpy()


def decode_multi_frame_panoptic_labels_from_protos(
    segmentation_proto_list: Sequence[dataset_pb2.CameraSegmentationLabel],
    remap_to_global: bool = True,
    remap_to_sequential: bool = False,
    new_panoptic_label_divisor: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
  warnings.warn(
      'This function is deprecated, please call '
      'decode_multi_frame_panoptic_labels_from_segmentation_labels '
      'instead.'
  )
  return decode_multi_frame_panoptic_labels_from_segmentation_labels(
      segmentation_proto_list,
      remap_to_global,
      remap_to_sequential,
      new_panoptic_label_divisor,
  )


def decode_multi_frame_panoptic_labels_from_segmentation_labels(
    segmentation_proto_list: Sequence[LabelProtoOrComponent],
    remap_to_global: bool = True,
    remap_to_sequential: bool = False,
    new_panoptic_label_divisor: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
  """Parses a set of panoptic labels with consistent instance ids from labels.

  This functions supports an arbitrary number of CameraSegmentationLabels,
  and can remap values both within the same and between different sequences.

  Both protos and components are supported by this function.

  Args:
    segmentation_proto_list: a sequence of CameraSegmentationLabel protos.
    remap_to_global: if true, will remap the instance ids using the
      instance_id_to_global_id_mapping such that they are consistent within each
      sequence, and that they are consecutive between all output labels.
    remap_to_sequential: whether to remap the instance IDs to sequential values.
      This is not recommended for eval.
    new_panoptic_label_divisor: if provided, the output panoptic label will be
      shifted to this new panoptic label divisor.

  Returns:
    A tuple containing
      panoptic_labels: a list of uint32 numpy arrays, containing the parsed
        panoptic labels. If any labels have instance_id_to_global_id_mappings,
        instances with these mappings will be mapped to the same value across
        frames.
      num_cameras_covered: a list of uint8 numpy arrays, containing the parsed
        arrays representing the number of cameras covered for each pixel. This
        is used to compute the weighted Segmentation and Tracking Quality (wSTQ)
        metric.
      is_tracked_masks: a list of uint8 numpy arrays, where a pixel is True if
        its instance is tracked over multiple frames.
      panoptic_label_divisor: the int32 divisor used to generate the panoptic
        labels.

  Raises:
    ValueError: if the output panoptic_label_divisor is lower than the maximum
      instance id.
  """
  # Use the existing panoptic_label_divisor if possible to maintain consistency
  # in the data and keep the panoptic labels more comprehensible.
  if new_panoptic_label_divisor is None:
    panoptic_label_divisor = segmentation_proto_list[0].panoptic_label_divisor
  else:
    panoptic_label_divisor = new_panoptic_label_divisor

  if remap_to_global:
    global_id_mapping = _remap_global_ids(
        segmentation_proto_list, remap_to_sequential=remap_to_sequential)
    if global_id_mapping:
      max_instance_id = max(
          [max([global_id for _, global_id in mapping.items()])
           for _, mapping in global_id_mapping.items()])
      if new_panoptic_label_divisor is None:
        panoptic_label_divisor = max(panoptic_label_divisor, max_instance_id)

  panoptic_labels = []
  num_cameras_covered = []
  is_tracked_masks = []
  for label in segmentation_proto_list:
    sequence = label.sequence_id
    panoptic_label = decode_single_panoptic_label_from_proto(label)
    semantic_label, instance_label = (
        decode_semantic_and_instance_labels_from_panoptic_label(
            panoptic_label, label.panoptic_label_divisor))
    camera_coverage = np.ones_like(instance_label, dtype=np.uint8)
    is_tracked_mask = np.zeros_like(instance_label, dtype=np.uint8)

    instance_label_copy = np.copy(instance_label)
    if remap_to_global:
      if isinstance(label, v2.CameraSegmentationLabelComponent):
        mapping_iter = _iterate_over_mapping(label)
      elif isinstance(label, dataset_pb2.CameraSegmentationLabel):
        mapping_iter = label.instance_id_to_global_id_mapping
      else:
        raise ValueError('Input label format not supported.')

      for mapping in mapping_iter:
        instance_mask = (instance_label == mapping.local_instance_id)
        is_tracked_mask[instance_mask] = mapping.is_tracked
        instance_label_copy[instance_mask] = global_id_mapping[sequence][
            mapping.global_instance_id]

    if np.amax(instance_label) >= panoptic_label_divisor:
      raise ValueError('A panoptic_label_divisor of '
                       f'{panoptic_label_divisor} is requested, but the '
                       'maximum instance id exceeds this.')

    if label.num_cameras_covered:
      camera_coverage = tf.io.decode_png(label.num_cameras_covered).numpy()

    panoptic_labels.append(
        encode_semantic_and_instance_labels_to_panoptic_label(
            semantic_label, instance_label_copy, panoptic_label_divisor))
    num_cameras_covered.append(camera_coverage)
    is_tracked_masks.append(is_tracked_mask)

  return (
      panoptic_labels,
      is_tracked_masks,
      num_cameras_covered,
      panoptic_label_divisor,
  )


def _generate_color_map(
    color_map_dict: Optional[
        Mapping[int, Sequence[int]]] = None
) -> np.ndarray:
  """Generates a mapping from segmentation classes (rows) to colors (cols).

  Args:
    color_map_dict: An optional dict mapping from semantic classes to colors. If
      None, the default colors in SEGMENTATION_COLOR_MAP will be used.
  Returns:
    A np array of shape [max_class_id + 1, 3], where each row encodes the color
      for the corresponding class id.
  """
  if color_map_dict is None:
    color_map_dict = SEGMENTATION_COLOR_MAP
  classes = list(color_map_dict.keys())
  colors = list(color_map_dict.values())
  color_map = np.zeros([np.amax(classes) + 1, 3], dtype=np.uint8)
  color_map[classes] = colors
  return color_map

DEFAULT_COLOR_MAP = _generate_color_map()


def semantic_label_to_rgb(
    semantic_label: np.ndarray,
    color_map: np.ndarray = DEFAULT_COLOR_MAP) -> np.ndarray:
  """Maps a semantic label to a color RGB image.

  Args:
    semantic_label: A 2D array with integer type, where each pixel stores its
      semantic class.
    color_map: A [num_classes, 3] uint8 array where each row encodes the color
      for the corresponding class id.

  Returns:
    A 3D array where the last dimension is the RGB color for each pixel.
  """
  if semantic_label.shape[-1] == 1:
    semantic_label = np.squeeze(semantic_label, axis=-1)
  return color_map[semantic_label]


def panoptic_label_to_rgb(semantic_label: np.ndarray,
                          instance_label: np.ndarray,
                          color_map: np.ndarray = DEFAULT_COLOR_MAP,
                          prime_num: int = 41,
                          max_delta: int = 50) -> np.ndarray:
  """Maps a panoptic label to a color RGB image.

  Semantic labels are mapped to colors via semantic_label_to_rgb. Instances are
  then differentiated by offsetting the green channel of the semantic color
  values by the instance_label scaled by a prime number.

  Args:
    semantic_label: A [height, width] integer array storing the semantic
      segmentation label.
    instance_label: A [height, width] integer array, storing the instance
      segmentation label.
    color_map: A [num_classes, 3] uint8 array, where each row encodes the color
      for the corresponding class id.
    prime_num: An integer used to generate noise in the color map to visualize
      different instances.
    max_delta: An integer used to control the noise level added to the instance
      colors.

  Returns:
    A 3D array with uint8 type. Each element in the array is the color indexed
      by the corresponding element in the input.
  """
  if semantic_label.shape[-1] == 1:
    semantic_label = np.squeeze(semantic_label, axis=-1)
  if instance_label.shape[-1] == 1:
    instance_label = np.squeeze(instance_label, axis=-1)

  color_image = semantic_label_to_rgb(semantic_label, color_map)
  # Visualize the instance IDs in the green channel, by offsetting the semantic
  # color values by the instance_label scaled by a prime number.
  ins_noise = instance_label * prime_num % (max_delta * 2) - max_delta
  color_image[:, :, 1] = np.where(instance_label > 0,
                                  color_image[:, :, 1] + ins_noise,
                                  color_image[:, :, 1])
  color_image = np.clip(color_image, 0, 255)
  return color_image


def save_panoptic_label_to_proto(
    panoptic_label: np.ndarray, panoptic_label_divisor: int,
    sequence_id: str, new_panoptic_label_divisor: int = 1000
) -> dataset_pb2.CameraSegmentationLabel:
  """Converts a panoptic prediction to a CameraSegmentationLabel proto.

  The instance segmentation labels will be remapped to sequential values, with
  the mapping stored in the instance_id_to_global_id_mapping.

  Args:
    panoptic_label: a [H, W, 1] ndarray containing the predicted panoptic
      segmentation.
    panoptic_label_divisor: the panoptic_label_divisor used to encode the input
      panoptic_prediciton.
    sequence_id: the sequence id corresponding to this panoptic label.
    new_panoptic_label_divisor: the panoptic label divisor used for the encoded
      CameraSegmentationLabel proto. If the number of instances exceeds this
      value, the maximum instance id + 1 will be used instead.

  Returns:
    The input panoptic prediction encoded into a CameraSegmentationLabel proto.

  Raises:
    ValueError: if the maximum output panoptic label is greater than the max
      uint16 value.
  """
  semantic_label, instance_label = (
      decode_semantic_and_instance_labels_from_panoptic_label(
          panoptic_label, panoptic_label_divisor))
  instance_label_shape = instance_label.shape
  instance_label_flat = np.reshape(instance_label, [-1])
  # Remap the instance labels to sequential values to make the range of instance
  # ids as compact as possible.
  instance_label_sequential, _, inverse_map = segmentation.relabel_sequential(
      instance_label_flat)
  instance_label_sequential = tf.reshape(
      instance_label_sequential, instance_label_shape)
  # Update the new_panoptic_label_divisor if the maximum number of instances
  # exceeds the desired one.
  new_panoptic_label_divisor = max(
      new_panoptic_label_divisor,
      tf.reduce_max(instance_label_sequential) + 1)
  panoptic_label = encode_semantic_and_instance_labels_to_panoptic_label(
      semantic_label, instance_label_sequential, new_panoptic_label_divisor)
  if np.amax(panoptic_label) >= np.iinfo(np.uint16).max:
    raise ValueError(
        f'The maximum panoptic label of {new_panoptic_label_divisor} would '
        'overflow a uint16. Consider lowering the panoptic_label_divisor or '
        'reducing the number of instances.')
  encoded_panoptic_label = tf.io.encode_png(
      tf.cast(panoptic_label, tf.uint16)).numpy()
  seg_proto = dataset_pb2.CameraSegmentationLabel(
      panoptic_label_divisor=new_panoptic_label_divisor,
      sequence_id=sequence_id,
      panoptic_label=encoded_panoptic_label,
  )
  # Store the mapping back to the original instance ids.
  for remapped, orig in zip(inverse_map.in_values, inverse_map.out_values):
    # No need to map the unknown class.
    if orig == 0 or remapped == 0:
      continue
    id_mapping = (
        dataset_pb2.CameraSegmentationLabel.InstanceIDToGlobalIDMapping(
            local_instance_id=remapped, global_instance_id=orig))
    seg_proto.instance_id_to_global_id_mapping.append(id_mapping)

  return seg_proto


def load_frames_with_labels_from_dataset(
    tf_dataset: tf.data.Dataset,
    keep_first_k: int = 3
) -> Tuple[List[dataset_pb2.Frame], Optional[str]]:
  """Loads dataset frames with panoptic labels and sequence id for evaluation.

  Args:
    tf_dataset: An instance of the tf.data.Dataset containing the Open Dataset
      frames.
    keep_first_k: An integer, indicating the number of frames to load. Load all
      frames with panoptic labels if the number is greater.

  Returns:
    A tuple containing the following elements.
      - A list of Open Dataset frames.
      - The sequence id for the camera segmentation label.
  """
  frames_with_seg = []
  sequence_id = None

  for data in tf_dataset:
    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    if frame.images[0].camera_segmentation_label.panoptic_label:
      frames_with_seg.append(frame)
      if sequence_id is None:
        sequence_id = frame.images[0].camera_segmentation_label.sequence_id

      if (frame.images[0].camera_segmentation_label.sequence_id != sequence_id
          or len(frames_with_seg) >= keep_first_k):
        break

  return frames_with_seg, sequence_id
