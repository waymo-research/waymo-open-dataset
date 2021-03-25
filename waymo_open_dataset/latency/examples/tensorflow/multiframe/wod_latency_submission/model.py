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
"""Module to run a simple PointNet on multiple frames of data."""
import numpy as np
import tensorflow as tf


# Request the top range image and vehicle pose for each of the 3 most current
# frames.
DATA_FIELDS = [
    'TOP_RANGE_IMAGE_FIRST_RETURN',
    'TOP_RANGE_IMAGE_FIRST_RETURN_1',
    'TOP_RANGE_IMAGE_FIRST_RETURN_2',
    'POSE',
    'POSE_1',
    'POSE_2',
]


def initialize_model():
  """Dummy function that would have initialized the model."""
  pass


def transform_point_cloud(pc, old_pose, new_pose):
  """Transform a point cloud from one vehicle frame to another.

  Args:
    pc: N x 6 float32 point cloud, where the final three dimensions are the
        cartesian coordinates of the points in the old vehicle frame.
    old_pose: 4 x 4 float32 vehicle pose at the timestamp of the point cloud
    new_pose: 4 x 4 float32 vehicle pose to transform the point cloud to.
  """
  # Extract the 3x3 rotation matrices and 3x1 translation vectors from the
  # old and new poses.
  # (3, 3)
  old_rot = old_pose[:3, :3]
  # (3, 1)
  old_trans = old_pose[:3, 3:4]
  # (3, 3)
  new_rot = new_pose[:3, :3]
  # (3, 1)
  new_trans = new_pose[:3, 3:4]

  # Extract the local cartesian coordinates from the N x 6 point cloud, adding
  # a new axis at the end to work with np.matmul.
  # (N, 3, 1)
  local_cartesian_coords = pc[..., 3:6][..., np.newaxis]

  # Transform the points from local coordinates to global using the old pose.
  # (N, 3, 1)
  global_coords = old_rot @ local_cartesian_coords + old_trans

  # Transform the points from global coordinates to the new local coordinates
  # using the new pose.
  # (N, 3, 1)
  new_local = np.matrix.transpose(new_rot) @ (global_coords - new_trans)

  # Reassign the dimensions of the range image with the cartesian coordinates
  # in
  pc[..., 3:6] = new_local[..., 0]


def get_model():
  """Returns a basic PointNet detection model in Keras.

  Returns:
    Keras model that implements a PointNet that outputs 50 detections. Its input
      is a 1 x N x 6 float32 point cloud and it ouptuts a dictionary with the
      following key-value pairs:
        "boxes": 1 x N x 7 float32 array
        "scores": 1 x N float32 array
  """
  # (1, N, 6)
  inp = tf.keras.Input(shape=(None, 6))

  # Run three per-point FC layers.
  feat = inp
  for i, layer_size in enumerate((128, 256, 512)):
    # (1, N, D)
    feat = tf.keras.layers.Dense(layer_size, activation='relu',
                                 name=f'linear_{i + 1}')(feat)

  # Run the global max pool and an FC layer on the global features.
  # (1, 512)
  global_feat = tf.keras.layers.GlobalMaxPool1D()(feat)
  global_feat = tf.keras.layers.Dense(512, activation='relu',
                                      name='global_mlp')(global_feat)

  # FC layers for each of the detection heads. Note that we regress each of the
  # 7 box dimensions separately since they require different activation
  # functions to constrain them to the appropriate ranges.
  num_boxes = 50
  # (1, 50)
  center_x = tf.keras.layers.Dense(num_boxes, name='center_x')(global_feat)
  center_y = tf.keras.layers.Dense(num_boxes, name='center_y')(global_feat)
  center_z = tf.keras.layers.Dense(num_boxes, name='center_z')(global_feat)
  length = tf.keras.layers.Dense(num_boxes, activation='softplus',
                                 name='length')(global_feat)
  width = tf.keras.layers.Dense(num_boxes, activation='softplus',
                                name='width')(global_feat)
  height = tf.keras.layers.Dense(num_boxes, activation='softplus',
                                 name='height')(global_feat)
  heading = np.pi * tf.keras.layers.Dense(num_boxes, activation='tanh',
                                          name='heading')(global_feat)
  scores = tf.keras.layers.Dense(num_boxes, activation='sigmoid',
                                 name='scores')(global_feat)

  # Combine the 7 box features into a single 1 x 50 x 7 box tensor.
  box_feats = [center_x, center_y, center_z, length, width, height, heading]
  reshape_layer = tf.keras.layers.Reshape((num_boxes, 1))
  for i in range(len(box_feats)):
    box_feats[i] = reshape_layer(box_feats[i])
  # (1, 50, 7)
  boxes = tf.keras.layers.Concatenate()(box_feats)

  return tf.keras.Model(inputs=inp, outputs={'boxes': boxes, 'scores': scores})


def run_model(**kwargs):
  """Run the model on range images and poses from the most recent 3 frames.

  Args:
    **kwargs: Keyword arguments whose names are the entries in DATA_FIELDS.

  Returns:
    Dict from string to numpy ndarray.
  """
  # (4, 4)
  newest_pose = kwargs['POSE']
  # Store the point clouds from each frame.
  point_clouds = []
  for suffix in ('', '_1', '_2'):
    # (H, W, 6)
    range_image = kwargs['TOP_RANGE_IMAGE_FIRST_RETURN' + suffix]
    # (N, 6)
    point_cloud = range_image[range_image[..., 0] > 0]
    if point_clouds:
      # (4, 4)
      src_pose = kwargs['POSE' + suffix]
      transform_point_cloud(point_cloud, src_pose, newest_pose)
    point_clouds.append(point_cloud)

  # Combine the point clouds from all thre frames into a single point cloud.
  # (N, 6)
  combined_pc = tf.concat(point_clouds, axis=0)

  # Run the detection model on the combined point cloud.
  model = get_model()
  output_tensors = model(tf.reshape(combined_pc, (1, -1, 6)))

  # Return the Tensors converted into numpy arrays.
  return {
      # Take the first example to go from 1 x B (x 7) to B (x 7).
      'boxes': output_tensors['boxes'][0, ...].numpy(),
      'scores': output_tensors['scores'][0, ...].numpy(),
      # Add a "classes" field that is always CAR.
      'classes': np.full(output_tensors['boxes'].shape[1], 1, dtype=np.uint8),
  }
