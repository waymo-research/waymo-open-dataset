# Copyright (c) 2019 Waymo LLC.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#    * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset import dataset_pb2
from third_party.camera.ops import py_camera_model_ops


class CameraModelOpsTest(tf.test.TestCase):

  def _BuildInput(self):
    """Builds input."""
    calibration = dataset_pb2.CameraCalibration()
    image = dataset_pb2.CameraImage()
    calibration_text = """
    name: FRONT
    intrinsic: 2055.55614936
    intrinsic: 2055.55614936
    intrinsic: 939.657469886
    intrinsic: 641.072182194
    intrinsic: 0.032316008498
    intrinsic: -0.321412482553
    intrinsic: 0.000793258395371
    intrinsic: -0.000625749354133
    intrinsic: 0.0
    extrinsic {
      transform: 0.999892684989
      transform: -0.00599320840002
      transform: 0.0133678704017
      transform: 1.53891424471
      transform: 0.00604223652133
      transform: 0.999975156055
      transform: -0.0036302411765
      transform: -0.0236339408393
      transform: -0.0133457814992
      transform: 0.00371062343188
      transform: 0.999904056092
      transform: 2.11527057298
      transform: 0.0
      transform: 0.0
      transform: 0.0
      transform: 1.0
    }
    width: 1920
    height: 1280
    rolling_shutter_direction: LEFT_TO_RIGHT
    """
    image_text = """
    name: FRONT
    image: "dummy"
    pose {
      transform: -0.913574384152
      transform: -0.406212760482
      transform: -0.0193141875914
      transform: -4069.03497872
      transform: 0.406637479491
      transform: -0.913082565675
      transform: -0.0304333457449
      transform: 11526.3118079
      transform: -0.00527303457417
      transform: -0.0356569976572
      transform: 0.999350175676
      transform: 86.504
      transform: 0.0
      transform: 0.0
      transform: 0.0
      transform: 1.0
    }
    velocity {
      v_x: -3.3991382122
      v_y: 1.50920391083
      v_z: -0.0169006548822
      w_x: 0.00158374733292
      w_y: 0.00212493073195
      w_z: -0.0270753838122
    }
    pose_timestamp: 1553640277.26
    shutter: 0.000424383993959
    camera_trigger_time: 1553640277.23
    camera_readout_done_time: 1553640277.28
    """
    text_format.Merge(calibration_text, calibration)
    text_format.Merge(image_text, image)
    return calibration, image

  def testCameraModel(self):
    calibration, image = self._BuildInput()
    g = tf.Graph()
    with g.as_default():
      extrinsic = tf.reshape(
          tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
          [4, 4])
      intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
      metadata = tf.constant([
          calibration.width, calibration.height,
          calibration.rolling_shutter_direction
      ],
                             dtype=tf.int32)
      camera_image_metadata = list(image.pose.transform)
      camera_image_metadata.append(image.velocity.v_x)
      camera_image_metadata.append(image.velocity.v_y)
      camera_image_metadata.append(image.velocity.v_z)
      camera_image_metadata.append(image.velocity.w_x)
      camera_image_metadata.append(image.velocity.w_y)
      camera_image_metadata.append(image.velocity.w_z)
      camera_image_metadata.append(image.pose_timestamp)
      camera_image_metadata.append(image.shutter)
      camera_image_metadata.append(image.camera_trigger_time)
      camera_image_metadata.append(image.camera_readout_done_time)
      image_points = tf.constant([[100, 1000, 20], [150, 1000, 20]],
                                 dtype=tf.float32)

      global_points = py_camera_model_ops.image_to_world(
          extrinsic, intrinsic, metadata, camera_image_metadata, image_points)
      image_points_t = py_camera_model_ops.world_to_image(
          extrinsic, intrinsic, metadata, camera_image_metadata, global_points)

    with self.test_session(graph=g) as sess:
      image_points, image_points_t, global_points = sess.run(
          [image_points, image_points_t, global_points])

    self.assertAllClose(
        global_points, [[-4091.97180, 11527.48092, 84.46586],
                        [-4091.771, 11527.941, 84.48779]],
        atol=0.1)
    self.assertAllClose(image_points_t[:, 0:2], image_points[:, 0:2], atol=0.1)
    self.assertAllClose(image_points_t[:, 2], [1.0, 1.0])


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
