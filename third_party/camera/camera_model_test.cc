/* Copyright (c) 2019 Waymo LLC. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
   * Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
==============================================================================*/

#include "third_party/camera/camera_model.h"

#include "google/protobuf/text_format.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "waymo_open_dataset/dataset.pb.h"

namespace waymo {
namespace open_dataset {
namespace {

using ::testing::Ge;

class CameraModelTest : public ::testing::Test {
 public:
  CameraModelTest() {
    static constexpr char kCalibrationStr[] = R"Text(
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
)Text";
    google::protobuf::TextFormat::ParseFromString(kCalibrationStr, &calibration_);

    static constexpr char kCameraImageStr[] = R"Text(
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
    )Text";
    google::protobuf::TextFormat::ParseFromString(kCameraImageStr, &camera_image_);
  }

 protected:
  CameraCalibration calibration_;
  CameraImage camera_image_;
};

TEST_F(CameraModelTest, RollingShutter) {
  CameraModel camera_model(calibration_);
  camera_model.PrepareProjection(camera_image_);

  double x, y, z;
  camera_model.ImageToWorld(100, 1000, 20, &x, &y, &z);

  double u_d, v_d;
  EXPECT_TRUE(camera_model.WorldToImage(x, y, z, /*check_image_bounds=*/true,
                                        &u_d, &v_d));
  EXPECT_NEAR(u_d, 100, 0.1);
  EXPECT_NEAR(v_d, 1000, 0.1);
  EXPECT_NEAR(x, -4091.88016, 0.1);
  EXPECT_NEAR(y, 11527.42299, 0.1);
  EXPECT_NEAR(z, 84.46667, 0.1);
}

TEST_F(CameraModelTest, GlobalShutter) {
  calibration_.set_rolling_shutter_direction(CameraCalibration::GLOBAL_SHUTTER);
  CameraModel camera_model(calibration_);
  camera_model.PrepareProjection(camera_image_);

  double x, y, z;
  camera_model.ImageToWorld(100, 1000, 20, &x, &y, &z);

  double u_d, v_d;
  EXPECT_TRUE(camera_model.WorldToImage(x, y, z, /*check_image_bounds=*/true,
                                        &u_d, &v_d));
  EXPECT_NEAR(u_d, 100, 0.1);
  EXPECT_NEAR(v_d, 1000, 0.1);
  EXPECT_NEAR(x, -4091.97180, 0.1);
  EXPECT_NEAR(y, 11527.48092, 0.1);
  EXPECT_NEAR(z, 84.46586, 0.1);
}

TEST_F(CameraModelTest, SubPixelChangeInPrinciplePointChangesPoseTimeOffset) {
  int center_x = static_cast<int>(calibration_.intrinsic(2));
  int center_y = static_cast<int>(calibration_.intrinsic(3));
  CameraCalibration calibration_a = calibration_;
  calibration_a.set_intrinsic(2, center_x);
  calibration_a.set_intrinsic(3, center_y);
  CameraCalibration calibration_b = calibration_;
  // Move principle point a little.
  const double sub_pixel = 0.1;
  calibration_a.set_intrinsic(2, center_x + sub_pixel);
  calibration_a.set_intrinsic(3, center_y + sub_pixel);
  CameraModel camera_a(calibration_a);
  CameraModel camera_b(calibration_b);

  camera_a.PrepareProjection(camera_image_);
  camera_b.PrepareProjection(camera_image_);

  const double readout_time = camera_image_.camera_readout_done_time() -
                              camera_image_.camera_trigger_time() -
                              camera_image_.shutter();
  const double min_seconds_per_col =
      readout_time / std::max(calibration_.width(), calibration_.height());
  EXPECT_THAT(std::abs(camera_a.t_pose_offset() - camera_b.t_pose_offset()),
              Ge(sub_pixel * min_seconds_per_col));
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
