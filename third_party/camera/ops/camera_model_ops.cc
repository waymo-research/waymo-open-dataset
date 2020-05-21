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

#include <glog/logging.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "waymo_open_dataset/dataset.pb.h"
#include "third_party/camera/camera_model.h"

namespace tensorflow {
namespace {
namespace co = ::waymo::open_dataset;

// Length of the instrinsic vector.
constexpr int kIntrinsicLen = 9;
// Length of the camera metadata vector.
constexpr int kMetadataLen = 3;
// Length of the camera image metadata vector.
constexpr int kCameraImageMedataLen = 26;

struct Input {
  const Tensor* extrinsic = nullptr;
  const Tensor* intrinsic = nullptr;
  const Tensor* metadata = nullptr;
  const Tensor* camera_image_metadata = nullptr;
  const Tensor* input_coordinate = nullptr;
};

template <typename T>
DataType GetTensorflowType() {
  if (std::is_same<absl::remove_const_t<T>, double>::value) {
    return DT_DOUBLE;
  }
  if (std::is_same<absl::remove_const_t<T>, float>::value) {
    return DT_FLOAT;
  }
  CHECK(false) << "Unsupported type.";
}

// Parse input tensors to protos.
template <typename T>
void ParseInput(const Input& input, co::CameraCalibration* calibration_ptr,
                co::CameraImage* image_ptr) {
  auto& calibration = *calibration_ptr;
  auto& image = *image_ptr;

  CHECK_EQ(input.extrinsic->dim_size(0), 4);
  CHECK_EQ(input.extrinsic->dim_size(1), 4);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      calibration.mutable_extrinsic()->add_transform(
          input.extrinsic->matrix<T>()(i, j));
    }
  }
  CHECK_EQ(input.intrinsic->dim_size(0), kIntrinsicLen);
  for (int i = 0; i < kIntrinsicLen; ++i) {
    calibration.add_intrinsic(input.intrinsic->vec<T>()(i));
  }
  CHECK_EQ(input.metadata->dim_size(0), kMetadataLen);
  calibration.set_width(input.metadata->vec<int32>()(0));
  calibration.set_height(input.metadata->vec<int32>()(1));
  calibration.set_rolling_shutter_direction(
      static_cast<co::CameraCalibration::RollingShutterReadOutDirection>(
          input.metadata->vec<int32>()(2)));
  CHECK_EQ(input.camera_image_metadata->dim_size(0), kCameraImageMedataLen);
  int idx = 0;
  const auto& cim = input.camera_image_metadata->vec<T>();
  for (; idx < 16; ++idx) {
    image.mutable_pose()->add_transform(cim(idx));
  }
  image.mutable_velocity()->set_v_x(cim(idx++));
  image.mutable_velocity()->set_v_y(cim(idx++));
  image.mutable_velocity()->set_v_z(cim(idx++));
  image.mutable_velocity()->set_w_x(cim(idx++));
  image.mutable_velocity()->set_w_y(cim(idx++));
  image.mutable_velocity()->set_w_z(cim(idx++));
  image.set_pose_timestamp(cim(idx++));
  image.set_shutter(cim(idx++));
  image.set_camera_trigger_time(cim(idx++));
  image.set_camera_readout_done_time(cim(idx++));
}

template <typename T>
class WorldToImageOp : public OpKernel {
 public:
  explicit WorldToImageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Input input;
    OP_REQUIRES_OK(ctx, ctx->input("extrinsic", &input.extrinsic));
    OP_REQUIRES_OK(ctx, ctx->input("intrinsic", &input.intrinsic));
    OP_REQUIRES_OK(ctx, ctx->input("metadata", &input.metadata));
    OP_REQUIRES_OK(
        ctx, ctx->input("camera_image_metadata", &input.camera_image_metadata));
    OP_REQUIRES_OK(ctx,
                   ctx->input("global_coordinate", &input.input_coordinate));

    co::CameraCalibration calibration;
    co::CameraImage image;
    ParseInput<T>(input, &calibration, &image);

    co::CameraModel model(calibration);
    model.PrepareProjection(image);

    const int num_points = input.input_coordinate->dim_size(0);
    CHECK_EQ(3, input.input_coordinate->dim_size(1));
    Tensor image_coordinates(GetTensorflowType<T>(), {num_points, 3});
    for (int i = 0; i < num_points; ++i) {
      double u_d = 0.0;
      double v_d = 0.0;
      const bool valid =
          model.WorldToImage(input.input_coordinate->matrix<T>()(i, 0),
                             input.input_coordinate->matrix<T>()(i, 1),
                             input.input_coordinate->matrix<T>()(i, 2),
                             /*check_image_bounds=*/false, &u_d, &v_d);
      image_coordinates.matrix<T>()(i, 0) = u_d;
      image_coordinates.matrix<T>()(i, 1) = v_d;
      image_coordinates.matrix<T>()(i, 2) = static_cast<T>(valid);
    }
    ctx->set_output(0, image_coordinates);
  }
};
REGISTER_KERNEL_BUILDER(
    Name("WorldToImage").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    WorldToImageOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("WorldToImage").Device(DEVICE_CPU).TypeConstraint<double>("T"),
    WorldToImageOp<double>);

template <typename T>
class ImageToWorldOp final : public OpKernel {
 public:
  explicit ImageToWorldOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Input input;
    OP_REQUIRES_OK(ctx, ctx->input("extrinsic", &input.extrinsic));
    OP_REQUIRES_OK(ctx, ctx->input("intrinsic", &input.intrinsic));
    OP_REQUIRES_OK(ctx, ctx->input("metadata", &input.metadata));
    OP_REQUIRES_OK(
        ctx, ctx->input("camera_image_metadata", &input.camera_image_metadata));
    OP_REQUIRES_OK(ctx,
                   ctx->input("image_coordinate", &input.input_coordinate));

    co::CameraCalibration calibration;
    co::CameraImage image;
    ParseInput<T>(input, &calibration, &image);

    co::CameraModel model(calibration);
    model.PrepareProjection(image);

    const int num_points = input.input_coordinate->dim_size(0);
    CHECK_EQ(3, input.input_coordinate->dim_size(1));
    Tensor global_coordinates(GetTensorflowType<T>(), {num_points, 3});
    for (int i = 0; i < num_points; ++i) {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      model.ImageToWorld(input.input_coordinate->matrix<T>()(i, 0),
                         input.input_coordinate->matrix<T>()(i, 1),
                         input.input_coordinate->matrix<T>()(i, 2), &x, &y, &z);
      global_coordinates.matrix<T>()(i, 0) = x;
      global_coordinates.matrix<T>()(i, 1) = y;
      global_coordinates.matrix<T>()(i, 2) = z;
    }
    ctx->set_output(0, global_coordinates);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("ImageToWorld").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ImageToWorldOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("ImageToWorld").Device(DEVICE_CPU).TypeConstraint<double>("T"),
    ImageToWorldOp<double>);

REGISTER_OP("WorldToImage")
    .Attr("T: {float, double}")
    .Input("extrinsic: T")
    .Input("intrinsic: T")
    .Input("metadata: int32")
    .Input("camera_image_metadata: T")
    .Input("global_coordinate: T")
    .Output("image_coordinate: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return Status::OK();
    })
    .Doc(R"doc(
Maps global coordinates to image coordinates. See dataset.proto for more
  description of each field.

extrinsic: [4, 4] camera extrinsic matrix. CameraCalibration::extrinsic.
intrinsic: [9] camera intrinsic matrix. CameraCalibration::intrinsic.
metadata: [3] CameraCalibration::[width, height, rolling_shutter_direction].
camera_image_metadata: [16 + 6 + 1 + 1 + 1 + 1]=[26] tensor.
  CameraImage::[pose(16), velocity(6), pose_timestamp(1), shutter(1),
  camera_trigger_time(1), camera_readout_done_time(1)].
global_coordinate: [N, 3] float tensor. Points in global frame.
image_coordinate: [N, 3] float tensor. [N, 0:2] are points in image frame.
  The points can be outside of the image. The last channel [N, 2] tells whether
  a projection is valid or not. 0 means invalid. 1 means valid. A projection
  can be invalid if the point is behind the camera or if the radial distortion
  is too large.
)doc");

REGISTER_OP("ImageToWorld")
    .Attr("T: {float, double}")
    .Input("extrinsic: T")
    .Input("intrinsic: T")
    .Input("metadata: int32")
    .Input("camera_image_metadata: T")
    .Input("image_coordinate: T")
    .Output("global_coordinate: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return Status::OK();
    })
    .Doc(R"doc(
Maps global coordinates to image coordinates. See dataset.proto for more
  description of each field.

extrinsic: [4, 4] camera extrinsic matrix. CameraCalibration::extrinsic.
intrinsic: [9] camera intrinsic matrix. CameraCalibration::intrinsic.
metadata: [3] CameraCalibration::[width, height, rolling_shutter_direction].
camera_image_metadata: [16 + 6 + 1 + 1 + 1 + 1]=[26] tensor.
  CameraImage::[pose(16), velocity(6), pose_timestamp(1), shutter(1),
  camera_trigger_time(1), camera_readout_done_time(1)].
image_coordinate: [N, 3] float tensor. Points in image frame with depth.
global_coordinate: [N, 3] float tensor. Points in global frame.
)doc");

}  // namespace
}  // namespace tensorflow
