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

#include <math.h>
#include <stddef.h>

#include <memory>

#include <glog/logging.h>
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "Eigen/Geometry"
#include "waymo_open_dataset/common/integral_types.h"
#include "waymo_open_dataset/dataset.pb.h"
#include "waymo_open_dataset/math/vec2d.h"

namespace waymo {
namespace open_dataset {

namespace {
// Bounds on the allowed radial distortion.
constexpr double kMinRadialDistortion = 0.8;
constexpr double kMaxRadialDistortion = 1.2;

Vec2d GetProjectionCenter(const CameraCalibration& calibration) {
  return Vec2d(calibration.intrinsic(2), calibration.intrinsic(3));
}

Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v) {
  // clang-format off
  Eigen::Matrix3d m;
  m <<     0, -v(2),  v(1),
        v(2),     0, -v(0),
       -v(1),  v(0),     0;
  // clang-format on
  return m;
}

Eigen::Isometry3d ToEigenTransform(const Transform& t) {
  Eigen::Isometry3d out;
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 4; c++) {
      const int ind = r * 4 + c;
      out(r, c) = t.transform(ind);
    }
  }
  return out;
}

double GetPixelTimestamp(
    CameraCalibration::RollingShutterReadOutDirection readout_direction,
    double shutter, double camera_trigger_time, double camera_readout_done_time,
    int image_width, int image_height, double x, double y) {
  // Please see dataset.proto for an explanation of shutter timings.
  const double readout_duration =
      camera_readout_done_time - camera_trigger_time - shutter;

  // Cameras have a rolling shutter, so each *sensor* row is exposed at a
  // slightly different time, starting with the top row and ending with the
  // bottom row. Because the sensor itself may be rotated, this means that the
  // *image* is captured row-by-row or column-by-column, depending on
  // `readout_direction`.
  // Final time for this pixel is the initial trigger time + the column and row
  // offset (exactly one of these will be non-zero) + half the shutter time to
  // get the middle of the exposure.
  const double base_ts = camera_trigger_time + 0.5 * shutter;
  switch (readout_direction) {
    case CameraCalibration::TOP_TO_BOTTOM:
      return base_ts + readout_duration / image_height * y;
    case CameraCalibration::BOTTOM_TO_TOP:
      return base_ts + readout_duration / image_height * (image_height - y);
    case CameraCalibration::LEFT_TO_RIGHT:
      return base_ts + readout_duration / image_width * x;
    case CameraCalibration::RIGHT_TO_LEFT:
      return base_ts + readout_duration / image_width * (image_width - x);
    default:
      LOG(FATAL) << "Should not reach here " << readout_direction;
  }
}

// In normalized camera, undistorts point coordinates via iteration.
void IterateUndistortion(const CameraCalibration& calibration, double u_nd,
                         double v_nd, double* u_n, double* v_n) {
  CHECK(u_n);
  CHECK(v_n);
  const double f_u = calibration.intrinsic(0);
  const double f_v = calibration.intrinsic(1);
  const double k1 = calibration.intrinsic(4);
  const double k2 = calibration.intrinsic(5);
  const double k3 = calibration.intrinsic(6);  // same as p1 in OpenCV.
  const double k4 = calibration.intrinsic(7);  // same as p2 in OpenCV
  const double k5 = calibration.intrinsic(8);  // same as k3 in OpenCV.

  double& u = *u_n;
  double& v = *v_n;

  // Initial guess.
  u = u_nd;
  v = v_nd;

  CHECK_GT(f_u, 0.0);
  CHECK_GT(f_v, 0.0);

  // Minimum required squared delta before terminating. Note that it is set in
  // normalized camera coordinates at a fraction of a pixel^2. The threshold
  // should satisfy unittest accuracy threshold kEpsilon = 1e-6 even for very
  // slow convergence.
  const double min_delta2 = 1e-12 / (f_u * f_u + f_v * f_v);

  // Iteratively apply the distortion model to undistort the image coordinates.
  // Maximum number of iterations when estimating undistorted point.
  constexpr int kMaxNumIterations = 20;

  for (int i = 0; i < kMaxNumIterations; ++i) {
    const double r2 = u * u + v * v;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double rd = 1.0 + r2 * k1 + r4 * k2 + r6 * k5;
    const double u_prev = u;
    const double v_prev = v;

    const double u_tangential = 2.0 * k3 * u * v + k4 * (r2 + 2.0 * u * u);
    const double v_tangential = 2.0 * k4 * u * v + k3 * (r2 + 2.0 * v * v);
    u = (u_nd - u_tangential) / rd;
    v = (v_nd - v_tangential) / rd;

    const double du = u - u_prev;
    const double dv = v - v_prev;
    // Early exit.
    if (du * du + dv * dv < min_delta2) {
      break;
    }
  }
}
}  // namespace

// Some naming conventions:
// tfm: transform
// dcm: direction cosine matrix
// xx0: xx frame at pose timestamp
// omega: angular velocity
struct CameraModel::RollingShutterState {
  // Define: t_pose_offset_ = t_pose - t_principal_point. In seconds.
  double t_pose_offset = 0.0;
  // sign * readout time / normalized_coordinate_range.
  // The sign depends on readout direction.
  double readout_time_factor = 0.0;
  // sign * readout time / range_in_pixel_space.
  // The sign depends on readout direction.
  double readout_time_factor_pixel = 0.0;
  // The principal point image coordinate, in pixels.
  Eigen::Vector2d principal_point;
  // Transformation from camera to ENU at pose timestamp.
  Eigen::Isometry3d n_tfm_cam0;
  // Velocity of camera at ENU frame at pose timestamp.
  Eigen::Vector3d n_vel_cam0;
  // Define: skew_omega = SkewSymmetric(cam_omega_cam0).
  Eigen::Matrix3d skew_omega;
  // Rotation from ENU to camera at pose timestamp.
  Eigen::Matrix3d cam0_dcm_n;
  // Define: skew_omega_dcm = skew_omega * cam0_dcm_n.
  Eigen::Matrix3d skew_omega_dcm;
  // Whether rolling shutter direction is horizontal.
  bool readout_horizontal_direction = false;
};

struct CameraModel::GlobalShutterState {
  // Transformation from camera to ENU at pose timestamp.
  Eigen::Isometry3d n_tfm_cam0;
  // Transformation from ENU to camera at pose timestamp.
  Eigen::Isometry3d cam_tfm_n;
};

CameraModel::CameraModel(const CameraCalibration& calibration)
    : calibration_(calibration) {}

CameraModel::~CameraModel() {}

double CameraModel::t_pose_offset() const {
  return rolling_shutter_state_->t_pose_offset;
}

void CameraModel::PrepareProjection(const CameraImage& camera_image) {
  const Eigen::Isometry3d n_tfm_vehicle0 =
      ToEigenTransform(camera_image.pose());
  const Eigen::Isometry3d vehicle_tfm_cam =
      ToEigenTransform(calibration_.extrinsic());
  if (global_shutter_state_ == nullptr) {
    global_shutter_state_ = absl::make_unique<GlobalShutterState>();
  }
  global_shutter_state_->n_tfm_cam0 = n_tfm_vehicle0 * vehicle_tfm_cam;
  global_shutter_state_->cam_tfm_n =
      global_shutter_state_->n_tfm_cam0.inverse();

  if (calibration_.rolling_shutter_direction() ==
      CameraCalibration::GLOBAL_SHUTTER) {
    return;
  }
  if (rolling_shutter_state_ == nullptr) {
    rolling_shutter_state_ =
        absl::make_unique<CameraModel::RollingShutterState>();
  }

  const double readout_time = camera_image.camera_readout_done_time() -
                              camera_image.camera_trigger_time() -
                              camera_image.shutter();

  const Vec2d principal_point_pixel = GetProjectionCenter(calibration_);
  rolling_shutter_state_->principal_point =
      Eigen::Vector2d{principal_point_pixel.x(), principal_point_pixel.y()};
  const double t_principal_point = GetPixelTimestamp(
      calibration_.rolling_shutter_direction(), camera_image.shutter(),
      camera_image.camera_trigger_time(),
      camera_image.camera_readout_done_time(), calibration_.width(),
      calibration_.height(), principal_point_pixel.x(),
      principal_point_pixel.y());
  rolling_shutter_state_->t_pose_offset =
      camera_image.pose_timestamp() - t_principal_point;
  if (calibration_.rolling_shutter_direction() ==
          CameraCalibration::RIGHT_TO_LEFT ||
      calibration_.rolling_shutter_direction() ==
          CameraCalibration::LEFT_TO_RIGHT) {
    rolling_shutter_state_->readout_horizontal_direction = true;
  } else {
    rolling_shutter_state_->readout_horizontal_direction = false;
  }

  // Compute readout time factor.
  double normalized_coord_range = 0;
  double range_in_pixel_space = 0;
  if (rolling_shutter_state_->readout_horizontal_direction) {
    double u_n_first = 0, v_n = 0, u_n_end = 0;
    ImageToDirection(0, 0.5 * calibration_.height(), &u_n_first, &v_n);
    ImageToDirection(calibration_.width(), 0.5 * calibration_.height(),
                     &u_n_end, &v_n);
    normalized_coord_range = u_n_end - u_n_first;
    range_in_pixel_space = calibration_.width();
  } else {
    double u_n = 0, v_n_first = 0, v_n_end = 0;
    ImageToDirection(0.5 * calibration_.width(), 0, &u_n, &v_n_first);
    ImageToDirection(0.5 * calibration_.width(), calibration_.height(), &u_n,
                     &v_n_end);
    normalized_coord_range = v_n_end - v_n_first;
    range_in_pixel_space = calibration_.height();
  }

  bool readout_reverse_direction = false;
  if (calibration_.rolling_shutter_direction() ==
          CameraCalibration::RIGHT_TO_LEFT ||
      calibration_.rolling_shutter_direction() ==
          CameraCalibration::BOTTOM_TO_TOP) {
    readout_reverse_direction = true;
  }
  rolling_shutter_state_->readout_time_factor =
      readout_reverse_direction ? -readout_time / normalized_coord_range
                                : readout_time / normalized_coord_range;

  rolling_shutter_state_->readout_time_factor_pixel =
      readout_reverse_direction ? -readout_time / range_in_pixel_space
                                : readout_time / range_in_pixel_space;

  rolling_shutter_state_->n_tfm_cam0 = n_tfm_vehicle0 * vehicle_tfm_cam;

  const Velocity& velocity = camera_image.velocity();
  // Compute cam_omega_cam0, n_vel_cam0.
  const Eigen::Vector3d n_vel_vehicle =
      Eigen::Vector3d{velocity.v_x(), velocity.v_y(), velocity.v_z()};
  const Eigen::Vector3d vehicle_omega_vehicle =
      Eigen::Vector3d{velocity.w_x(), velocity.w_y(), velocity.w_z()};
  const Eigen::Vector3d n_omega_vehicle =
      n_tfm_vehicle0.rotation() * vehicle_omega_vehicle;

  const Eigen::Vector3d cam_omega_cam0 =
      vehicle_tfm_cam.rotation().transpose() * vehicle_omega_vehicle;
  rolling_shutter_state_->skew_omega = SkewSymmetric(cam_omega_cam0);
  // Need to compensate velocity lever arm effect.
  // Lever arm effect:
  // https://en.wikipedia.org/wiki/Torque
  // v_1 = v_0 + omega x arm_length.
  const Eigen::Vector3d n_pos_cam0 =
      n_tfm_vehicle0.rotation() * vehicle_tfm_cam.translation();
  rolling_shutter_state_->n_vel_cam0 =
      n_vel_vehicle + SkewSymmetric(n_omega_vehicle) * n_pos_cam0;
  rolling_shutter_state_->cam0_dcm_n =
      rolling_shutter_state_->n_tfm_cam0.rotation().transpose();
  rolling_shutter_state_->skew_omega_dcm =
      -rolling_shutter_state_->skew_omega * rolling_shutter_state_->cam0_dcm_n;
}

// In this function, we are solving a scalar nonlinear optimization problem:
//  Min || t_h - IndexToTimeFromNormalizedCoord(Cam_p_f(t)) + t_offset ||
//  over t_h where t_h is explained below.
// where Cam_p_f(t) = projection(n_p_f, n_tfm_cam(t))
// The timestamps involved in the optimization problem:
// t_capture: The timestamp the rolling shutter camera can actually catch the
// point landmark. (this defines which scanline the point landmark falls in the
// image).
// t_principal_point: The timestamp of the principal point.
// t_pose: The timestamp of the anchor pose.
// Now we can define:
// t_offset := t_pose - t_principal_point.
// t_h := t_capture - t_pose.
// IndexToTimeFromNormalizedCoord(normalized_coord) := t_capture -
// t_principal_point.
// For this optimization problem we have the equality:
// t_h - IndexToTimeFromNormalizedCoord(.) + t_offset = 0
// This is efficient because it is a 1-dim problem, and typically converges in
// 2-3 iterations.
// To get the best performance and factor in the fact that our camera has little
// lens distortion, the IndexToTime(.) function is done in the normalized
// coordinate space instead of going to the distortion space. The testing
// results show that we get sufficiently good results already in normalized
// coordinate space.
bool CameraModel::WorldToImage(double x, double y, double z,
                               bool check_image_bounds, double* u_d,
                               double* v_d) const {
  if (calibration_.rolling_shutter_direction() ==
      CameraCalibration::GLOBAL_SHUTTER) {
    return WorldToImageGlobalShutter(x, y, z, check_image_bounds, u_d, v_d);
  }

  // The initial guess is the center of the image.
  double t_h = 0.;
  const Eigen::Vector3d n_pos_f{x, y, z};
  size_t iter_num = 0;

  // This threshold roughly corresponds to sub-pixel error for our camera
  // because the readout time per scan line is in the order of 1e-5 seconds.
  // Of course this number varies with the image size as well.
  constexpr double kThreshold = 1e-5;  // seconds.
  constexpr size_t kMaxIterNum = 4;

  Eigen::Vector2d normalized_coord;
  double residual = 2 * kThreshold;
  double jacobian = 0.;

  while (std::fabs(residual) > kThreshold && iter_num < kMaxIterNum) {
    if (!ComputeResidualAndJacobian(n_pos_f, t_h, &normalized_coord, &residual,
                                    &jacobian)) {
      return false;
    }

    // Solve for delta t;
    const double delta_t = -residual / jacobian;
    t_h += delta_t;
    ++iter_num;
  }

  // Get normalized coordinate.
  if (!ComputeResidualAndJacobian(n_pos_f, t_h, &normalized_coord, &residual,
                                  /*jacobian=*/nullptr)) {
    return false;
  }

  if (!DirectionToImage(normalized_coord(0), normalized_coord(1), u_d, v_d)) {
    return false;
  }

  // If requested, check if the returned pixel is inside the image.
  if (check_image_bounds) {
    return InImage(*u_d, *v_d);
  }

  return true;
}

void CameraModel::ImageToWorld(double u_d, double v_d, double depth, double* x,
                               double* y, double* z) const {
  if (calibration_.rolling_shutter_direction() ==
      CameraCalibration_RollingShutterReadOutDirection_GLOBAL_SHUTTER) {
    ImageToWorldGlobalShutter(u_d, v_d, depth, x, y, z);
    return;
  }
  const auto& rolling_shutter_state = *rolling_shutter_state_;
  // Interpolates the pose of camera.
  const double pixel_spacing =
      rolling_shutter_state.readout_horizontal_direction
          ? u_d - rolling_shutter_state.principal_point(0)
          : v_d - rolling_shutter_state.principal_point(1);
  const double t_h =
      rolling_shutter_state.readout_time_factor_pixel * pixel_spacing -
      rolling_shutter_state.t_pose_offset;

  const Eigen::Matrix3d cam_dcm_n = rolling_shutter_state.cam0_dcm_n +
                                    t_h * rolling_shutter_state.skew_omega_dcm;
  const Eigen::Vector3d n_pos_cam =
      rolling_shutter_state.n_tfm_cam0.translation() +
      t_h * rolling_shutter_state.n_vel_cam0;

  // Projects back to world frame.
  double u_n = 0, v_n = 0;
  ImageToDirection(u_d, v_d, &u_n, &v_n);
  const Eigen::Vector3d cam_pos_f{depth, -u_n * depth, -v_n * depth};
  const Eigen::Vector3d n_pos_f = cam_dcm_n.transpose() * cam_pos_f + n_pos_cam;
  *x = n_pos_f(0);
  *y = n_pos_f(1);
  *z = n_pos_f(2);
}

void CameraModel::ImageToWorldGlobalShutter(double u_d, double v_d,
                                            double depth, double* x, double* y,
                                            double* z) const {
  CHECK(x);
  CHECK(y);
  CHECK(z);
  double u_n = 0.0, v_n = 0.0;
  ImageToDirection(u_d, v_d, &u_n, &v_n);
  const Eigen::Vector3d wp = global_shutter_state_->n_tfm_cam0 *
                             Eigen::Vector3d(depth, -u_n * depth, -v_n * depth);
  *x = wp(0);
  *y = wp(1);
  *z = wp(2);
}

bool CameraModel::CameraToImage(double x, double y, double z,
                                bool check_image_bounds, double* u_d,
                                double* v_d) const {
  // Return if the 3D point is behind the camera.
  if (x <= 0.0) {
    *u_d = -1.0;
    *v_d = -1.0;
    return false;
  }

  // Convert the 3D point to a direction vector. If the distortion is out of
  // the limits, still compute u_d and v_d but return false.
  const double u = -y / x;
  const double v = -z / x;
  if (!DirectionToImage(u, v, u_d, v_d)) return false;

  // If requested, check if the projected pixel is inside the image.
  return check_image_bounds ? InImage(*u_d, *v_d) : true;
}

bool CameraModel::InImage(double u, double v) const {
  const double max_u = static_cast<double>(calibration_.width());
  const double max_v = static_cast<double>(calibration_.height());
  return u >= 0.0 && u < max_u && v >= 0.0 && v < max_v;
}

bool CameraModel::WorldToImageGlobalShutter(double x, double y, double z,
                                            bool check_image_bounds,
                                            double* u_d, double* v_d) const {
  CHECK(u_d);
  CHECK(v_d);
  const Eigen::Vector3d cp =
      global_shutter_state_->cam_tfm_n * Eigen::Vector3d(x, y, z);
  return CameraToImage(cp(0), cp(1), cp(2), check_image_bounds, u_d, v_d);
}

void CameraModel::ImageToDirection(double u_d, double v_d, double* u_n,
                                   double* v_n) const {
  const double f_u = calibration_.intrinsic(0);
  const double f_v = calibration_.intrinsic(1);
  const double c_u = calibration_.intrinsic(2);
  const double c_v = calibration_.intrinsic(3);

  // Initial guess, as a direction vector.
  const double u_nd = (u_d - c_u) / f_u;
  const double v_nd = (v_d - c_v) / f_v;
  // Iteratively refine estimate.
  IterateUndistortion(calibration_, u_nd, v_nd, u_n, v_n);
}

bool CameraModel::DirectionToImage(double u_n, double v_n, double* u_d,
                                   double* v_d) const {
  const double f_u = calibration_.intrinsic(0);
  const double f_v = calibration_.intrinsic(1);
  const double c_u = calibration_.intrinsic(2);
  const double c_v = calibration_.intrinsic(3);
  const double k1 = calibration_.intrinsic(4);
  const double k2 = calibration_.intrinsic(5);
  const double k3 = calibration_.intrinsic(6);  // same as p1 in OpenCV.
  const double k4 = calibration_.intrinsic(7);  // same as p2 in OpenCV
  const double k5 = calibration_.intrinsic(8);  // same as k3 in OpenCV.

  // (u, v, 1) is a normalized direction relative to ROI and principal point.
  const double r2 = u_n * u_n + v_n * v_n;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;

  // Radial distortion factor based on radius. This is the same for both the
  // perspective and the fisheye camera model.
  const double r_d = 1.0 + k1 * r2 + k2 * r4 + k5 * r6;

  double u_nd, v_nd;
  // If the radial distortion is too large, the computed coordinates will
  // be unreasonable (might even flip signs).
  if (r_d < kMinRadialDistortion || r_d > kMaxRadialDistortion) {
    // Check on which side of the image we overshoot, and set the coordinates
    // out of the image bounds accordingly. The coordinates will end up in a
    // viable range and direction but the exact values cannot be trusted.
    const double roi_clipping_radius =
        std::hypot(calibration_.width(), calibration_.height());
    const double r2_sqrt_rcp = 1.0 / std::sqrt(r2);
    *u_d = u_n * r2_sqrt_rcp * roi_clipping_radius + c_u;
    *v_d = v_n * r2_sqrt_rcp * roi_clipping_radius + c_v;
    return false;
  }

  // Normalized distorted camera coordinates.
  u_nd = u_n * r_d + 2.0 * k3 * u_n * v_n + k4 * (r2 + 2.0 * u_n * u_n);
  v_nd = v_n * r_d + k3 * (r2 + 2.0 * v_n * v_n) + 2.0 * k4 * u_n * v_n;

  // Un-normalize, un-center, and un-correct for ROI. Output coordinates are in
  // the current ROI frame.
  *u_d = u_nd * f_u + c_u;
  *v_d = v_nd * f_v + c_v;
  return true;
}

bool CameraModel::ComputeResidualAndJacobian(const Eigen::Vector3d& n_pos_f,
                                             double t_h,
                                             Eigen::Vector2d* normalized_coord,
                                             double* residual,
                                             double* jacobian) const {
  // The jacobian is allowed to be a nullptr.
  CHECK(normalized_coord);
  CHECK(residual);
  CHECK(rolling_shutter_state_);
  const RollingShutterState& rolling_shutter_state = *rolling_shutter_state_;

  const Eigen::Matrix3d cam_dcm_n = rolling_shutter_state.cam0_dcm_n +
                                    t_h * rolling_shutter_state.skew_omega_dcm;
  const Eigen::Vector3d n_pos_cam =
      rolling_shutter_state.n_tfm_cam0.translation() +
      t_h * rolling_shutter_state.n_vel_cam0;
  const Eigen::Vector3d cam_pos_f = cam_dcm_n * (n_pos_f - n_pos_cam);

  if (cam_pos_f(0) <= 0) {
    // The point is behind camera.
    return false;
  }

  (*normalized_coord)(0) = -cam_pos_f(1) / cam_pos_f(0);
  (*normalized_coord)(1) = -cam_pos_f(2) / cam_pos_f(0);

  const double normalized_spacing =
      rolling_shutter_state.readout_horizontal_direction
          ? (*normalized_coord)(0)
          : (*normalized_coord)(1);
  *residual = t_h -
              normalized_spacing * rolling_shutter_state.readout_time_factor +
              rolling_shutter_state.t_pose_offset;

  if (jacobian) {
    // The following is based on a reduced form of the derivative. The details
    // of the way to derive that derivative are skipped here.
    const Eigen::Vector3d jacobian_landmark_to_index =
        -cam_dcm_n * rolling_shutter_state.n_vel_cam0 -
        rolling_shutter_state.skew_omega * cam_pos_f;

    const double jacobian_combined =
        rolling_shutter_state.readout_horizontal_direction
            ? rolling_shutter_state.readout_time_factor / cam_pos_f(0) *
                  ((*normalized_coord)(0) * jacobian_landmark_to_index(0) -
                   jacobian_landmark_to_index(1))
            : rolling_shutter_state.readout_time_factor / cam_pos_f(0) *
                  ((*normalized_coord)(1) * jacobian_landmark_to_index(0) -
                   jacobian_landmark_to_index(2));
    *jacobian = 1. - jacobian_combined;
  }

  return true;
}

}  // namespace open_dataset
}  // namespace waymo
