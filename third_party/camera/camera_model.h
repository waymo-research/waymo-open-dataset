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

#ifndef WAYMO_OPEN_DATASET_THIRD_PARTY_CAMERA_CAMERA_MODEL_H_
#define WAYMO_OPEN_DATASET_THIRD_PARTY_CAMERA_CAMERA_MODEL_H_

#include <memory>

#include "Eigen/Geometry"
#include "waymo_open_dataset/dataset.pb.h"

namespace waymo {
namespace open_dataset {

// Example usage:
// CameraModel camera_model(calibration);
// camera_model.PrepareProjection(image);
// camera_model.WorldToImage(...);
// camera_model.WorldToImage(...);
//
// This class is not threadsafe.
class CameraModel {
 public:
  explicit CameraModel(const CameraCalibration& calibration);
  virtual ~CameraModel();

  // This function should be called once per image before calling
  // `WorldToImage`. It pre-computes some projection relevant variables.
  void PrepareProjection(const CameraImage& camera_image);

  // Projects a 3D point in global coordinates into the lens distorted image
  // coordinates (u_d, v_d). These projections are in original image frame (x:
  // image width, y: image height).
  //
  // Returns false if the point is behind the camera or if the coordinates
  // cannot be trusted because the radial distortion is too large. When the
  // point is not within the field of view of the camera, u_d, v_d are still
  // assigned meaningful values. If the point is in front of the camera image
  // plane, actual u_d and v_d values are calculated.
  //
  // If the flag check_image_bounds is true, also returns false if the point is
  // not within the field of view of the camera.
  //
  // It does rolling shutter projection if the camera is not a
  // global shutter camera. To disable rolling shutter projection, override
  // rolling_shutter_direction in the camera calibration.
  // Requires: `PrepareProjection` is called.
  bool WorldToImage(double x, double y, double z, bool check_image_bounds,
                    double* u_d, double* v_d) const;

  // Converts a point in the image with a known depth into world coordinates.
  // Similar as `WorldToImage`. This method also compensates for rolling shutter
  // effect if applicable.
  // Requires: `PrepareProjection` is called.
  void ImageToWorld(double u_d, double v_d, double depth, double* x, double* y,
                    double* z) const;

  // True if the given image coordinates are within the image.
  bool InImage(double u, double v) const;

  // For testing.
  double t_pose_offset() const;

 private:
  // Projects a point in the 3D camera frame into the lens distorted image
  // coordinates (u_d, v_d).
  //
  // Returns false if the point is behind the camera or if the coordinates
  // cannot be trusted because the radial distortion is too large. When the
  // point is not within the field of view of the camera, u_d, v_d are still
  // assigned meaningful values. If the point is in front of the camera image
  // plane, actual u_d and v_d values are calculated.
  //
  // If the flag check_image_bounds is true, also returns false if the point is
  // not within the field of view of the camera.
  bool CameraToImage(double x, double y, double z, bool check_image_bounds,
                     double* u_d, double* v_d) const;

  // Similar as `WorldToImage` but only for global shutter.
  // Requires: `PrepareProjection` is called.
  bool WorldToImageGlobalShutter(double x, double y, double z,
                                 bool check_image_bounds, double* u_d,
                                 double* v_d) const;
  // Similar as `ImageToWorld` but only for global shutter.
  // Requires: `PrepareProjection` is called.
  void ImageToWorldGlobalShutter(double u_d, double v_d, double depth,
                                 double* x, double* y, double* z) const;

  // Converts lens distorted image coordinates (u_d, v_d) to the normalized
  // direction (u_n, v_n).
  void ImageToDirection(double u_d, double v_d, double* u_n, double* v_n) const;

  // Converts normalized direction (u_n, v_n) to the lens-distorted image
  // coordinates (u_d, v_d). Returns false if the radial distortion is too high,
  // but still sets u_d and v_d to clamped out-of-bounds values to get
  // directional information.
  bool DirectionToImage(double u_n, double v_n, double* u_d, double* v_d) const;

  // This is a helper function for rolling shutter projection.
  // It takes the rolling shutter state variable, position of landmark in ENU
  // frame, estimated time t_h, and computes projected feature in normalized
  // coordinate frame, the residual and the jacobian.
  // If the jacobian is given as nullptr we will skip its computation.
  bool ComputeResidualAndJacobian(const Eigen::Vector3d& n_pos_f, double t_h,
                                  Eigen::Vector2d* normalized_coord,
                                  double* residual, double* jacobian) const;

  // Forward declaration of an internal state used for global shutter projection
  // computation.
  struct GlobalShutterState;
  // Forward declaration of an internal state used for rolling shutter
  // projection computation.
  struct RollingShutterState;

  const CameraCalibration calibration_;
  std::unique_ptr<RollingShutterState> rolling_shutter_state_;
  std::unique_ptr<GlobalShutterState> global_shutter_state_;
};

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_THIRD_PARTY_CAMERA_CAMERA_MODEL_H_
