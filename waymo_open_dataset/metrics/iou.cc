/* Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "waymo_open_dataset/metrics/iou.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <glog/logging.h>
#include "absl/base/attributes.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/math/aabox2d.h"
#include "waymo_open_dataset/math/box2d.h"
#include "waymo_open_dataset/math/polygon2d.h"
#include "waymo_open_dataset/math/vec2d.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {

namespace {
constexpr double kEpsilon = 1e-6;
// Min,max box dimensions (length, width, height).
constexpr double kMinBoxDim = 1e-2;
constexpr double kMaxBoxDim = 1e6;

// Returns true if the closed interval [min1, max1] overlaps at all with the
// closed interval [min2, max2]. If overlap_min and overlap_max are not
// null, set them to the closed overlap interval. If the overlap is empty,
// then overlap_min will be greater than overlap_max.
template <typename T>
bool ClosedRangesOverlap(T min1, T max1, T min2, T max2,
                         T* overlap_min = nullptr, T* overlap_max = nullptr) {
  CHECK_LE(min1, max1);
  CHECK_LE(min2, max2);
  T min = std::max(min1, min2);
  T max = std::min(max1, max2);
  if (overlap_min != nullptr) {
    *overlap_min = min;
  }
  if (overlap_max != nullptr) {
    *overlap_max = max;
  }
  return min <= max;
}

// Converts a box proto to a 2d box.
Box2d ToBox2d(const Label::Box& box) {
  return Box2d(Vec2d(box.center_x(), box.center_y()), box.heading(),
               box.length(), box.width());
}

// Converts a box proto to a 2d AA box.
AABox2d<double> ToAABox2d(const Label::Box& box) {
  return AABox2d<double>(box.center_x(), box.center_y(), box.length() * 0.5,
                         box.width() * 0.5);
}

// Computes the dot product of the centers of two boxes.
double CenterDotProduct(const Label::Box& box1, const Label::Box& box2) {
  return (box1.center_x() * box2.center_x()) +
         (box1.center_y() * box2.center_y()) +
         (box1.center_z() * box2.center_z());
}

// Computes the square of the length from the box center to the origin.
double CenterVectorLengthSquare(const Label::Box& box) {
  return box.center_x() * box.center_x() + box.center_y() * box.center_y() +
         box.center_z() * box.center_z();
}

// Computes the length from the box center to the origin.
double CenterVectorLength(const Label::Box& box) {
  return sqrt(CenterVectorLengthSquare(box));
}

// Apply translation to a 3D box proto.
Label::Box TranslateBox(const Label::Box& box, const double& t_x,
                        const double& t_y, const double& t_z) {
  Label::Box translated_box(box);
  translated_box.set_center_x(box.center_x() + t_x);
  translated_box.set_center_y(box.center_y() + t_y);
  translated_box.set_center_z(box.center_z() + t_z);
  return translated_box;
}
}  // namespace

Polygon2d ToPolygon2d(const Label::Box& box) { return Polygon2d(ToBox2d(box)); }

namespace {
double ComputeIoU2d(const Label::Box& b1, const Label::Box& b2) {
  if (b1.length() <= kMinBoxDim || b1.width() <= kMinBoxDim ||
      b2.length() <= kMinBoxDim || b2.width() <= kMinBoxDim) {
    LOG_EVERY_N(WARNING, 1000)
        << "Tiny box dim seen, return 0.0 IOU."
        << "\nb1: " << b1.DebugString() << "\nb2: " << b2.DebugString();
    return 0.0;
  }

  const Polygon2d p1 = ToPolygon2d(b1);
  const Polygon2d p2 = ToPolygon2d(b2);

  const double intersection_area = p1.ComputeIntersectionArea(p2);
  const double p1_area = b1.length() * b1.width();
  const double p2_area = b2.length() * b2.width();

  const double union_area = p1_area + p2_area - intersection_area;
  if (union_area <= kEpsilon) return 0.0;
  const double iou = intersection_area / union_area;
  CHECK(!std::isnan(iou)) << "b1: " << b1.DebugString()
                          << "\nb2: " << b2.DebugString();
  CHECK_GE(iou, -kEpsilon) << "b1: " << b1.DebugString()
                           << "\nb2: " << b2.DebugString();
  CHECK_LE(iou, 1.0 + kEpsilon)
      << "b1: " << b1.DebugString() << "\nb2: " << b2.DebugString();

  return std::max(std::min(iou, 1.0), 0.0);
}

double ComputeIoUAA2d(const Label::Box& b1, const Label::Box& b2) {
  if (b1.length() <= kMinBoxDim || b1.width() <= kMinBoxDim ||
      b2.length() <= kMinBoxDim || b2.width() <= kMinBoxDim) {
    LOG_EVERY_N(WARNING, 1000)
        << "Tiny box dim seen, return 0.0 IOU."
        << "\nb1: " << b1.DebugString() << "\nb2: " << b2.DebugString();
    return 0.0;
  }

  const AABox2d<double> aabox1 = ToAABox2d(b1);
  const AABox2d<double> aabox2 = ToAABox2d(b2);

  const double intersection_area = aabox1.ComputeIntersectionArea(aabox2);
  const double b1_area = b1.length() * b1.width();
  const double b2_area = b2.length() * b2.width();
  const double union_area = b1_area + b2_area - intersection_area;
  if (union_area <= kEpsilon) return 0.0;
  const double iou = intersection_area / union_area;
  CHECK(!std::isnan(iou)) << "b1: " << b1.DebugString()
                          << "\nb2: " << b2.DebugString();
  CHECK_GE(iou, -kEpsilon) << "b1: " << b1.DebugString()
                           << "\nb2: " << b2.DebugString();
  CHECK_LE(iou, 1.0 + kEpsilon)
      << "b1: " << b1.DebugString() << "\nb2: " << b2.DebugString();

  return std::max(std::min(iou, 1.0), 0.0);
}

double ComputeIoU3d(const Label::Box& b1, const Label::Box& b2) {
  if (b1.length() <= kMinBoxDim || b1.width() <= kMinBoxDim ||
      b1.height() <= kMinBoxDim || b2.length() <= kMinBoxDim ||
      b2.width() <= kMinBoxDim || b2.height() <= kMinBoxDim) {
    LOG_EVERY_N(WARNING, 1000)
        << "Tiny box dim seen, return 0.0 IOU."
        << "\nb1: " << b1.DebugString() << "\nb2: " << b2.DebugString();
    return 0.0;
  }

  double z_overlap_min = 0.0;
  double z_overlap_max = 0.0;
  if (!ClosedRangesOverlap(
          b1.center_z() - b1.height() * 0.5, b1.center_z() + b1.height() * 0.5,
          b2.center_z() - b2.height() * 0.5, b2.center_z() + b2.height() * 0.5,
          &z_overlap_min, &z_overlap_max)) {
    return 0.0;
  }
  const Polygon2d p1 = ToPolygon2d(b1);
  const Polygon2d p2 = ToPolygon2d(b2);

  const double intersection_area = p1.ComputeIntersectionArea(p2);
  const double intersection_volume =
      intersection_area * (z_overlap_max - z_overlap_min);

  const double b1_volume = b1.length() * b1.width() * b1.height();
  const double b2_volume = b2.length() * b2.width() * b2.height();
  const double union_volume = b1_volume + b2_volume - intersection_volume;
  if (union_volume <= kEpsilon) return 0.0;
  const double iou = intersection_volume / union_volume;
  CHECK(!std::isnan(iou)) << "b1: " << b1.DebugString()
                          << "\nb2: " << b2.DebugString();
  CHECK_GE(iou, -kEpsilon) << "b1: " << b1.DebugString()
                           << "\nb2: " << b2.DebugString();
  CHECK_LE(iou, 1.0 + kEpsilon)
      << "b1: " << b1.DebugString() << "\nb2: " << b2.DebugString();

  return std::max(std::min(iou, 1.0), 0.0);
}
}  // namespace

ABSL_CONST_INIT const int kMaxIoU = 1000 * 1000;

double ComputeIoU(const Label::Box& b1, const Label::Box& b2,
                  Label::Box::Type box_type) {
  if (b1.length() >= kMaxBoxDim || b1.width() >= kMaxBoxDim ||
      b1.height() >= kMaxBoxDim || b2.length() >= kMaxBoxDim ||
      b2.width() >= kMaxBoxDim || b2.height() >= kMaxBoxDim) {
    LOG_EVERY_N(WARNING, 1000)
        << "Huge box dim seen, return 0.0 IOU."
        << "\nb1: " << b1.DebugString() << "\nb2: " << b2.DebugString();
    return 0.0;
  }

  switch (box_type) {
    case Label::Box::TYPE_3D:
      return ComputeIoU3d(b1, b2);
    case Label::Box::TYPE_2D:
      return ComputeIoU2d(b1, b2);
    case Label::Box::TYPE_AA_2D:
      return ComputeIoUAA2d(b1, b2);
    case Label::TYPE_UNKNOWN:
      LOG(FATAL) << "Unknown box type.";
  }

  return 0.0;
}

double ComputeLocalizationAffinity(
    const Label::Box& prediction_box, const Label::Box& ground_truth_box,
    const Config::LocalizationErrorTolerantConfig& let_metric_config) {
  // Transform the boxes into the sensor coordinate system.
  const Label::Box calibrated_prediction_box =
      TranslateBox(prediction_box, -let_metric_config.sensor_location().x(),
                   -let_metric_config.sensor_location().y(),
                   -let_metric_config.sensor_location().z());
  const Label::Box calibrated_ground_truth_box =
      TranslateBox(ground_truth_box, -let_metric_config.sensor_location().x(),
                   -let_metric_config.sensor_location().y(),
                   -let_metric_config.sensor_location().z());

  // Dot product between the ground truth center vector and the prediction
  // center vector.
  const double gt_dot_pd =
      CenterDotProduct(calibrated_prediction_box, calibrated_ground_truth_box);
  const double gt_range =
      std::max(CenterVectorLength(calibrated_ground_truth_box), kEpsilon);
  const double pd_range =
      std::max(CenterVectorLength(calibrated_prediction_box), kEpsilon);

  // Compute the cos(theta), where theta is the angle between the center vectors
  // of prediction and ground truth.
  const double cos_of_gt_pd_angle =
      std::clamp(gt_dot_pd / gt_range / pd_range, 0.0, 1.0);

  // Compute the error terms as a percentage of the max tolerance.
  const float max_range_tolerance_meter =
      std::max(let_metric_config.longitudinal_tolerance_percentage() * gt_range,
               static_cast<double>(
                   let_metric_config.min_longitudinal_tolerance_meter()));
  const double range_error = std::abs(
      (pd_range * cos_of_gt_pd_angle - gt_range) / max_range_tolerance_meter);
  // Convert to affinity with value range [0.0, 1.0].
  return std::clamp(1.0 - range_error, 0.0, 1.0);
}

ComputeLocalizationAffinityFunc GetComputeLocalizationAffinityFunc(
    const Config::LocalizationErrorTolerantConfig& let_metric_config) {
  ComputeLocalizationAffinityFunc let_compute_localization_affinity_func =
      [&let_metric_config](const Label::Box& prediction_box,
                           const Label::Box& ground_truth_box) {
        return ComputeLocalizationAffinity(prediction_box, ground_truth_box,
                                           let_metric_config);
      };
  return let_compute_localization_affinity_func;
}

Label::Box AlignedPredictionBox(
    const Label::Box& prediction_box, const Label::Box& ground_truth_box,
    Config::LocalizationErrorTolerantConfig::AlignType align_type) {
  Label::Box aligned_prediction_box = Label::Box(prediction_box);
  switch (align_type) {
    case Config::LocalizationErrorTolerantConfig::TYPE_NOT_ALIGNED:
      // No alignment is performed.
      break;
    case Config::LocalizationErrorTolerantConfig::TYPE_CENTER_ALIGNED:
      // Moves the prediction box's center to be same as ground truth.
      aligned_prediction_box.set_center_x(ground_truth_box.center_x());
      aligned_prediction_box.set_center_y(ground_truth_box.center_y());
      break;
    case Config::LocalizationErrorTolerantConfig::TYPE_RANGE_ALIGNED: {
      // To move the prediction box's center along the line of sight so that
      // it has the closest distance to the ground truth box's center, the
      // projected vector can be described as:
      //   P' = |G|* cos(theta) * P/|P| = dot(G, P)/|P|^2 * P,
      // where G = [gt_x, gt_y, gt_z] and P = [pd_x, pd_y, pd_z] are the vectors
      // that describe the centers of a ground truth box and a prediction box.
      const double gt_dot_pd =
          CenterDotProduct(prediction_box, ground_truth_box);
      const double pd_range_sq =
          std::max(CenterVectorLengthSquare(prediction_box), kEpsilon);
      const double range_multiplier = gt_dot_pd / pd_range_sq;
      aligned_prediction_box.set_center_x(prediction_box.center_x() *
                                          range_multiplier);
      aligned_prediction_box.set_center_y(prediction_box.center_y() *
                                          range_multiplier);
      aligned_prediction_box.set_center_z(prediction_box.center_z() *
                                          range_multiplier);
      break;
    }
    case Config::LocalizationErrorTolerantConfig::TYPE_UNKNOWN:
      LOG(FATAL) << "Unknown IoU type.";
  }
  return aligned_prediction_box;
}

double ComputeLetIoU(
    const Label::Box& prediction_box, const Label::Box& ground_truth_box,
    const Config::LocalizationErrorTolerantConfig::Location3D& sensor_location,
    Config::LocalizationErrorTolerantConfig::AlignType align_type) {
  // Transform the boxes into the sensor coordinate system.
  const Label::Box calibrated_prediction_box =
      TranslateBox(prediction_box, -sensor_location.x(), -sensor_location.y(),
                   -sensor_location.z());
  const Label::Box calibrated_ground_truth_box =
      TranslateBox(ground_truth_box, -sensor_location.x(), -sensor_location.y(),
                   -sensor_location.z());
  const Label::Box aligned_prediction_box = AlignedPredictionBox(
      calibrated_prediction_box, calibrated_ground_truth_box, align_type);

  return ComputeIoU(aligned_prediction_box, calibrated_ground_truth_box,
                    Label::Box::TYPE_3D);
}

ComputeIoUFunc GetComputeLetIoUFunc(
    const Config::LocalizationErrorTolerantConfig::Location3D& sensor_location,
    Config::LocalizationErrorTolerantConfig::AlignType align_type) {
  ComputeIoUFunc compute_let_iou_func =
      [&sensor_location, align_type](const Label::Box& prediction_box,
                                     const Label::Box& ground_truth_box) {
        return ComputeLetIoU(prediction_box, ground_truth_box, sensor_location,
                             align_type);
      };
  return compute_let_iou_func;
}
}  // namespace open_dataset
}  // namespace waymo
