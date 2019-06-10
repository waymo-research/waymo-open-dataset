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

#include "waymo_open_dataset/metrics/ops/utils.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {
using LB = Label::Box;
typedef void (LB::*Setter)(double);

// Returns a 3d box from the tensor at a given index. This function does not
// check param validity.
// Requires:: bbox: [N, 7] float tensor. 0 <= box_index < N.
Label::Box GetBox3d(const tensorflow::Tensor& bbox, int box_index) {
  LB box;
  const auto matrix = bbox.matrix<float>();
  static const std::vector<Setter>* setters = new std::vector<Setter>{
      &LB::set_center_x, &LB::set_center_y, &LB::set_center_z, &LB::set_length,
      &LB::set_width,    &LB::set_height,   &LB::set_heading};
  for (int i = 0; i < 7; ++i) {
    (box.*(*setters)[i])(matrix(box_index, i));
  }
  return box;
}

// Returns a 2d box from the tensor at a given index. This function does not
// check param validity.
// Requires:: bbox: [N, 5] float tensor. 0 <= box_index < N.
Label::Box GetBox2d(const tensorflow::Tensor& bbox, int box_index) {
  LB box;
  const auto matrix = bbox.matrix<float>();
  static const std::vector<Setter>* setters = new std::vector<Setter>{
      &LB::set_center_x, &LB::set_center_y, &LB::set_length, &LB::set_width,
      &LB::set_heading};
  for (int i = 0; i < 5; ++i) {
    (box.*(*setters)[i])(matrix(box_index, i));
  }
  return box;
}

// Returns a 2d AA box from the tensor at a given index. This function does not
// check param validity.
// Requires:: bbox: [N, 4] float tensor. 0 <= box_index < N.
Label::Box GetAABox2d(const tensorflow::Tensor& bbox, int box_index) {
  LB box;
  const auto matrix = bbox.matrix<float>();
  static const std::vector<Setter>* setters = new std::vector<Setter>{
      &LB::set_center_x, &LB::set_center_y, &LB::set_length, &LB::set_width};
  for (int i = 0; i < 4; ++i) {
    (box.*(*setters)[i])(matrix(box_index, i));
  }
  return box;
}
}  // namespace

int GetDesiredBoxDOF(Label::Box::Type box_type) {
  switch (box_type) {
    case Label::Box::TYPE_AA_2D:
      return 4;
    case Label::Box::TYPE_2D:
      return 5;
    case Label::Box::TYPE_3D:
      return 7;
    case Label::Box::TYPE_UNKNOWN:
    default:
      LOG(FATAL) << "Unknown box type.";
  }
}

absl::flat_hash_map<int64, std::vector<Object>> ParseObjectFromTensors(
    const tensorflow::Tensor& bbox, const tensorflow::Tensor& type,
    const tensorflow::Tensor& frame_id,
    const absl::optional<const tensorflow::Tensor>& score,
    const absl::optional<const tensorflow::Tensor>& overlap_nlz,
    const absl::optional<const tensorflow::Tensor>& detection_difficulty,
    const absl::optional<const tensorflow::Tensor>& tracking_difficulty) {
  CHECK_EQ(bbox.dim_size(0), type.dim_size(0));
  CHECK_EQ(bbox.dim_size(0), frame_id.dim_size(0));
  if (score.has_value()) {
    CHECK_EQ(bbox.dim_size(0), score.value().dim_size(0));
  }
  if (overlap_nlz.has_value()) {
    CHECK_EQ(bbox.dim_size(0), overlap_nlz->dim_size(0));
  }
  if (detection_difficulty.has_value()) {
    CHECK_EQ(bbox.dim_size(0), detection_difficulty->dim_size(0));
  }
  if (tracking_difficulty.has_value()) {
    CHECK_EQ(bbox.dim_size(0), tracking_difficulty->dim_size(0));
  }

  absl::flat_hash_map<int64, std::vector<Object>> objects;
  const int32 box_dof = bbox.dim_size(1);
  for (int i = 0, n = bbox.dim_size(0); i < n; ++i) {
    Object object;
    object.mutable_object()->set_type(
        static_cast<Label::Type>(type.vec<uint8>()(i)));
    if (score.has_value()) {
      object.set_score(score.value().vec<float>()(i));
    }
    switch (box_dof) {
      case 4:
        *object.mutable_object()->mutable_box() = GetAABox2d(bbox, i);
        break;
      case 5:
        *object.mutable_object()->mutable_box() = GetBox2d(bbox, i);
        break;
      case 7:
        *object.mutable_object()->mutable_box() = GetBox3d(bbox, i);
        break;
      default:
        LOG(FATAL) << "Incorrect number of box DOF " << box_dof;
    }
    if (overlap_nlz.has_value()) {
      object.set_overlap_with_nlz(overlap_nlz.value().vec<bool>()(i));
    }
    if (detection_difficulty.has_value()) {
      object.mutable_object()->set_detection_difficulty_level(
          static_cast<Label::DifficultyLevel>(
              detection_difficulty.value().vec<uint8>()(i)));
    }
    if (tracking_difficulty) {
      object.mutable_object()->set_tracking_difficulty_level(
          static_cast<Label::DifficultyLevel>(
              detection_difficulty.value().vec<uint8>()(i)));
    }
    const int64 id = frame_id.vec<int64>()(i);
    objects[id].emplace_back(std::move(object));
  }
  return objects;
}

}  // namespace open_dataset
}  // namespace waymo
