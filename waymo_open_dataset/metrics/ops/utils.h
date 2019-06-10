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

#ifndef WAYMO_OPEN_DATASET_METRICS_OPS_UTILS_H_
#define WAYMO_OPEN_DATASET_METRICS_OPS_UTILS_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/tensor.h"
#include "waymo_open_dataset/common/integral_types.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {

// Returns the desired number of box degrees of freedom given the config.
// 3d -> 7, 2d -> 5, 2d axis aligned -> 4.
int GetDesiredBoxDOF(Label::Box::Type box_type);

// Parses objects from tensors.
// bbox: [N, D] float tensor encodes the boxes.
// type: [N] uint8 tensor encodes the object types.
// frame_id: [N] int64 tensor tells which frame the box belongs to.
// score: [N] float tensor encodes the confidence scrore of each box.
// overlap_nlz: [N] boolean tensor tells whether a box overlaps with any no
//   label zone.
// detection_dificulty: [N] uint8 tensor tells the difficulty level of each box
//   for detection problems.
// tracking_dificulty: [N] uint8 tensor tells the difficulty level of each box
//   for tracking problems.
// Returns objects parsed from tensors grouped by frame IDs.
absl::flat_hash_map<int64, std::vector<Object>> ParseObjectFromTensors(
    const tensorflow::Tensor& bbox, const tensorflow::Tensor& type,
    const tensorflow::Tensor& frame_id,
    const absl::optional<const tensorflow::Tensor>& score,
    const absl::optional<const tensorflow::Tensor>& overlap_nlz,
    const absl::optional<const tensorflow::Tensor>& detection_difficulty,
    const absl::optional<const tensorflow::Tensor>& tracking_difficulty);

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_OPS_UTILS_H_
