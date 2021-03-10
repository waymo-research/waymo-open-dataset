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

#ifndef WAYMO_OPEN_DATASET_METRICS_CONFIG_UTIL_H_
#define WAYMO_OPEN_DATASET_METRICS_CONFIG_UTIL_H_

#include <string>
#include <vector>

#include "waymo_open_dataset/metrics/breakdown_generator.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"
#include "waymo_open_dataset/protos/motion_metrics.pb.h"

namespace waymo {
namespace open_dataset {

// Returns names for each metrics breakdown defined by the 'config'.
// The output vector is ordered as:
// [{generator_i_shard_j_difficulty_level_k}].
// i \in [0, num_breakdown_generators).
// j \in [0, num_shards for the i-th breakdown generator).
// k \in [0, num_difficulty_levels for each shard in the i-th breakdown
//   generator).
std::vector<std::string> GetBreakdownNamesFromConfig(const Config& config);

// Returns names for each metrics breakdown defined by `MotionConfig`.
// The output vector is ordered as:
// [{object_type_i_step_j}]
// j \in [0, len(step_configrations) for ith object_type]
// i \in [0, num_object_types (currently at 3: VEHICLE, PEDESTRIAN, CYCLIST)]
std::vector<std::string> GetBreakdownNamesFromMotionConfig(
    const MotionMetricsConfig& config);

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_CONFIG_UTIL_H_
