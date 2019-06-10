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

#ifndef WAYMO_OPEN_DATASET_METRICS_MOT_H_
#define WAYMO_OPEN_DATASET_METRICS_MOT_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {

// This implements MOT metrics for a sequence of frames defined in the paper.
// [1] Keni Bernardin, Alexander Elbs, Rainer Stiefelhagen
//     Multiple Object Tracking Performance Metrics and Evaluation in a
//     Smart Room Environment
//     https://cvhci.anthropomatik.kit.edu/~stiefel/papers/ECCV2006WorkshopCameraReady.pdf
//
// Example usage:
// MOT mot;
// // Each frame comes with a matcher.
// for (Matcher* matcher : frames) {
//   mot.Eval(matcher, difficulty_level);
// Measurement m = mot.measurement();
//
// Algorithm summary:
// Let M0 = {}. For every time frame t,
// Step 1:
//   For every mapping (o_i, h_j) in M_{t−1}, verify if it is still valid. If
//   object o_i is still visible and tracker hypothesis h_j still exists at time
//   t, and if their distance does not exceed the threshold T, make the
//   correspondence between o_i and h_j for frame t.
// Step 2:
//   For all objects for which no correspondence was made yet, try to find a
//   matching hypothesis. Allow only one-to-one matches, and pairs for which the
//   distance does not exceed T. The matching should be made in a way that
//   minimizes the total object hypothesis distance error for the concerned
//   objects. This is a minimum weight assignment problem. If a correspondence
//   (o_i, h_k) is made that contradicts a mapping (o_i, h_j) in M_{t−1},
//   replace (o_i, h_j) with (o_i, h_k) in M_t. Count this as a mismatch error.
// Step 3:
//   After the first two steps, a complete set of matching pairs for the current
//   time frame is known. For each of theses matches, calculate the distance d_i
//   between the object o_i and its corresponding hypothesis.
// Step 4:
//   All remaining hypotheses are considered false positives. Similarly, all
//   remaining objects are considered misses.
// Step 5:
//   Update M_{t} and repeat.
//
// This class is not threadsafe.
class MOT final {
 public:
  // Evaluates the predictions and ground truths specified in the given
  // 'matcher'.
  // This function implements the 'Mapping procedure' in the paper.
  //
  // This function call does not take ownership of 'matcher_ptr'. It calls
  // non-const functions in the given matcher. Ideally we should have another
  // wrapper on Matcher to abstracts these non-const calls to avoid mutating
  // matcher in this class. For now, we leave it as this way for simpler code.
  void Eval(Matcher *matcher_ptr, Label::DifficultyLevel difficulty_level);

  // Returns tracking result for all evaluations so far.
  TrackingMeasurement measurement() const { return measurement_; }

 private:
  struct MatchResult {
    // Number of false positives.
    int num_fps = 0;
    // Number of false negatives.
    int num_fns = 0;
    // Some false negatives are above the given difficulty level which should
    // not be really counted as false negatives. They will excluded from the
    // total number of ground truths later.
    int num_fns_above_difficulty_level = 0;
    // gt->pd IDs map.
    absl::flat_hash_map<std::string, std::string> gt_pd_matchings;
  };

  // Initializes new gt->pd mapping based on the mapping result from the
  // previous frame. It also updates {pd,gt}_map by removing those objects added
  // to the new mapping.
  // Requires: pd_map != nullptr, gt_map != nullptr.
  absl::flat_hash_map<std::string, std::string> InitializeNewMapping(
      const Matcher& matcher, absl::flat_hash_map<std::string, int>* pd_map,
      absl::flat_hash_map<std::string, int>* gt_map);

  // For all objects for which no correspondence was made yet, try to find a
  // matching hypothesis, and then updates the measurement_ and the new gt->pd
  // mapping.
  // Requires: matcher != nullptr.
  MatchResult Match(const absl::flat_hash_map<std::string, int>& pd_map,
                    const absl::flat_hash_map<std::string, int>& gt_map,
                    Label::DifficultyLevel difficulty_level, Matcher* matcher);

  // Computes the number of mismatches by comparing the new matching and
  // matching from last frame.
  int ComputeNumMismatches(
      const absl::flat_hash_map<std::string, std::string>& gt_pd_new_matchings,
      const absl::flat_hash_map<std::string, int>& gt_map);

  TrackingMeasurement measurement_;

  // Tracks the ground truth tracking ID to predicted tracking ID matchings till
  // now.
  // This is the 'M_{t}' in the paper.
  // These two maps ensures 1-1 mapping between pds and gts.
  absl::flat_hash_map<std::string, std::string> gt_pd_matchings_;
  absl::flat_hash_map<std::string, std::string> pd_gt_matchings_;
};

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_MOT_H_
