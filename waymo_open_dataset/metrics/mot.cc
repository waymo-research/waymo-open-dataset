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

#include "waymo_open_dataset/metrics/mot.h"

#include <memory>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/metrics/metrics_utils.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {
// Builds maps that map from object tracking ID to its index the subset in the
// matcher.
void BuildTrackingIdToDetectionBoxIndexMaps(
    const Matcher& matcher, absl::flat_hash_map<std::string, int>* pd_map,
    absl::flat_hash_map<std::string, int>* gt_map) {
  CHECK(pd_map != nullptr);
  CHECK(gt_map != nullptr);
  for (int i = 0, sz = matcher.prediction_subset().size(); i < sz; ++i) {
    const std::string& id =
        matcher.predictions()[matcher.prediction_subset()[i]].object().id();
    auto it = pd_map->emplace(id, i);
    // Well formatted predictions should not have duplicate predictions. But if
    // that happens, we just silently ignore redundant predictions.
    if (!it.second) {
      LOG(WARNING) << absl::StrFormat(
          "Duplicate predictions found for tracking ID %s (subset id %d).", id,
          i);
    }
  }
  for (int i = 0, sz = matcher.ground_truth_subset().size(); i < sz; ++i) {
    const std::string& id =
        matcher.ground_truths()[matcher.ground_truth_subset()[i]].object().id();
    auto it = gt_map->emplace(id, i);
    if (!it.second) {
      const std::string err = absl::StrFormat(
          "Duplicate ground truths found for tracking ID %s.", id);
      LOG(FATAL) << err;
    }
  }
}

}  // namespace

void MOT::Eval(Matcher* matcher_ptr, Label::DifficultyLevel difficulty_level) {
  CHECK(matcher_ptr != nullptr);
  Matcher& matcher = *matcher_ptr;
  // The implementation follows the 'Mapping procedure' in the paper.
  // Some comments below are copied from the paper.

  // Number of ground_truths.
  // Note that the number of ground_truths here is not the same as the number of
  // ground truths recorded at the metrics as some of the ground truths can have
  // difficulty level higher than what is specified in the inputs. They will be
  // excluded if the user does not predict anything for them.
  const int raw_num_gts = matcher.ground_truth_subset().size();

  // Maps from tracking ID to the detection box index (index in the subset).
  absl::flat_hash_map<std::string, int> pd_map;
  absl::flat_hash_map<std::string, int> gt_map;
  BuildTrackingIdToDetectionBoxIndexMaps(matcher, &pd_map, &gt_map);

  // New ground truth -> prediction mapping.
  absl::flat_hash_map<std::string, std::string> gt_pd_new_matchings =
      InitializeNewMapping(matcher, &pd_map, &gt_map);

  const MOT::MatchResult match_result =
      Match(pd_map, gt_map, difficulty_level, &matcher);
  gt_pd_new_matchings.insert(match_result.gt_pd_matchings.begin(),
                             match_result.gt_pd_matchings.end());
  // Sanity check to make sure there are not predictions matched to multiple
  // ground truths.
  {
    absl::flat_hash_set<std::string> pd_ids;
    for (const auto& kv : gt_pd_new_matchings) {
      CHECK(pd_ids.insert(kv.second).second)
          << "Duplicate prediction found for " << kv.second << ".";
    }
  }

  const int num_matches = gt_pd_new_matchings.size();
  // Update measurements.
  measurement_.set_num_misses(measurement_.num_misses() + match_result.num_fns);
  // We treat redundant predicted boxes as FP, but not boxes inside NLZs.
  measurement_.set_num_fps(measurement_.num_fps() + match_result.num_fps);
  measurement_.set_num_objects_gt(measurement_.num_objects_gt() + raw_num_gts -
                                  match_result.num_fns_above_difficulty_level);
  measurement_.set_num_matches(measurement_.num_matches() + num_matches);

  const int num_mismatches = ComputeNumMismatches(gt_pd_new_matchings, gt_map);
  measurement_.set_num_mismatches(measurement_.num_mismatches() +
                                  num_mismatches);

  // Update M_{t} and repeat.
  std::vector<std::string> gt_to_evict;
  for (const auto& kv : gt_pd_new_matchings) {
    gt_pd_matchings_[kv.first] = kv.second;
    // Ensure 1-1 mapping property.
    auto it = pd_gt_matchings_.find(kv.second);
    if (it != pd_gt_matchings_.end() &&
        gt_pd_new_matchings.find(it->second) == gt_pd_new_matchings.end()) {
      gt_to_evict.push_back(it->second);
    }
  }
  for (const auto& gt : gt_to_evict) {
    gt_pd_matchings_.erase(gt);
  }
  pd_gt_matchings_.clear();
  for (const auto& kv : gt_pd_matchings_) {
    CHECK(pd_gt_matchings_.emplace(kv.second, kv.first).second)
        << kv.first << " " << kv.second;
  }
}

absl::flat_hash_map<std::string, std::string> MOT::InitializeNewMapping(
    const Matcher& matcher, absl::flat_hash_map<std::string, int>* pd_map,
    absl::flat_hash_map<std::string, int>* gt_map) {
  CHECK(pd_map != nullptr);
  CHECK(gt_map != nullptr);
  // New ground truth -> prediction mapping.
  absl::flat_hash_map<std::string, std::string> gt_pd_new_matchings;

  // For every mapping (o_i, h_j) in M_{t−1}, verify if it is still valid. If
  // object o_i is still visible and tracker hypothesis h_j still exists at time
  // t, and if their distance does not exceed the threshold T, make the
  // correspondence between o_i and h_j for frame t.
  for (const auto& object_and_hyphothsis : gt_pd_matchings_) {
    auto pit = pd_map->find(object_and_hyphothsis.second);
    if (pit == pd_map->end()) continue;
    auto git = gt_map->find(object_and_hyphothsis.first);
    if (git == gt_map->end()) continue;

    const int pd_index = matcher.prediction_subset()[pit->second];
    const int gt_index = matcher.ground_truth_subset()[git->second];

    if (matcher.CanMatch(pd_index, gt_index)) {
      gt_pd_new_matchings.insert(object_and_hyphothsis);
      measurement_.set_matching_cost(measurement_.matching_cost() + 1.0 -
                                     matcher.IoU(pd_index, gt_index));

      pd_map->erase(pit);
      gt_map->erase(git);
    }
  }
  return gt_pd_new_matchings;
}

MOT::MatchResult MOT::Match(const absl::flat_hash_map<std::string, int>& pd_map,
                            const absl::flat_hash_map<std::string, int>& gt_map,
                            Label::DifficultyLevel difficulty_level,
                            Matcher* matcher) {
  CHECK(matcher != nullptr);
  // For all objects for which no correspondence was made yet, try to find a
  // matching hypothesis. Allow only one-to-one matches, and pairs for which the
  // distance does not exceed T. The matching should be made in a way that
  // minimizes the total object hypothesis distance error for the concerned
  // objects. This is a minimum weight assignment problem. If a correspondence
  // (o_i, h_k) is made that contradicts a mapping (o_i, h_j) in M_{t−1},
  // replace (o_i, h_j) with (o_i, h_k) in M_t. Count this as a mismatch error.

  // Remaining boxes that are not yet matched. Note that we have already deleted
  // matched boxes in the {predicted,reference}_boxes_map above.
  std::vector<int> pds_unmatched;
  std::vector<int> gts_unmatched;
  for (const auto& kv : pd_map) {
    pds_unmatched.push_back(matcher->prediction_subset()[kv.second]);
  }
  matcher->SetPredictionSubset(pds_unmatched);
  for (const auto& kv : gt_map) {
    gts_unmatched.push_back(matcher->ground_truth_subset()[kv.second]);
  }
  matcher->SetGroundTruthSubset(gts_unmatched);

  MatchResult result;
  std::vector<int> pd_matches;
  std::vector<int> gt_matches;
  matcher->Match(&pd_matches, &gt_matches);

  for (int i = 0, sz = pd_matches.size(); i < sz; ++i) {
    if (internal::IsFP(*matcher, pd_matches, i)) {
      ++result.num_fps;
      continue;
    }
    // Predictions that are neither TP or FP are those that overlap with NLZs.
    if (!internal::IsTP(pd_matches, i)) continue;

    const int prediction_index = matcher->prediction_subset()[i];
    const int ground_truth_index =
        matcher->ground_truth_subset()[pd_matches[i]];

    result.gt_pd_matchings
        [matcher->ground_truths()[ground_truth_index].object().id()] =
        matcher->predictions()[prediction_index].object().id();
    measurement_.set_matching_cost(
        measurement_.matching_cost() + 1.0 -
        matcher->IoU(prediction_index, ground_truth_index));
  }

  for (int i = 0, sz = gt_matches.size(); i < sz; ++i) {
    if (internal::IsTrackingFN(*matcher, gt_matches, i, difficulty_level)) {
      ++result.num_fns;
      continue;
    }
    if (gt_matches[i] < 0) {
      ++result.num_fns_above_difficulty_level;
    }
  }

  // Sanity check to make sure there are no predictions matched to multiple
  // ground truths.
  {
    absl::flat_hash_set<std::string> pd_ids;
    for (const auto& kv : result.gt_pd_matchings) {
      CHECK(pd_ids.insert(kv.second).second)
          << "Duplicate prediction found for " << kv.second << ".";
    }
  }
  return result;
}

int MOT::ComputeNumMismatches(
    const absl::flat_hash_map<std::string, std::string>& gt_pd_new_matchings,
    const absl::flat_hash_map<std::string, int>& gt_map) {
  // To maintain 1-1 mapping property. New matches might evict some old
  // matches, which are considered as mismatches as well.
  // Example (o2 is evicted): (see test Mismatch_GT_Eviction).
  // t0: o1->h1, o2->h2
  // t1: o1->h2,
  int num_mismatches = 0;
  // If a correspondence (o_i, h_k) is made that contradicts a mapping (o_i,
  // h_j) in M_{t−1}, replace (o_i, h_j) with (o_i, h_k) in M_t. Count this as a
  // mismatch error.
  for (const auto& kv : gt_pd_new_matchings) {
    auto git = gt_pd_matchings_.find(kv.first);
    if (git != gt_pd_matchings_.end() && git->second != kv.second) {
      ++num_mismatches;
    }
    // If a prediction is matched to a ground truth and it was matched to
    // another ground-truth, AND that ground truth is not in the latest frame,
    // then that ground truth is evicted.
    // NOTE: The last criterion is different from checking whether the ground
    // truth is not matched to anything in the latest frame as we want to avoid
    // double counting a ground truth as both a miss and a mismatch.
    // Example:
    // t0: o1->h1, o2->h2
    // t1: o1->h2, o2->
    // o2 is a false positive but not a mismatch (though it is but we don't want
    // to count it twice).
    auto pit = pd_gt_matchings_.find(kv.second);
    if (pit != pd_gt_matchings_.end() &&
        gt_map.find(pit->second) == gt_map.end() &&
        // This is needed as gt_map is not a complete set of ground truths in
        // the latest frame.
        gt_pd_new_matchings.find(pit->second) == gt_pd_new_matchings.end()) {
      ++num_mismatches;
    }
  }
  return num_mismatches;
}

}  // namespace open_dataset
}  // namespace waymo
