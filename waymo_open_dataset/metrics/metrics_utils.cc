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

#include "waymo_open_dataset/metrics/metrics_utils.h"

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/breakdown_generator.h"
#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace internal {

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
PositiveModulo(T x, T y) {
  // Specialize for the case y is a constant (which it most often is). In this
  // case, the division 1 / y is optimized away.
  const T x_div_y = x * (1 / y);
  const T truncated_result = std::trunc(x_div_y);
  const T modulo = x - truncated_result * y;
  const T modulo_shifted = y + modulo;
  return modulo >= 0 ? modulo : (modulo_shifted >= y ? 0 : modulo_shifted);
}

// Wraps a value to be in [min_val, max_val).
template <typename T>
inline T WrapToRange(T min_val, T max_val, T val) {
  return PositiveModulo(val - min_val, max_val - min_val) + min_val;
}

// Wraps an angle in radians to be in [-pi, pi).
template <class T>
inline T NormalizeAngle(T rad) {
  return WrapToRange<T>(-M_PI, M_PI, rad);
}

bool IsTP(const std::vector<int>& pd_matches, int i) {
  CHECK_GE(i, 0);
  CHECK_LE(i, pd_matches.size());
  return pd_matches[i] >= 0;
}

bool IsFP(const Matcher& matcher, const std::vector<int>& pd_matches, int i) {
  CHECK_GE(i, 0);
  CHECK_LE(i, pd_matches.size());
  return pd_matches[i] < 0 &&
         !matcher.predictions()[matcher.prediction_subset()[i]]
              .overlap_with_nlz();
}

bool IsDetectionFN(const Matcher& matcher, const std::vector<int>& gt_matches,
                   int i, Label::DifficultyLevel level) {
  CHECK_GE(i, 0);
  CHECK_LE(i, gt_matches.size());
  if (gt_matches[i] >= 0) return false;

  const Object& object =
      matcher.ground_truths()[matcher.ground_truth_subset()[i]];
  return static_cast<int>(object.object().detection_difficulty_level()) <=
         static_cast<int>(level);
}

bool IsTrackingFN(const Matcher& matcher, const std::vector<int>& gt_matches,
                  int i, Label::DifficultyLevel level) {
  CHECK_GE(i, 0);
  CHECK_LE(i, gt_matches.size());
  if (gt_matches[i] >= 0) return false;

  const Object& object =
      matcher.ground_truths()[matcher.ground_truth_subset()[i]];
  return static_cast<int>(object.object().tracking_difficulty_level()) <=
         static_cast<int>(level);
}

float ComputeHeadingAccuracy(const Matcher& matcher, int prediction_index,
                             int ground_truth_index) {
  matcher.ValidPredictionIndex(prediction_index);
  matcher.ValidGroundTruthIndex(ground_truth_index);
  const float pd_heading = NormalizeAngle(
      matcher.predictions()[prediction_index].object().box().heading());
  const float gt_heading = NormalizeAngle(
      matcher.ground_truths()[ground_truth_index].object().box().heading());
  float diff_heading = std::abs(pd_heading - gt_heading);
  // Normalize heading error to [0, PI] (+PI and -PI are the same).
  if (diff_heading > M_PI) {
    diff_heading = 2.0 * M_PI - diff_heading;
  }
  // Clamp the range to avoid numerical errors.
  return std::min(1.0, std::max(0.0, 1.0 - diff_heading / M_PI));
}

std::vector<BreakdownShardSubset> BuildSubsets(
    const Config& config, const std::vector<Object>& objects, bool is_gt,
    bool is_detection) {
  std::vector<BreakdownShardSubset> result;
  for (int i = 0, sz = config.breakdown_generator_ids_size(); i < sz; ++i) {
    std::unique_ptr<BreakdownGenerator> breakdown_generator =
        BreakdownGenerator::Create(config.breakdown_generator_ids(i));
    const int num_shards = breakdown_generator->NumShards();
    std::vector<std::vector<int>> breakdown_subsets(num_shards);
    for (int i = 0, sz = objects.size(); i < sz; ++i) {
      if (is_detection && breakdown_generator->IsGroundTruthOnlyBreakdown()) {
        const std::vector<int> shards =
            breakdown_generator->ShardsForMatching(objects[i]);
        for (int s : shards) {
          breakdown_subsets[s].push_back(i);
        }
      } else {
        const int shard = breakdown_generator->Shard(objects[i]);
        CHECK_LT(shard, num_shards);
        if (shard >= 0) {
          breakdown_subsets[shard].push_back(i);
        }
      }
    }

    for (int shard = 0; shard < num_shards; ++shard) {
      result.emplace_back();

      // Create a subset for each breakdown shard.
      BreakdownShardSubset& subset = result.back();
      subset.breakdown_generator_id_index = i;
      subset.breakdown_shard = shard;
      if (is_gt) {
        subset.indices.emplace_back(std::move(breakdown_subsets[shard]));
        continue;
      }
      for (int i = 0, num_score_cutoffs = config.score_cutoffs_size();
           i < num_score_cutoffs; ++i) {
        subset.indices.emplace_back();
        std::vector<int>& indices = subset.indices.back();

        // For each object in the shard, check whether it is above the score
        // cutoff.
        for (int j = 0, sz = breakdown_subsets[shard].size(); j < sz; ++j) {
          if (objects[breakdown_subsets[shard][j]].score() >=
              config.score_cutoffs(i)) {
            indices.push_back(breakdown_subsets[shard][j]);
          }
        }
      }
    }
  }
  return result;
}

std::vector<Label::DifficultyLevel> GetDifficultyLevels(
    const Config& config, int breakdown_generator_id_index) {
  std::vector<Label::DifficultyLevel> difficulty_levels;
  if (config.difficulties(breakdown_generator_id_index).levels_size() > 0) {
    difficulty_levels.reserve(
        config.difficulties(breakdown_generator_id_index).levels_size());
    for (int level :
         config.difficulties(breakdown_generator_id_index).levels()) {
      difficulty_levels.push_back(static_cast<Label::DifficultyLevel>(level));
    }
  } else {
    difficulty_levels.push_back(Label::LEVEL_2);
  }
  return difficulty_levels;
}

namespace {
// Returns a sequence of floats {start + i * delta} where i >= 0 and i * delta
// <= end.
std::vector<float> Sequence(float start, float end, float delta) {
  std::vector<float> ret;
  while (start < end) {
    ret.push_back(start);
    start += delta;
  }
  if (!ret.empty()) {
    if (end - ret.back() > delta * 0.5) {
      ret.push_back(end);
    }
  }
  return ret;
}
}  // namespace

std::vector<float> DecideScoreCutoffs(const std::vector<float>& scores,
                                      int num_desired_cutoffs) {
  CHECK_GT(num_desired_cutoffs, 0);
  if (num_desired_cutoffs == 1) {
    return {0.0};
  }
  const int num_scores = scores.size();
  // If we do not have enough scores to estimate, use uniform distribution.
  if (num_desired_cutoffs >= num_scores) {
    return Sequence(/*start=*/0.0, /*end=*/1.0,
                    1.0 / (num_desired_cutoffs - 1));
  }

  // Find the size of each bucket.
  const int num_buckets = num_desired_cutoffs - 1;
  std::vector<int> bucket_sizes(num_buckets);
  int num_scores_remaining = num_scores;
  for (int i = 0; i < num_buckets; ++i) {
    bucket_sizes[i] = num_scores_remaining / (num_buckets - i);
    num_scores_remaining -= bucket_sizes[i];
  }

  std::vector<float> cutoffs;
  cutoffs.reserve(num_desired_cutoffs);
  cutoffs.push_back(scores[0]);
  int last_idx = 0;
  for (int i = 0; i < num_buckets - 1; ++i) {
    cutoffs.push_back(
        std::max(0.0f, std::min(1.0f, scores[last_idx + bucket_sizes[i]])));
    last_idx = last_idx + bucket_sizes[i];
  }
  cutoffs.push_back(1.0);
  CHECK_EQ(cutoffs.size(), num_desired_cutoffs);
  return cutoffs;
}

float ComputeMeanAveragePrecision(const std::vector<float>& precisions,
                                  const std::vector<float>& recalls,
                                  float max_recall_delta) {
  CHECK_EQ(precisions.size(), recalls.size());
  CHECK_GT(max_recall_delta, 0.0);
  CHECK_LE(max_recall_delta, 1.0);
  if (precisions.empty()) return 0.0;

  // Recall to precision mapping.
  std::map<float, float> recall_precision;
  recall_precision[0.0] = 1.0;
  for (int i = 0, sz = precisions.size(); i < sz; ++i) {
    recall_precision[recalls[i]] =
        std::max(recall_precision[recalls[i]], precisions[i]);
  }

  // Precision/Recall pair.
  struct PR {
    PR(float p, float r) : p(p), r(r) {}
    float p = 0.0;
    float r = 0.0;
  };

  // This vector is ordered by recall in descending order.
  std::vector<PR> precision_recall;
  // The last recall value we saw when iterating rp.
  float last_recall = recall_precision.cbegin()->first;
  // The maximum precision so far.
  float max_precision = 0.0;
  // Iterate from high recall to low recall.
  for (auto it = recall_precision.crbegin(); it != recall_precision.crend();
       ++it) {
    static constexpr float kError = 1e-6;
    while (last_recall - it->first > max_recall_delta + kError) {
      last_recall -= max_recall_delta;
      precision_recall.emplace_back(max_precision, last_recall);
    }
    // Update precision.
    max_precision = std::max(it->second, max_precision);
    precision_recall.emplace_back(max_precision, it->first);
    last_recall = it->first;
  }
  // Override the entry for recall 0.0.
  precision_recall[precision_recall.size() - 1].p =
      precision_recall[precision_recall.size() - 2].p;

  // Do integration to compute the area under the P/R curve to produce mAP.
  float mean_average_precision = 0.0;
  for (int i = 1, sz = precision_recall.size(); i < sz; ++i) {
    mean_average_precision +=
        0.5 * (precision_recall[i - 1].r - precision_recall[i].r) *
        (precision_recall[i - 1].p + precision_recall[i].p);
  }
  return mean_average_precision;
}

namespace {
inline float Sqr(float x) { return x * x; }

int Argmin(const std::vector<float>& nums) {
  CHECK(!nums.empty());
  int i = 0;
  for (int j = 1; j < nums.size(); ++j) {
    if (nums[j] < nums[i]) {
      i = j;
    }
  }
  return i;
}
}  // namespace

// Find the nearest ground truth for each prediction and use that ground truth's
// speed as the speed for the prediction.
std::vector<Object> EstimateObjectSpeed(const std::vector<Object>& pds,
                                        const std::vector<Object>& gts) {
  const int pd_size = pds.size();
  const int gt_size = gts.size();
  auto d = [](const Object& o1, const Object& o2) {
    return std::sqrt(
        Sqr(o1.object().box().center_x() - o2.object().box().center_x()) +
        Sqr(o1.object().box().center_y() - o2.object().box().center_y()) +
        Sqr(o1.object().box().center_z() - o2.object().box().center_z()));
  };

  std::vector<Object> pds_with_velocity(pds);
  if (gt_size == 0) {
    for (int i = 0; i < pd_size; ++i) {
      auto* metadata =
          pds_with_velocity[i].mutable_object()->mutable_metadata();
      metadata->set_speed_x(0.0);
      metadata->set_speed_y(0.0);
    }
  } else {
    std::vector<float> distances(gt_size);
    for (int i = 0; i < pd_size; ++i) {
      for (int j = 0; j < gt_size; ++j) {
        distances[j] = d(pds[i], gts[j]);
      }
      const int closest_gt_id = Argmin(distances);
      auto* metadata =
          pds_with_velocity[i].mutable_object()->mutable_metadata();
      metadata->set_speed_x(gts[closest_gt_id].object().metadata().speed_x());
      metadata->set_speed_y(gts[closest_gt_id].object().metadata().speed_y());
    }
  }
  return pds_with_velocity;
}

std::vector<std::vector<Object>> EstimateObjectSpeed(
    const std::vector<std::vector<Object>>& pds,
    const std::vector<std::vector<Object>>& gts) {
  CHECK_EQ(pds.size(), gts.size());
  std::vector<std::vector<Object>> pds_with_velocity(pds.size());
  for (int i = 0, sz = pds.size(); i < sz; ++i) {
    pds_with_velocity[i] = EstimateObjectSpeed(pds[i], gts[i]);
  }
  return pds_with_velocity;
}

bool HasVelocityBreakdown(const Config& config) {
  for (auto id : config.breakdown_generator_ids()) {
    if (id == Breakdown::VELOCITY) {
      return true;
    }
  }
  return false;
}

int FindGTWithLargestIoU(const Matcher& matcher, int pd_subset_id,
                         double iou_threshold) {
  const std::vector<int>& gt_subset = matcher.ground_truth_subset();
  double max_iou = iou_threshold;
  int gt_subset_id = -1;

  for (int i = 0, sz = gt_subset.size(); i < sz; ++i) {
    const double iou = matcher.IoU(matcher.prediction_subset()[pd_subset_id],
                                   matcher.ground_truth_subset()[i]);
    if (iou >= max_iou) {
      gt_subset_id = i;
      max_iou = iou;
    }
  }
  return gt_subset_id;
}

bool IsInBreakdown(const Object& object, const Breakdown& breakdown) {
  auto breakdown_generator =
      BreakdownGenerator::Create(breakdown.generator_id());
  return breakdown_generator->Shard(object) == breakdown.shard();
}

bool IsGroundTruthOnlyBreakdown(const Breakdown& breakdown) {
  auto breakdown_generator =
      BreakdownGenerator::Create(breakdown.generator_id());
  return breakdown_generator->IsGroundTruthOnlyBreakdown();
}

}  // namespace internal
}  // namespace open_dataset
}  // namespace waymo
