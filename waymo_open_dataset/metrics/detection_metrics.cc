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

#include "waymo_open_dataset/metrics/detection_metrics.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include <glog/logging.h>
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/metrics/metrics_utils.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {
// Computes a detection measurement (TP, FP, FN) from matching result at a given
// difficulty level.
// A ground truth without any prediction matched is not considered as an FN if
// the ground truth has a detection difficulty level higher than the given one.
// Note: pd_matches, gt_matches use indices to the *subsets* maintained by
// matcher.
DetectionMeasurement ComputeDetectionMeasurementFromMatchingResult(
    const Config& config, const Matcher& matcher,
    const std::vector<int>& pd_matches, const std::vector<int>& gt_matches,
    Label::DifficultyLevel difficulty_level, const Breakdown& breakdown) {
  int num_true_positives = 0;
  int num_false_positives = 0;
  int num_false_negatives = 0;
  float sum_heading_accuracy = 0.0;

  DetectionMeasurement measurement;
  DetectionMeasurement::Details* details = nullptr;
  if (config.include_details_in_measurements()) {
    details = measurement.add_details();
  }

  auto is_in_breakdown = [&matcher, &breakdown](int gt_subset_id) {
    return internal::IsInBreakdown(
        matcher.ground_truths()[matcher.ground_truth_subset()[gt_subset_id]],
        breakdown);
  };
  for (int i = 0, sz = pd_matches.size(); i < sz; ++i) {
    const int pd_index = matcher.prediction_subset()[i];
    const std::string& pd_id = matcher.predictions()[pd_index].object().id();
    // This is a true positive only if
    // 1) This prediction matches a ground truth.
    // 2) The matched ground truth is in the given breakdown.
    if (internal::IsTP(pd_matches, i) &&
        (!internal::IsGroundTruthOnlyBreakdown(breakdown) ||
         is_in_breakdown(pd_matches[i]))) {
      const int gt_index = matcher.ground_truth_subset()[pd_matches[i]];
      const float heading_accuracy =
          internal::ComputeHeadingAccuracy(matcher, pd_index, gt_index);
      if (heading_accuracy <= config.min_heading_accuracy()) continue;
      ++num_true_positives;
      if (details != nullptr) {
        details->add_tp_pr_ids(pd_id);
        details->add_tp_gt_ids(matcher.ground_truths()[gt_index].object().id());
      }
      sum_heading_accuracy += heading_accuracy;
    }
    // This is a false positive only if
    // 1) This prediction does not match to any ground truth.
    // 2) The prediction does not overlap with any other ground truth that is
    //  not inside this breakdown. The threshold of deciding whether there is an
    //  overlap is set to kOverlapIoUThreshold for now.
    if (internal::IsFP(matcher, pd_matches, i)) {
      bool is_fp = false;
      if (internal::IsGroundTruthOnlyBreakdown(breakdown)) {
        static constexpr double kOverlapIoUThreshold = 0.01;
        const int gt_subset_id = internal::FindGTWithLargestIoU(
            matcher, i, /*iou_threshold=*/kOverlapIoUThreshold);
        const bool overlap_with_gt_in_other_shard =
            gt_subset_id >= 0 && !is_in_breakdown(gt_subset_id);
        if (!overlap_with_gt_in_other_shard) {
          is_fp = true;
        }
      } else {
        is_fp = true;
      }
      if (is_fp) {
        ++num_false_positives;
        if (details != nullptr) {
          details->add_fp_ids(pd_id);
        }
      }
    }
  }
  for (int i = 0, sz = gt_matches.size(); i < sz; ++i) {
    // This is false negative only if
    // 1) This ground truth is not matched to any prediction.
    // 2) This ground truth is inside the given breakdown.
    if (internal::IsDetectionFN(matcher, gt_matches, i, difficulty_level) &&
        (!internal::IsGroundTruthOnlyBreakdown(breakdown) ||
         is_in_breakdown(i))) {
      ++num_false_negatives;
      if (details != nullptr) {
        const int gt_index = matcher.ground_truth_subset()[i];
        details->add_fn_ids(matcher.ground_truths()[gt_index].object().id());
      }
    }
  }
  measurement.set_num_tps(num_true_positives);
  measurement.set_num_fps(num_false_positives);
  measurement.set_num_fns(num_false_negatives);
  measurement.set_sum_ha(sum_heading_accuracy);
  if (details != nullptr &&
      details->tp_gt_ids_size() != details->tp_pr_ids_size()) {
    LOG(FATAL) << "True positive sizes should be equal. pr size: "
               << details->tp_pr_ids_size()
               << ", gt size: " << details->tp_gt_ids_size();
  }
  return measurement;
}

// Computes detection measurements for a single breakdown shard.
// Returns a vector of detection measurements. Each element of that corresponds
// to a difficulty level. The order of the vector is same as the difficulty
// levels specified in the config.
std::vector<DetectionMeasurements>
ComputeDetectionMeasurementsPerBreakdownShard(
    const Config& config, const internal::BreakdownShardSubset& pd_subset,
    const internal::BreakdownShardSubset& gt_subset, Matcher* matcher) {
  CHECK(matcher != nullptr);
  std::vector<DetectionMeasurements> measurements;
  CHECK(!gt_subset.indices.empty());
  matcher->SetGroundTruthSubset(gt_subset.indices[0]);
  const std::vector<Label::DifficultyLevel> difficulty_levels =
      internal::GetDifficultyLevels(config,
                                    pd_subset.breakdown_generator_id_index);
  measurements.resize(difficulty_levels.size());
  for (int i = 0, sz = difficulty_levels.size(); i < sz; ++i) {
    auto* breakdown = measurements[i].mutable_breakdown();
    breakdown->set_generator_id(
        config.breakdown_generator_ids(pd_subset.breakdown_generator_id_index));
    breakdown->set_shard(pd_subset.breakdown_shard);
    breakdown->set_difficulty_level(difficulty_levels[i]);
  }

  // For each score cutoff in the config, filter predictions with score below
  // the cutoff and then do matching. Then computes detection measurements for
  // each difficulty level.
  for (int score_idx = 0, scores_sz = config.score_cutoffs_size();
       score_idx < scores_sz; ++score_idx) {
    matcher->SetPredictionSubset(pd_subset.indices[score_idx]);

    std::vector<int> pd_matches;
    std::vector<int> gt_matches;
    matcher->Match(&pd_matches, &gt_matches);

    for (int dl_idx = 0, dl_sz = difficulty_levels.size(); dl_idx < dl_sz;
         ++dl_idx) {
      *measurements[dl_idx].add_measurements() =
          ComputeDetectionMeasurementFromMatchingResult(
              config, *matcher, pd_matches, gt_matches,
              difficulty_levels[dl_idx], measurements[dl_idx].breakdown());
      measurements[dl_idx].mutable_measurements()->rbegin()->set_score_cutoff(
          config.score_cutoffs(score_idx));
    }
  }
  return measurements;
}

// Merges two detection measurements.
DetectionMeasurement MergeDetectionMeasurement(const DetectionMeasurement& m1,
                                               const DetectionMeasurement& m2) {
  if (!m1.has_score_cutoff()) return m2;
  if (!m2.has_score_cutoff()) return m1;
  CHECK_EQ(m1.score_cutoff(), m2.score_cutoff());
  DetectionMeasurement m;

#define ADD_FIELD(FIELD_NAME) \
  m.set_##FIELD_NAME(m1.FIELD_NAME() + m2.FIELD_NAME())
  ADD_FIELD(num_fps);
  ADD_FIELD(num_tps);
  ADD_FIELD(num_fns);
  ADD_FIELD(sum_ha);
#undef ADD_FIELD

  // If we enables details population, appends it as a new frame. The new
  // frame's `details()` size should be 1.
  if (m1.details_size() == 1 || m2.details_size() == 1) {
    *m.mutable_details() = m1.details();
    m.mutable_details()->MergeFrom(m2.details());
  }

  m.set_score_cutoff(m1.score_cutoff());
  return m;
}

// Merges new_m to m.
void MergeDetectionMeasurements(const DetectionMeasurements& new_m,
                                DetectionMeasurements* m) {
  CHECK(m != nullptr);
  if (m->measurements_size() == 0) {
    *m = new_m;
    return;
  }
  CHECK_EQ(m->measurements_size(), new_m.measurements_size());
  CHECK_EQ(m->breakdown().generator_id(), new_m.breakdown().generator_id());
  CHECK_EQ(m->breakdown().shard(), new_m.breakdown().shard());
  CHECK_EQ(m->breakdown().difficulty_level(),
           new_m.breakdown().difficulty_level());
  for (int i = 0, sz = m->measurements_size(); i < sz; ++i) {
    *m->mutable_measurements(i) =
        MergeDetectionMeasurement(m->measurements(i), new_m.measurements(i));
  }
}

// Merges new_m to m element by element.
void MergeDetectionMeasurementsVector(
    const std::vector<DetectionMeasurements>& new_m,
    std::vector<DetectionMeasurements>* m) {
  CHECK(m != nullptr);
  if (m->empty()) {
    *m = new_m;
    return;
  }

  CHECK_EQ(new_m.size(), m->size());
  for (int i = 0, sz = m->size(); i < sz; ++i) {
    MergeDetectionMeasurements(new_m[i], &(*m)[i]);
  }
}

// Converts detection measurements to detection metrics.
DetectionMetrics ToDetectionMetrics(const Config& config,
                                    DetectionMeasurements&& measurements,
                                    float desired_recall_delta) {
  DetectionMetrics metrics;
  *metrics.mutable_measurements() = measurements;
  const auto& m = metrics.measurements();
  *metrics.mutable_breakdown() = m.breakdown();

  for (const DetectionMeasurement& measurement : m.measurements()) {
    metrics.add_score_cutoffs(measurement.score_cutoff());

    const int tp_fp_sum = measurement.num_tps() + measurement.num_fps();
    if (tp_fp_sum <= 0) {
      metrics.add_precisions(0.0);
      metrics.add_precisions_ha_weighted(0.0);
    } else {
      const float precision =
          static_cast<float>(measurement.num_tps()) / tp_fp_sum;
      const float precision_ha = measurement.sum_ha() / tp_fp_sum;

      metrics.add_precisions(precision < config.min_precision() ? 0.0
                                                                : precision);
      metrics.add_precisions_ha_weighted(
          precision_ha < config.min_precision() ? 0.0 : precision_ha);
    }

    const int tp_fn_sum = measurement.num_tps() + measurement.num_fns();
    if (tp_fn_sum <= 0) {
      metrics.add_recalls(0.0);
      metrics.add_recalls_ha_weighted(0.0);
    } else {
      metrics.add_recalls(static_cast<float>(measurement.num_tps()) /
                          tp_fn_sum);
      // Use num_tps directly instead of sum_ha as none of the existing matcher
      // implementations takes heading accuracy into account.
      metrics.add_recalls_ha_weighted(
          static_cast<float>(measurement.num_tps()) / tp_fn_sum);
    }
    // If recall = 0.0, manually set precision = 1.0.
    if (*metrics.recalls().rbegin() == 0.0) {
      *metrics.mutable_precisions()->rbegin() = 1.0;
    }
    if (*metrics.recalls_ha_weighted().rbegin() == 0.0) {
      *metrics.mutable_precisions_ha_weighted()->rbegin() = 1.0;
    }
  }
  std::vector<float> precisions;
  std::vector<float> recalls;
  std::vector<float> precisions_ha_weighted;
  std::vector<float> recalls_ha_weighted;
  std::copy(metrics.precisions().begin(), metrics.precisions().end(),
            std::back_inserter(precisions));
  std::copy(metrics.recalls().begin(), metrics.recalls().end(),
            std::back_inserter(recalls));
  std::copy(metrics.precisions_ha_weighted().begin(),
            metrics.precisions_ha_weighted().end(),
            std::back_inserter(precisions_ha_weighted));
  std::copy(metrics.recalls_ha_weighted().begin(),
            metrics.recalls_ha_weighted().end(),
            std::back_inserter(recalls_ha_weighted));

  metrics.set_mean_average_precision(internal::ComputeMeanAveragePrecision(
      precisions, recalls, desired_recall_delta));
  metrics.set_mean_average_precision_ha_weighted(
      internal::ComputeMeanAveragePrecision(
          precisions_ha_weighted, recalls_ha_weighted, desired_recall_delta));

  return metrics;
}
}  // namespace

std::vector<DetectionMeasurements> ComputeDetectionMeasurements(
    const Config& config, const std::vector<Object>& pds,
    const std::vector<Object>& gts, ComputeIoUFunc custom_iou_func) {
  CHECK_GT(config.score_cutoffs_size(), 0)
      << "config.scores() must be populated: " << config.DebugString();
  std::unique_ptr<Matcher> matcher = Matcher::Create(config);
  matcher->SetGroundTruths(gts);
  if (custom_iou_func != nullptr) {
    matcher->SetCustomIoUComputeFunc(custom_iou_func);
  }
  // Matcher stores a pointer to pds, so make a copy so pds lives the lifetime
  // of the matcher.
  auto pds_copy = pds;
  if (internal::HasVelocityBreakdown(config) && !pds.empty() &&
      !pds[0].object().metadata().has_accel_x()) {
    pds_copy = internal::EstimateObjectSpeed(pds, gts);
  }
  matcher->SetPredictions(pds_copy);
  std::vector<DetectionMeasurements> measurements;
  const std::vector<internal::BreakdownShardSubset> pd_subsets =
      internal::BuildSubsets(config, matcher->predictions(), /*is_gt=*/false,
                             /*is_detection=*/true);
  const std::vector<internal::BreakdownShardSubset> gt_subsets =
      internal::BuildSubsets(config, matcher->ground_truths(), /*is_gt=*/true,
                             /*is_detection=*/true);
  CHECK_EQ(pd_subsets.size(), gt_subsets.size());

  for (int i = 0, sz = pd_subsets.size(); i < sz; ++i) {
    const internal::BreakdownShardSubset& pd_subset = pd_subsets[i];
    const internal::BreakdownShardSubset& gt_subset = gt_subsets[i];
    // Computes detection measurements for each difficulty level for the
    // breakdown shard.
    std::vector<DetectionMeasurements> measurements_per_breadown_shard =
        ComputeDetectionMeasurementsPerBreakdownShard(config, pd_subset,
                                                      gt_subset, matcher.get());
    for (auto& m : measurements_per_breadown_shard) {
      measurements.emplace_back(std::move(m));
    }
  }

  return measurements;
}

std::vector<DetectionMetrics> ComputeDetectionMetrics(
    const Config& config, const std::vector<std::vector<Object>>& pds,
    const std::vector<std::vector<Object>>& gts,
    ComputeIoUFunc custom_iou_func) {
  std::vector<DetectionMeasurements> measurements;
  CHECK_EQ(pds.size(), gts.size());
  const int num_frames = pds.size();
  const Config config_copy = config.score_cutoffs_size() > 0
                                 ? config
                                 : EstimateScoreCutoffs(config, pds, gts);
  for (int i = 0; i < num_frames; ++i) {
    if (i == 0) {
      measurements = ComputeDetectionMeasurements(config_copy, pds[i], gts[i],
                                                  custom_iou_func);
    } else {
      MergeDetectionMeasurementsVector(
          ComputeDetectionMeasurements(config_copy, pds[i], gts[i],
                                       custom_iou_func),
          &measurements);
    }
  }
  std::vector<DetectionMetrics> metrics;
  metrics.reserve(measurements.size());
  for (auto& m : measurements) {
    metrics.emplace_back(ToDetectionMetrics(config, std::move(m),
                                            config.desired_recall_delta()));
  }
  return metrics;
}

std::vector<DetectionMetrics> ComputeDetectionMetrics(
    const Config& config,
    const std::vector<std::vector<DetectionMeasurements>>& measurements) {
  const int num_frames = measurements.size();
  if (measurements.empty()) return {};
  std::vector<DetectionMeasurements> measurements_merged = measurements[0];
  for (int i = 1; i < num_frames; ++i) {
    MergeDetectionMeasurementsVector(measurements[i], &measurements_merged);
  }
  std::vector<DetectionMetrics> metrics;
  metrics.reserve(measurements_merged.size());
  for (auto& m : measurements_merged) {
    metrics.emplace_back(ToDetectionMetrics(config, std::move(m),
                                            config.desired_recall_delta()));
  }
  return metrics;
}

Config EstimateScoreCutoffs(const Config& config,
                            const std::vector<std::vector<Object>>& pds,
                            const std::vector<std::vector<Object>>& gts) {
  CHECK_EQ(pds.size(), gts.size());
  CHECK_EQ(config.score_cutoffs_size(), 0);
  std::vector<float> pd_scores;
  const int num_frames = pds.size();
  Config config_copy(config);
  for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
    for (int pd_idx = 0, num_pds = pds[frame_idx].size(); pd_idx < num_pds;
         ++pd_idx) {
      pd_scores.push_back(pds[frame_idx][pd_idx].score());
    }
  }
  std::sort(pd_scores.begin(), pd_scores.end());
  std::vector<float> score_cutoffs = internal::DecideScoreCutoffs(
      pd_scores, config.num_desired_score_cutoffs());
  for (auto s : score_cutoffs) {
    config_copy.add_score_cutoffs(s);
  }
  return config_copy;
}

}  // namespace open_dataset
}  // namespace waymo
