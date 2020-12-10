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

#include "waymo_open_dataset/metrics/tracking_metrics.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>

#include <glog/logging.h>
#include "waymo_open_dataset/metrics/breakdown_generator.h"
#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/metrics/metrics_utils.h"
#include "waymo_open_dataset/metrics/mot.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {
// Computes tracking measurements for a single breakdown shard.
// Returns a vector of tracking measurements. Each element of that corresponds
// to a difficulty level. The order of the vector is same as the difficulty
// levels specified in the config.
// pd_subsets, gt_subsets: dim-0 is indexed by frame, dim-1 is indexed by
// breakdown shard. breakdown_shard_index specifies which breakdown shard
// (dim-1) to consider in this function.
// matcher_ptrs is indexed by frame.
std::vector<TrackingMeasurements> ComputeTrackingMeasurementPerBreakdownShard(
    const Config& config,
    const std::vector<std::vector<internal::BreakdownShardSubset>>& pd_subsets,
    const std::vector<std::vector<internal::BreakdownShardSubset>>& gt_subsets,
    int breakdown_shard_index,
    std::vector<std::unique_ptr<Matcher>>* matcher_ptrs) {
  CHECK(matcher_ptrs != nullptr);
  std::vector<std::unique_ptr<Matcher>>& matchers = *matcher_ptrs;
  const int num_frames = pd_subsets.size();
  if (num_frames <= 0) return {};
  const int breakdown_generator_id_index =
      pd_subsets[0][breakdown_shard_index].breakdown_generator_id_index;

  const std::vector<Label::DifficultyLevel> difficulty_levels =
      internal::GetDifficultyLevels(config, breakdown_generator_id_index);
  std::vector<TrackingMeasurements> measurements;
  measurements.resize(difficulty_levels.size());

  for (int i = 0, sz = difficulty_levels.size(); i < sz; ++i) {
    auto* breakdown = measurements[i].mutable_breakdown();
    breakdown->set_generator_id(config.breakdown_generator_ids(
        pd_subsets[0][breakdown_shard_index].breakdown_generator_id_index));
    breakdown->set_shard(pd_subsets[0][breakdown_shard_index].breakdown_shard);
    breakdown->set_difficulty_level(difficulty_levels[i]);
  }

  // For each score cutoff and difficulty level, run multi object tracking
  // metrics computation.
  for (int score_idx = 0, scores_sz = config.score_cutoffs_size();
       score_idx < scores_sz; ++score_idx) {
    for (int dl_idx = 0, dl_sz = difficulty_levels.size(); dl_idx < dl_sz;
         ++dl_idx) {
      MOT mot;
      for (int frame_index = 0; frame_index < num_frames; ++frame_index) {
        matchers[frame_index]->SetPredictionSubset(
            pd_subsets[frame_index][breakdown_shard_index].indices[score_idx]);
        // All score cutoffs share the same ground truth subset in a frame.
        matchers[frame_index]->SetGroundTruthSubset(
            gt_subsets[frame_index][breakdown_shard_index].indices[0]);

        mot.Eval(matchers[frame_index].get(), difficulty_levels[dl_idx]);
      }
      *measurements[dl_idx].add_measurements() = mot.measurement();
      measurements[dl_idx].mutable_measurements()->rbegin()->set_score_cutoff(
          config.score_cutoffs(score_idx));
    }
  }
  return measurements;
}

// Merges two tracking measurements.
TrackingMeasurement MergeTrackingMeasurement(const TrackingMeasurement& m1,
                                             const TrackingMeasurement& m2) {
  CHECK_EQ(m1.score_cutoff(), m2.score_cutoff());
  TrackingMeasurement m;
#define ADD_FIELD(FIELD_NAME) \
  m.set_##FIELD_NAME(m1.FIELD_NAME() + m2.FIELD_NAME())
  ADD_FIELD(num_misses);
  ADD_FIELD(num_fps);
  ADD_FIELD(num_mismatches);
  ADD_FIELD(matching_cost);
  ADD_FIELD(num_matches);
  ADD_FIELD(num_objects_gt);
#undef ADD_FIELD
  m.set_score_cutoff(m1.score_cutoff());
  return m;
}

// Merges new_m to m element by element.
void MergeTrackingMeasurements(const TrackingMeasurements& new_m,
                               TrackingMeasurements* m) {
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

  CHECK_EQ(m->measurements_size(), new_m.measurements_size());
  for (int i = 0, sz = m->measurements_size(); i < sz; ++i) {
    *m->mutable_measurements(i) =
        MergeTrackingMeasurement(m->measurements(i), new_m.measurements(i));
  }
}

// Merges new_m to m element by element.
void MergeTrackingMeasurementsVector(
    const std::vector<TrackingMeasurements>& new_m,
    std::vector<TrackingMeasurements>* m) {
  CHECK(m != nullptr);
  if (m->empty()) {
    *m = new_m;
    return;
  }
  if (new_m.empty()) {
    return;
  }
  CHECK_EQ(new_m.size(), m->size());
  for (int i = 0, sz = m->size(); i < sz; ++i) {
    MergeTrackingMeasurements(new_m[i], &(*m)[i]);
  }
}

// Converts tracking measurements to tracking metrics. It compute MOT metrics
// for each score cutoff and then picks the one with highest MOTA.
TrackingMetrics ToTrackingMetrics(TrackingMeasurements&& measurements) {
  TrackingMetrics metrics;
  *metrics.mutable_measurements() = measurements;
  const auto& m = metrics.measurements();
  *metrics.mutable_breakdown() = m.breakdown();

  float miss_ratio = 0.0;
  float fp_ratio = 0.0;
  float mismatch_ratio = 0.0;
  float mota = 0.0;
  float motp = 0.0;

  // Find the score cutoff that yields best MOTA.
  for (const auto& measurement : m.measurements()) {
    if (measurement.num_objects_gt() == 0) continue;
    const float num_objects_gt = measurement.num_objects_gt();
    miss_ratio = measurement.num_misses() / num_objects_gt;
    fp_ratio = measurement.num_fps() / num_objects_gt;
    mismatch_ratio = measurement.num_mismatches() / num_objects_gt;
    mota = 1.0 - (miss_ratio + fp_ratio + mismatch_ratio);

    // "<" means that we prefer lower score cutoffs when MOTA ties.
    if (metrics.mota() < mota) {
      metrics.set_mota(mota);
      if (measurement.num_matches() > 0) {
        motp = measurement.matching_cost() / measurement.num_matches();
      } else {
        motp = 0.0;
      }
      metrics.set_motp(motp);
      metrics.set_miss(miss_ratio);
      metrics.set_mismatch(mismatch_ratio);
      metrics.set_fp(fp_ratio);
      metrics.set_score_cutoff(measurement.score_cutoff());
    }
  }
  return metrics;
}
}  // namespace

std::vector<TrackingMeasurements> ComputeTrackingMeasurements(
    const Config& config, const std::vector<std::vector<Object>>& pds,
    const std::vector<std::vector<Object>>& gts) {
  // Create one matcher per frame.
  std::vector<std::unique_ptr<Matcher>> matchers;
  const int num_frames = pds.size();
  CHECK_EQ(num_frames, gts.size());

  matchers.reserve(num_frames);
  for (int i = 0; i < num_frames; ++i) {
    matchers.push_back(Matcher::Create(config));
  }
  const bool need_to_estimate_speed =
      internal::HasVelocityBreakdown(config) && !pds.empty() &&
      !pds[0].empty() && !pds[0][0].object().metadata().has_speed_x();
  std::vector<std::vector<Object>> pds_with_velocity;
  if (need_to_estimate_speed) {
    pds_with_velocity = internal::EstimateObjectSpeed(pds, gts);
  }
  for (int i = 0; i < num_frames; ++i) {
    matchers[i]->SetPredictions(need_to_estimate_speed ? pds_with_velocity[i]
                                                       : pds[i]);
    matchers[i]->SetGroundTruths(gts[i]);
  }

  std::vector<std::vector<internal::BreakdownShardSubset>> pd_subsets(
      num_frames);
  std::vector<std::vector<internal::BreakdownShardSubset>> gt_subsets(
      num_frames);
  int num_breakdown_shards = -1;
  for (int i = 0; i < num_frames; ++i) {
    pd_subsets[i] = internal::BuildSubsets(
        config, (need_to_estimate_speed ? pds_with_velocity[i] : pds[i]),
        /*is_gt=*/false, /*is_detection=*/false);
    gt_subsets[i] = internal::BuildSubsets(config, gts[i], /*is_gt=*/true,
                                           /*is_detection=*/false);

    if (num_breakdown_shards < 0) {
      num_breakdown_shards = pd_subsets[i].size();
    }

    // Subsets in every frame must have the same size.
    CHECK_EQ(num_breakdown_shards, pd_subsets[i].size());
    CHECK_EQ(num_breakdown_shards, gt_subsets[i].size());
  }

  std::vector<TrackingMeasurements> measurements;
  for (int i = 0; i < num_breakdown_shards; ++i) {
    std::vector<TrackingMeasurements> measurements_per_breadown_shard =
        ComputeTrackingMeasurementPerBreakdownShard(config, pd_subsets,
                                                    gt_subsets, i, &matchers);
    for (auto& m : measurements_per_breadown_shard) {
      measurements.emplace_back(std::move(m));
    }
  }
  return measurements;
}

std::vector<TrackingMetrics> ComputeTrackingMetrics(
    const Config& config,
    const std::vector<std::vector<TrackingMeasurements>>& measurements) {
  std::vector<TrackingMeasurements> measurements_merged;
  const int num_scenes = measurements.size();
  for (int scene_idx = 0; scene_idx < num_scenes; ++scene_idx) {
    MergeTrackingMeasurementsVector(measurements[scene_idx],
                                    &measurements_merged);
  }
  std::vector<TrackingMetrics> metrics;
  metrics.reserve(measurements_merged.size());
  for (auto& m : measurements_merged) {
    metrics.emplace_back(ToTrackingMetrics(std::move(m)));
  }
  return metrics;
}

std::vector<TrackingMetrics> ComputeTrackingMetrics(
    const Config& config,
    const std::vector<std::vector<std::vector<Object>>>& pds,
    const std::vector<std::vector<std::vector<Object>>>& gts) {
  const int num_scenes = pds.size();
  const Config config_copy = config.score_cutoffs_size() > 0
                                 ? config
                                 : EstimateScoreCutoffs(config, pds, gts);

  std::vector<TrackingMeasurements> measurements;
  for (int scene_idx = 0; scene_idx < num_scenes; ++scene_idx) {
    MergeTrackingMeasurementsVector(
        ComputeTrackingMeasurements(config_copy, pds[scene_idx],
                                    gts[scene_idx]),
        &measurements);
  }
  std::vector<TrackingMetrics> metrics;
  metrics.reserve(measurements.size());
  for (auto& m : measurements) {
    metrics.emplace_back(ToTrackingMetrics(std::move(m)));
  }
  return metrics;
}

Config EstimateScoreCutoffs(
    const Config& config,
    const std::vector<std::vector<std::vector<Object>>>& pds,
    const std::vector<std::vector<std::vector<Object>>>& gts) {
  CHECK_EQ(pds.size(), gts.size());
  const int num_scenes = pds.size();
  if (num_scenes == 0) return {};
  Config config_copy(config);
  if (config.score_cutoffs_size() == 0) {
    CHECK_GT(config.num_desired_score_cutoffs(), 0);
    std::vector<float> pd_scores;
    for (int scene_idx = 0; scene_idx < num_scenes; ++scene_idx) {
      const int num_frames = pds[scene_idx].size();
      for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        for (int pd_idx = 0, num_pds = pds[frame_idx].size(); pd_idx < num_pds;
             ++pd_idx) {
          pd_scores.push_back(pds[scene_idx][frame_idx][pd_idx].score());
        }
      }
    }
    std::sort(pd_scores.begin(), pd_scores.end());
    std::vector<float> score_cutoffs = internal::DecideScoreCutoffs(
        pd_scores, config.num_desired_score_cutoffs());
    for (auto s : score_cutoffs) {
      config_copy.add_score_cutoffs(s);
    }
  }
  return config_copy;
}

}  // namespace open_dataset
}  // namespace waymo
