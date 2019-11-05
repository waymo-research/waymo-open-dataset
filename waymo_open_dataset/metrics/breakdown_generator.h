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

#ifndef WAYMO_OPEN_DATASET_METRICS_BREAKDOWN_GENERATOR_H_
#define WAYMO_OPEN_DATASET_METRICS_BREAKDOWN_GENERATOR_H_

#include <memory>
#include <string>

#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {

// Base class for metrics breakdown generators.
// A breakdown generator defines a way to shard a set of Objects such that users
// can compute metrics for different subsets of objects. Each breakdown
// generator comes with a unique breakdown generator ID.
class BreakdownGenerator {
 public:
  virtual ~BreakdownGenerator() = default;

  // Returns the shard of this 'object' (either a groundtruth or a prediction).
  // Returns -1 if the object does not belong to any shard.
  virtual int Shard(const Object& object) const = 0;

  // Returns a list of shards that this object should be included while
  // performing matching when computing metrics.
  //
  // Some breakdowns require to perform a matching between predictions and
  // ground truths beyond objects in its own shard (defined by 'Shard' above).
  // This is only enabled for VELOCITY breakdown in detection metrics
  // computation for now.
  virtual std::vector<int> ShardsForMatching(const Object& object) const {
    return {Shard(object)};
  }

  // The unique ID of this breakdown method.
  virtual Breakdown::GeneratorId Id() const = 0;

  // The total number of shards the generator produces.
  virtual int NumShards() const = 0;

  // The name of the shard used for UI.
  virtual std::string ShardName(int i) const = 0;

  // Whether this is a groundtruth-only breakdown. A groundtruth-only
  // breakdown shards on properties that are only available on a groundtruth,
  // such as velocity in a detection problem.
  virtual bool IsGroundTruthOnlyBreakdown() const { return false; }

  // Creates a breakdown generator given a breadown generator ID.
  static std::unique_ptr<BreakdownGenerator> Create(Breakdown::GeneratorId id);
};

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_BREAKDOWN_GENERATOR_H_
