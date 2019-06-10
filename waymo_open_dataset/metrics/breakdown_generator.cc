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

#include "waymo_open_dataset/metrics/breakdown_generator.h"

#include <math.h>

#include <glog/logging.h>
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {
double Hypot3(double a, double b, double c) {
  return std::sqrt(a * a + b * b + c * c);
}
}  // namespace

// This breakdown generator considers everything as one shard.
class BreakdownGeneratorOneShard : public BreakdownGenerator {
 public:
  ~BreakdownGeneratorOneShard() override {}

  int Shard(const Object& object) const override { return 0; }

  Breakdown::GeneratorId Id() const override { return Breakdown::ONE_SHARD; }

  int NumShards() const override { return 1; }

  std::string ShardName(int shard) const override { return "ONE_SHARD"; }
};

// This breakdown generator breaks down the objects based on its object type.
class BreakdownGeneratorObjectType : public BreakdownGenerator {
 public:
  ~BreakdownGeneratorObjectType() override {}

  int Shard(const Object& object) const override {
    // This returns -1 for TYPE_UNKNOWN. -1 indicates that TYPE_UNKNOWN is not
    // in any shard.
    return static_cast<int>(object.object().type() - 1);
  }

  Breakdown::GeneratorId Id() const override { return Breakdown::OBJECT_TYPE; }

  int NumShards() const override { return static_cast<int>(Label::Type_MAX); }

  std::string ShardName(int shard) const override {
    return absl::StrCat(Breakdown::GeneratorId_Name(Id()), "_",
                        Label::Type_Name(static_cast<Label::Type>(shard + 1)));
  }
};

// This breakdown generator breaks down the objects based on their center
// distance (w.r.t. SDC if the box is in vehicle frame).
class BreakdownGeneratorRange : public BreakdownGenerator {
 public:
  ~BreakdownGeneratorRange() override {}

  int Shard(const Object& object) const override {
    double range = Hypot3(object.object().box().center_x(),
                          object.object().box().center_y(),
                          object.object().box().center_z());
    constexpr float kNearRange = 30.0;
    constexpr float kMidRange = 50.0;
    const int shard_offset = 3 * (object.object().type() - 1);
    if (shard_offset < 0) {
      return -1;
    }
    if (range < kNearRange) {
      return 0 + shard_offset;
    } else if (range < kMidRange) {
      return 1 + shard_offset;
    } else {
      return 2 + shard_offset;
    }
  }

  Breakdown::GeneratorId Id() const override { return Breakdown::RANGE; }

  int NumShards() const override {
    return 3 * static_cast<int>(Label::Type_MAX);
  }

  std::string ShardName(int shard) const override {
    const Label::Type object_type = static_cast<Label::Type>(shard / 3 + 1);
    CHECK_LE(object_type, Label::Type_MAX) << shard;
    CHECK_GE(object_type, 1) << shard;

    const std::string prefix = absl::StrCat(Breakdown::GeneratorId_Name(Id()),
                                            "_", Label::Type_Name(object_type));
    const int range_shard = shard % 3;
    switch (range_shard) {
      case 0:
        return absl::StrCat(prefix, "_", "[0, 30)");
      case 1:
        return absl::StrCat(prefix, "_", "[30, 50)");
      case 2:
        return absl::StrCat(prefix, "_", "[50, +inf)");
      default:
        LOG(FATAL) << "Code should not reach here.";
    }
  }
};

std::unique_ptr<BreakdownGenerator> BreakdownGenerator::Create(
    Breakdown::GeneratorId id) {
  switch (id) {
    case Breakdown::OBJECT_TYPE:
      return absl::make_unique<BreakdownGeneratorObjectType>();
    case Breakdown::ONE_SHARD:
      return absl::make_unique<BreakdownGeneratorOneShard>();
    case Breakdown::RANGE:
      return absl::make_unique<BreakdownGeneratorRange>();
    default:
      LOG(FATAL) << "Unimplemented breakdown generator "
                 << Breakdown::GeneratorId_Name(id);
  }
}

}  // namespace open_dataset
}  // namespace waymo
