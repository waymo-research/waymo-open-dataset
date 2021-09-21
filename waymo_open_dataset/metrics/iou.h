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

#ifndef WAYMO_OPEN_DATASET_METRICS_IOU_H_
#define WAYMO_OPEN_DATASET_METRICS_IOU_H_

#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/math/polygon2d.h"

namespace waymo {
namespace open_dataset {

// Discretizes IoU from float: [0.0, 1.0] to int: [0, kMaxIoU] as some matching
// algorithms (e.g. Hungarian) require integer version IoU.
extern const int kMaxIoU;

// Computes the intersection of union of two boxes.
// box_type is not specified in b1 and b2 as that is usually the same for all
// boxes in one problem setup.
//
// REQUIRES: The input boxes' length/width/height are >= 0.
double ComputeIoU(const Label::Box& b1, const Label::Box& b2,
                  Label::Box::Type box_type);

// The function type for customized IOU computation function.
typedef double (*ComputeIoUFunc)(const Label::Box&, const Label::Box&);

// Converts a box proto to a polygon.
Polygon2d ToPolygon2d(const open_dataset::Label::Box& box);

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_IOU_H_
