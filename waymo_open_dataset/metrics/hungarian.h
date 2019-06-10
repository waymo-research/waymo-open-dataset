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

// Purpose: Find a maximum weight matching in a complete bipartite graph.

#ifndef WAYMO_OPEN_DATASET_METRICS_HUNGARIAN_H_
#define WAYMO_OPEN_DATASET_METRICS_HUNGARIAN_H_

namespace waymo {
namespace open_dataset {

// The type of the edge weights. In the current implementation, the weights
// must be an integer type, since we need exact arithmetic.
// The algorithm itself also requires that the weights are non-negative.
using weight_t = int;

// Given a complete bipartite graph 'edge', where edge[i*n + j] is the
// non-negative weight of the edge which joins node i and node j,
// find an optimal assignment for the node sets 'i' and 'j'.
// On exit, the pairs (i, perm[i]) for i in [0, n-1] contain the edges
// of the matching.
void Hungarian(int n, const weight_t* edge, int* perm);

// This version is exposed for unit testing. In addition to the matching,
// it also returns the feasible labeling which was estimated by the algorithm.
void Hungarian(int n, const weight_t* edge, int* perm, weight_t* lx,
               weight_t* ly);

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_HUNGARIAN_H_
