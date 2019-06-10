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

#include "waymo_open_dataset/metrics/hungarian.h"

#include <stdlib.h>

#include <algorithm>
#include <functional>
#include <set>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace waymo {
namespace open_dataset {
namespace {

// Make a random complete bipartite graph and store the edge weights
// in the row-major array 'w', so that w[i*n + j] is the weight
// of the edge (i, j).
void MakeRandomBipartiteGraph(int n, double frac_zeros, std::vector<int>* w) {
  w->clear();
  w->resize(n * n, 0);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (drand48() >= frac_zeros) (*w)[i * n + j] = lrand48() % 1024;
    }
  }
}

// This function tests the max weight matching function
// by verifying that we have a complete matching in the
// equality subgraph.
void TestRandomWithConstraintVerification(int n, double frac_zeros) {
  std::vector<int> w;      // edge weights
  std::vector<int> p(n);   // permutation of [0, n-1], i.e. the matching.
  std::vector<int> lx(n);  // feasible labeling of the rows of 'w'.
  std::vector<int> ly(n);  // feasible labeling of the cols of 'w'.

  MakeRandomBipartiteGraph(n, frac_zeros, &w);
  Hungarian(n, &w[0], &p[0], &lx[0], &ly[0]);

  // Check that lx, ly is a feasible labeling.
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) CHECK_GE(lx[i] + ly[j], w[i * n + j]);

  // Check that the returned permutation represents a complete matching,
  // or in other words that 'p' is a permutation of [0, n-1].
  std::vector<int> q = p;
  std::sort(q.begin(), q.end());
  for (int i = 0; i < n; ++i) CHECK_EQ(i, q[i]);

  // Check that matched edges are all in the equality subgraph
  // defined by lx[i] + ly[j] == w[i*n + j].
  for (int i = 0; i < n; ++i) {
    int j = p[i];
    CHECK_EQ(lx[i] + ly[j], w[i * n + j]);
  }
}

TEST(Hungarian, TestManyRandomGraphsIndirectly) {
  for (int i = 1; i < 100; ++i) {
    TestRandomWithConstraintVerification(i, 0.0);
    TestRandomWithConstraintVerification(i, 0.01);
    TestRandomWithConstraintVerification(i, 0.99);
    TestRandomWithConstraintVerification(i, 1.0);
  }
}

// This function tests the max weight matching function
// to the result found by using a naive exhaustive search
// of all n! permutations.
void TestRandomWithExhaustiveVerification(
    int n, double frac_zeros,
    const std::function<void(int, const int*, int*)>& align_func) {
  std::vector<int> w;     // edge weights
  std::vector<int> p(n);  // permutation of [0, n-1], i.e. the matching.

  // Find the best permutation using the hungarian algorithm.
  MakeRandomBipartiteGraph(n, frac_zeros, &w);
  align_func(n, &w[0], &p[0]);

  // Find the best permutation by trying all n! of them.

  // Start with the lexicographically smallest permutation 0,1,2...n-1.
  std::vector<int> cur_p(n);
  int sum = 0;

  for (int i = 0; i < cur_p.size(); ++i) {
    cur_p[i] = i;
    sum += w[i * n + i];
  }

  // Keep track of the permutation(s) with max weight.
  // There may be several with the same weight sum.
  std::vector<std::vector<int> > best_ps;
  best_ps.push_back(cur_p);
  int max_weight = sum;

  while (std::next_permutation(cur_p.begin(), cur_p.end())) {
    sum = 0;
    for (int i = 0; i < cur_p.size(); ++i) sum += w[i * n + cur_p[i]];

    if (sum > max_weight) {
      best_ps.clear();
      best_ps.push_back(cur_p);
      max_weight = sum;
    } else if (sum == max_weight) {
      best_ps.push_back(cur_p);
    }
  }

  int best_calculated = 0;
  for (int i = 0; i < n; i++) {
    if (p[i] >= 0) {
      best_calculated += w[i * n + p[i]];
    }
  }

  // If the weight is equal and there's no same elements in result(other than
  // "not found") the answer is OK.
  std::set<int> existed;
  bool no_collide = true;
  for (int i = 0; i < n; i++) {
    if (p[i] == -1) continue;
    if (existed.find(p[i]) != existed.end()) {
      no_collide = false;
      break;
    }
    existed.insert(p[i]);
  }
  if (best_calculated == max_weight && no_collide) {
    return;
  }

  LOG(FATAL) << "Estimated permutation is not optimal";
}

TEST(Hungarian, TestSmallGraphsExhaustively) {
  typedef void (*FuncType)(int, const int*, int*);
  std::function<void(int, const int*, int*)> align_func =
      static_cast<FuncType>(Hungarian);
  for (int i = 1; i < 10; ++i) {
    TestRandomWithExhaustiveVerification(i, 0.0, align_func);
    TestRandomWithExhaustiveVerification(i, 0.1, align_func);
    TestRandomWithExhaustiveVerification(i, 0.9, align_func);
    TestRandomWithExhaustiveVerification(i, 1.0, align_func);
  }
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
