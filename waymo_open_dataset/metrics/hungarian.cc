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

#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>

#include <glog/logging.h>

// This file contains an implementation of Kuhn and Munkres' algorithm for
// finding a maximum weight matching in a complete weighted bipartite graph.
// The algorithm is also known as the 'Hungarian' algorithm.
//
// A complete bipartite weighted graph can be defined by two vertex sets,
// indexed by 'i' and 'j' resp., and an edge weight function w(i, j).
// A "vertex labeling" is defined by two real-valued functions lx(i) and ly(j),
// such that lx(i) + ly(j) >= w(i, j) for all pairs (i, j).
//
// A vertex labeling defines an "equality subgraph" consisting of the edges for
// for which lx(i) + ly(j) == w(i, j).
//
// The algorithm is based on a theorem stating that a complete matching in the
// equality subgraph w.r.t a vertex labeling (lx, ly) is also a maximum weight
// matching in the original graph. The main idea is to alternate between two
// steps:
//         1) increase the size of the matching in the equality subgraph,
//         2) improve the vertex labeling to facilitate (1).

namespace waymo {
namespace open_dataset {
namespace {
// A set of integers with the following properties:
//   1. Remembers the order in which elements were added,
//   2. Can in constant time tell you if a given integer is a member.
class OrderedSet {
 public:
  // Creates an empty set which can hold integers between 0 and n-1.
  explicit OrderedSet(int n) : M_(n) {}

  void clear() {
    int n = M_.size();
    M_.clear();
    M_.resize(n, false);
    I_.clear();
  }
  bool has(int i) const { return M_[i]; }

  void insert(int i) {
    if (!M_[i]) {
      M_[i] = true;
      I_.push_back(i);
    }
  }

  int size() const { return I_.size(); }

  // Returns the ith member.
  int operator[](int idx) const { return I_[idx]; }

  bool operator==(const OrderedSet& s) const {
    if (size() != s.size()) return false;

    for (int k = 0; k < size(); ++k) {
      if (!s.has(I_[k])) return false;
    }
    return true;
  }

 private:
  std::vector<bool> M_;
  std::vector<int> I_;
};

// Contains a partial or complete matching in a bipartite graph.
// Stores a permutation and an inverse permutation of the integers [0, n-1].
class Matching {
 public:
  explicit Matching(int n) : p_(n, -1), ip_(n, -1) {}

  void Add(int i, int j) {
    p_[i] = j;
    ip_[j] = i;
  }

  int GetJ(int i) const { return p_[i]; }
  int GetI(int j) const { return ip_[j]; }

  // Return an unmatched 'i' vertex.
  // If no such vertex exists, returns n.
  int FindUnmatchedI() const {
    return std::find(p_.begin(), p_.end(), -1) - p_.begin();
  }

  void CopyTo(int* p) {
    for (int i = 0; i < p_.size(); ++i) p[i] = p_[i];
  }

 private:
  std::vector<int> p_;
  std::vector<int> ip_;
};

// Add to 'N' all the 'j' vertices that are joined to 'i' by an edge in
// the equality subgraph.
void AddNeighbours(int i, int parent_index, int n, const weight_t* edge,
                   const weight_t* lx, const weight_t* ly, OrderedSet* N,
                   std::vector<int>* parentN) {
  int numN = N->size();

  for (int j = 0; j < n; ++j) {
    if (lx[i] + ly[j] == edge[i * n + j]) {
      N->insert(j);
    }
  }
  parentN->insert(parentN->end(), N->size() - numN, parent_index);
}

// As above, but perform the operation for each vertex stored in 'S'.
void FindNeighbours(const std::vector<int>& S, int n, const weight_t* edge,
                    const weight_t* lx, const weight_t* ly, OrderedSet* N,
                    std::vector<int>* parentN) {
  N->clear();
  parentN->clear();

  for (int ii = 0; ii < S.size(); ++ii) {
    AddNeighbours(S[ii], ii, n, edge, lx, ly, N, parentN);
  }
}

// Improves the feasible labeling so that there is at least one vertex
// of the equality subgraph reachable from S that is not in T.
void ImproveFeasibleLabeling(int n, const weight_t* edge,
                             const std::vector<int>& S, const OrderedSet& T,
                             weight_t* lx, weight_t* ly) {
  weight_t a = std::numeric_limits<weight_t>::max();

  for (int ii = 0; ii < S.size(); ++ii) {
    int i = S[ii];
    for (int j = 0; j < n; ++j) {
      if (!T.has(j)) {
        a = std::min(a, lx[i] + ly[j] - edge[i * n + j]);
      }
    }
  }

  for (int i = 0; i < S.size(); ++i) lx[S[i]] -= a;
  for (int j = 0; j < T.size(); ++j) ly[T[j]] += a;
}

// Augments the matching in 'm' by 1, by finding a path
// from T[T.size(0-1] to S[0] in the alternating tree defined
// by S, T, and parentT.
//
// parentT[k] is an index into S of the parent of T[k].
void AugmentMatching(const std::vector<int>& S, const OrderedSet& T,
                     const std::vector<int>& parentT, Matching* m) {
  CHECK(S.size() == T.size());

  // The leaf of the alternating tree. It should be unmatched.
  int j = T[T.size() - 1];
  CHECK_EQ(m->GetI(j), -1);

  // Find an alternating path from the leaf to the root.
  // An alternating path is a path in which the edges alternate
  // between unmatched and matched.
  for (int k = T.size() - 1; k >= 0; --k) {
    if (T[k] == j) {
      int ii = parentT[k];
      int i = S[ii];

      // (i, j) is an unmatched edge. Add it to the matching.
      // tmp_j tells us how to get one step further up the tree.
      int tmp_j = m->GetJ(i);
      m->Add(i, j);
      j = tmp_j;

      // If we have reached the root of the alternating tree, we are done.
      if (ii == 0) {
        // Checks that the root was unmatched when we got to it.
        CHECK_EQ(tmp_j, -1);
        return;
      }
    }
  }

  // If we get here, we have a bug somewhere in this file...
  LOG(FATAL);
}

// Returns an index to a member of the set N that is not in T.
// The function assumes there is at least one such element.
int PickY(const OrderedSet& N, const OrderedSet& T) {
  for (int j = 0; j < N.size(); ++j) {
    if (!T.has(N[j])) return j;
  }

  // If we get here, we shouldn't have called this function in the first place.
  LOG(FATAL);
  return -1;
}
}  // namespace

void Hungarian(int n, const weight_t* edge, int* perm) {
  std::vector<weight_t> lx(n);
  std::vector<weight_t> ly(n);
  Hungarian(n, edge, perm, &lx[0], &ly[0]);
}

void Hungarian(int n, const weight_t* edge, int* perm, weight_t* lx,
               weight_t* ly) {
  CHECK_GT(n, 0);
  CHECK(edge);
  CHECK(perm);
  CHECK(lx);
  CHECK(ly);

  // Find initial feasible labeling: lx[i] + ly[j] >= edge[i][j] for all (i, j).
  for (int i = 0; i < n; ++i) {
    lx[i] = 0;
    ly[i] = 0;
    for (int j = 0; j < n; ++j) {
      lx[i] = std::max(lx[i], edge[i * n + j]);
    }
  }

  // This is where we build the complete matching.
  // It stores the permutation and the inverse permutation.
  Matching M(n);

  // We use the following variables to build an alternating tree - a tree where
  // the root is an unmatched vertex and the tree edges alternate between
  // unmatched and matched.

  // Tree nodes in the 'i' vertex set. S[0] is the root of the tree.
  std::vector<int> S;

  // Tree nodes in the 'j' vertex set.
  OrderedSet T(n);
  std::vector<int> parentT;  // for each element in T, contains an index into S.

  // Vertices reachable from S in the equality subgraph defined by lx and ly.
  OrderedSet N(n);
  std::vector<int> parentN;  // for each element in N, contains index into S.

  while (true) {
    // Find an unmatched vertex in the 'i' vertex set.
    int i = M.FindUnmatchedI();

    // If there is no free vertex, we have a complete matching and are done.
    if (i == n) {
      M.CopyTo(perm);
      return;
    }

    // Clear the alternating tree.
    S.clear();
    T.clear();
    parentT.clear();

    // Start a new tree. The root is xi.
    S.push_back(i);
    FindNeighbours(S, n, edge, lx, ly, &N, &parentN);

    // In each iteration, we make progress either by adding a node
    // to the alternating tree, or by augmenting the matching by 1.
    while (true) {
      // If we have used up all members of N, we need to update labels.
      if (N == T) {
        ImproveFeasibleLabeling(n, edge, S, T, lx, ly);
        FindNeighbours(S, n, edge, lx, ly, &N, &parentN);
      }

      // Pick a 'j' vertex in N - T
      int jj = PickY(N, T);
      int j = N[jj];
      T.insert(j);
      parentT.push_back(parentN[jj]);

      // If j is free, there is an alternating path between S[0] and j, so
      // we can augment the matching. If not, we extend the alternating tree.
      int z = M.GetI(j);
      if (z == -1) {
        AugmentMatching(S, T, parentT, &M);
        break;
      } else {
        S.push_back(z);
        AddNeighbours(z, S.size() - 1, n, edge, lx, ly, &N, &parentN);
      }
    }
  }
}

}  // namespace open_dataset
}  // namespace waymo
