# Copyright 2023 The Waymo Open Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for tensorflow tf.train.Example protocol buffers."""

import numpy as np
import tensorflow as tf


def examples_equal(
    example_a: tf.train.Example,
    example_b: tf.train.Example,
    atol: float = 1e-3,
    rtol: float = 0.0,
) -> bool:
  """Returns true if the given tf.Example protos have identical features.

  Args:
    example_a: The first example to be compared.
    example_b: The second example to be compared.
    atol: The absolute tolerance (see numpy all_close())
    rtol: The relative tolerance (see numpy all_close())
  """
  # Check that feature contents are identical.
  for key, feature in example_a.features.feature.items():
    if key not in example_b.features.feature:
      return False
    feature_b = example_b.features.feature[key]
    if feature_b.HasField('float_list'):
      num_elements = len(feature_b.float_list.value)
      if len(feature.float_list.value) != num_elements:
        return False
      npa = np.array(feature.float_list.value)
      npb = np.array(feature_b.float_list.value)
      if not np.allclose(npa, npb, atol=atol, rtol=rtol):
        print(f'features_differ:\n{feature}\n vs.:\n{feature_b}')
        return False
    else:
      if example_b.features.feature[key] != feature:
        print(f'features_differ:\n{feature}\n vs.:\n{feature_b}')
        return False

  # Check for features in example_b that do not exist in example_a.
  for key in example_b.features.feature:
    if key not in example_a.features.feature:
      return False
  return True
