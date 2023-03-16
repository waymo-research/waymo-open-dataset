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
"""Tests for waymo_open_dataset.v2.dataframe_utils."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import dask.dataframe as dd
import pandas as pd

from waymo_open_dataset.v2 import dataframe_utils as _lib


def _all_rows(result: pd.DataFrame) -> list[dict[str, Any]]:
  return [r.to_dict() for _, r in result.iterrows()]


def create_dask_dataframe(data: dict[str, Any]) -> dd.DataFrame:
  return dd.from_pandas(pd.DataFrame(data=data), npartitions=1)


def create_pandas_dataframe(data: dict[str, Any]) -> pd.DataFrame:
  return pd.DataFrame(data=data)


@parameterized.named_parameters(
    ('pandas', create_pandas_dataframe),
    ('dask', create_dask_dataframe),
)
class MergeTest(parameterized.TestCase, absltest.TestCase):

  def test_single_key_in_common(self, create_dataframe):
    left = create_dataframe({'key.a': [1, 2], 'left': [3, 4]})
    right = create_dataframe({'key.a': [1, 2], 'right': [5, 6]})

    result = _lib.merge(left, right)

    self.assertEqual(result.columns.to_list(), ['key.a', 'left', 'right'])

  def test_two_keys_in_common(self, create_dataframe):
    left = create_dataframe(
        data={
            'key.a': [1, 2, 1],
            'key.b': [2, 2, 1],
            'left': [3, 4, 5],
        }
    )
    right = create_dataframe(
        data={'key.a': [1, 2], 'key.b': [1, 2], 'right': [6, 7]}
    )

    result = _lib.merge(left, right)

    self.assertEqual(
        result.columns.to_list(),
        ['key.a', 'key.b', 'left', 'right'],
    )
    self.assertCountEqual(
        _all_rows(result),
        [
            {'key.a': 1, 'key.b': 1, 'left': 5, 'right': 6},
            {'key.a': 2, 'key.b': 2, 'left': 4, 'right': 7},
        ],
    )

  def test_use_right_merge_when_left_dataframe_is_nullable(
      self, create_dataframe
  ):
    left = create_dataframe({'key.a': [1], 'left': [3]})
    right = create_dataframe({'key.a': [1, 2], 'right': [5, 6]})

    # Replace np.nan, because nan != nan.
    result = _lib.merge(left, right, left_nullable=True).fillna(-1)

    self.assertCountEqual(
        _all_rows(result),
        [
            {'key.a': 1, 'left': 3, 'right': 5},
            {'key.a': 2, 'left': -1, 'right': 6},
        ],
    )

  def test_use_left_merge_when_right_dataframe_is_nullable(
      self, create_dataframe
  ):
    left = create_dataframe({'key.a': [1, 2], 'left': [3, 4]})
    right = create_dataframe({'key.a': [1], 'right': [5]})

    # Replace np.nan, because nan != nan.
    result = _lib.merge(left, right, right_nullable=True).fillna(-1)

    self.assertCountEqual(
        _all_rows(result),
        [
            {'key.a': 1, 'left': 3, 'right': 5},
            {'key.a': 2, 'left': 4, 'right': -1},
        ],
    )

  def test_use_outer_merge_when_both_left_and_right_dataframe_nullable(
      self, create_dataframe
  ):
    left = create_dataframe({'key.a': [1, 2], 'left': [3, 4]})
    right = create_dataframe({'key.a': [1, 3], 'right': [5, 6]})

    # Replace np.nan, because nan != nan.
    result = _lib.merge(
        left, right, left_nullable=True, right_nullable=True
    ).fillna(-1)

    self.assertCountEqual(
        _all_rows(result),
        [
            {'key.a': 1, 'left': 3, 'right': 5},
            {'key.a': 2, 'left': 4, 'right': -1},
            {'key.a': 3, 'left': -1, 'right': 6},
        ],
    )

  def test_group_left_only_columns_when_specified(self, create_dataframe):
    left = create_dataframe(
        data={'key.a': [1, 2, 1], 'key.b': [1, 2, 3], 'left': [3, 4, 5]}
    )
    right = create_dataframe({'key.a': [1, 2], 'right': [6, 7]})

    result = _lib.merge(left, right, left_group=True)

    self.assertCountEqual(
        _all_rows(result),
        [
            {'key.a': 1, 'key.b': [1, 3], 'left': [3, 5], 'right': 6},
            {'key.a': 2, 'key.b': [2], 'left': [4], 'right': 7},
        ],
    )

  def test_group_right_only_columns_when_specified(self, create_dataframe):
    left = create_dataframe({'key.a': [1, 2], 'left': [6, 7]})
    right = create_dataframe(
        data={'key.a': [1, 2, 1], 'key.b': [1, 2, 3], 'right': [3, 4, 5]}
    )

    result = _lib.merge(left, right, right_group=True)

    self.assertCountEqual(
        _all_rows(result),
        [
            {'key.a': 1, 'key.b': [1, 3], 'right': [3, 5], 'left': 6},
            {'key.a': 2, 'key.b': [2], 'right': [4], 'left': 7},
        ],
    )

  def test_group_preserves_original_dtype_of_columns(self, create_dataframe):
    left = create_dataframe(
        data={
            'key.a': [1, 2, 1],
            'key.b': [3, 4, 3],
            'key.c': [1, 2, 3],
            'left': [3, 4, 5],
        }
    )
    left['key.a'] = left['key.a'].astype('int8')
    right = create_dataframe(
        data={'key.a': [1, 2], 'key.b': [3, 4], 'right': [6, 7]}
    )
    right['key.a'] = right['key.a'].astype('int8')

    result = _lib.merge(left, right, left_group=True)

    self.assertEqual(result['key.a'].dtype, left['key.a'].dtype)

  def test_unmatched_keys_without_grouping_false_output_cross_product(
      self, create_dataframe
  ):
    left = create_dataframe(
        data={'key.a': [1, 2, 2], 'key.b': [3, 4, 5], 'left': [6, 7, 8]}
    )
    right = create_dataframe(
        data={'key.a': [1, 1, 2], 'key.c': [-3, -4, -5], 'right': [-6, -7, -8]}
    )

    result = _lib.merge(left, right)

    self.assertCountEqual(
        _all_rows(result),
        [
            {'key.a': 1, 'key.b': 3, 'key.c': -3, 'left': 6, 'right': -6},
            {'key.a': 1, 'key.b': 3, 'key.c': -4, 'left': 6, 'right': -7},
            {'key.a': 2, 'key.b': 4, 'key.c': -5, 'left': 7, 'right': -8},
            {'key.a': 2, 'key.b': 5, 'key.c': -5, 'left': 8, 'right': -8},
        ],
    )


if __name__ == '__main__':
  absltest.main()
