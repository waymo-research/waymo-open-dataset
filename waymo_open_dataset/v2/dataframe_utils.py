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
"""Utils to work with Pandas and Dask DataFrames."""

from typing import TypeVar, Union

import dask.dataframe as dd
import pandas as pd


_AnyDataFrame = Union[pd.DataFrame, dd.DataFrame]
_DataFrame = TypeVar('_DataFrame')


def _select_key_columns(df: _AnyDataFrame, prefix: str) -> set[str]:
  return set([c for c in df.columns if c.startswith(prefix)])


def _how(left_nullable: bool = False, right_nullable: bool = False):
  if left_nullable and right_nullable:
    return 'outer'
  elif left_nullable and not right_nullable:
    return 'right'
  elif not left_nullable and right_nullable:
    return 'left'
  else:
    return 'inner'


def _cast_keys(src, dst, keys):
  for key in keys:
    if dst.dtypes[key] != src.dtypes[key]:
      dst[key] = dst[key].astype(src[key].dtype)


def _group_by(src, keys):
  dst = src.groupby(list(keys)).agg(list).reset_index()
  # Fix key types automatically created from the MultiIndex
  _cast_keys(src, dst, keys)
  return dst


def merge(
    left: _DataFrame,
    right: _DataFrame,
    left_nullable: bool = False,
    right_nullable: bool = False,
    left_group: bool = False,
    right_group: bool = False,
    key_prefix: str = 'key.',
) -> _DataFrame:
  """Merges two tables using automatically select columns.

  This operation is called JOIN in SQL, but we use "merge" to make it consistent
  with Pandas and Dask.

  If the sets of key columns in the left and right tables do not match it will
  group by shared columns first to avoid unexpected cross products of unmatched
  columns.

  When both `left_nullable` and `right_nullable` are set to True it will perform
  an outer JOIN and output all rows from both tables in the output. When both
  set to False it will perform INNER join, otherwise - LEFT or RIGHT joins
  accordingly.

  Args:
    left: a left table.
    right: a right table.
    left_nullable: if True output may contain rows where only right columns are
      present, while left columns are null.
    right_nullable: if True output may contain rows where only left columns are
      present, while right columns are null.
    left_group: If True it will group records in the left table by common keys.
    right_group: If True it will group records in the right table by common
      keys.
    key_prefix: a string prefix used to select key columns.

  Returns:
    A new table which is the result of the join operation.
  """
  left_keys = _select_key_columns(left, key_prefix)
  right_keys = _select_key_columns(right, key_prefix)
  common_keys = left_keys.intersection(right_keys)
  if left_group and left_keys != common_keys:
    left = _group_by(left, common_keys)
  if right_group and right_keys != common_keys:
    right = _group_by(right, common_keys)
  return left.merge(
      right, on=list(common_keys), how=_how(left_nullable, right_nullable)
  )
