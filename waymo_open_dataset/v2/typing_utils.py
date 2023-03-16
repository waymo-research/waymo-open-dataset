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
"""Various utils related to python type annotations."""

import collections
import typing
from typing import Callable, Iterable, TypeVar

_T = TypeVar('_T')


def iterable_item_type(method: Callable[..., Iterable[_T]]) -> type[_T]:
  """Outputs the type of elements in the `Iterable` object returned by `method`.

  Args:
    method: a callable object, which returns an `Iterable` (e.g. a list, tuple
      or a generator).

  Raises:
    AssertionError: if the `method` does not return an iterable.

  Returns:
    Class of the elements in the `Iterable` object returned by the `method`.
  """
  return_type = typing.get_type_hints(method)['return']
  assert typing.get_origin(return_type) == collections.abc.Iterable, return_type
  return typing.get_args(return_type)[0]
