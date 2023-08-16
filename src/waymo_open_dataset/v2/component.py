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
"""Classes and functions to define v2 dataset components."""
import abc
import collections
import dataclasses
import types
import typing
from typing import Any, Callable, Optional, Type, TypeVar, Union

import pyarrow as pa

# Keys for dataclasses.Field metadata to store extra information for columns.
_META_IS_REPEATED = 'is_repeated'
_META_ARROW_TYPE = 'arrow_type'

_T = TypeVar('_T')


class Component(metaclass=abc.ABCMeta):
  """Base class for all dataset components.

  This class is used as a base class for all component dataclasses:
  - to "annotate" a top level component dataclass as a component (this can be
    checked by calling `isinstance` method)
  - to add functionality to convert dataclasses into flattened dictionaries and
    back (used to store data on disk in columnar format).

  All child classes have to have a field `key` with a type `Key`.

  As an example, here is how to define the MyComponent class:
    class Foo:
      name: str = create_column(arrow_type=pa.list_(pa.string()))
      value: int = create_column(arrow_type=pa.list_(pa.int32()))

    class Bar:
      foo: Foo

    class Baz:
      value: float = create_column(arrow_type=pa.float32())

    class MyKey(Key):
      context_name: str = create_column(arrow_type=pa.string())
      object_id: str = create_column(arrow_type=pa.string())

    class MyComponent(Component):
      key: MyKey
      bar: Bar = create_column(is_repeated=True)
      baz: Baz
  """

  @classmethod
  def from_dict(cls: Type[_T], columns: dict[str, Any]) -> _T:
    """Creates an instance of the component from a flat dictionary.

    Args:
      columns: a flat dictionary with (key, value) pairs where values are python
        numerical or string values, or list of numbers or strings, or tensors.

    Returns:
      An instance of the Component class.
    """
    return unflatten_dict(cls, columns)

  def to_flatten_dict(self) -> dict[str, Any]:
    """Returns flat dict with all fields.

    Usually a component has a number of fields which are dataclasses and their
    fields are dataclasses and so on. Thus all fields in a component form a tree
    where only leafs store actual data and non-leaf nodes provide a convenient
    access the data.

    For MyComponent from the example in the class' docstring `to_flatten_dict()`
    will will return a dictionary with the following keys:
       key.context_name
       key.object_id
       [MyComponent].bar[*].foo.name
       [MyComponent].bar[*].foo.value
       [MyComponent].baz.value

    This way same keys from different components will have matching column
    names. While fields from different components with same name will not
    conflict when we join data from multiple components. Annotation "bar[*]"
    means that the bar field is repeated and its values were concatentation into
    a list, i.e.

    comp = MyComponent(...)
    columns = comp.to_flatten_dict()
    i: int = ...
    assert comp.bar[i].foo.name == columns['comp.bar[*].foo.name'][i]

    For more examples see refer to component_test.py.

    Returns:
      A dict with all columns for the component with (key, value) pairs, where
      keys generated from the dataclass fields, but concatenating them with '.'
      and added prefixes to distinguish component fields from keys and
      annotations for repeated fields.
    """
    output = collections.OrderedDict()
    flatten_dict(
        self,
        parents=[],
        columns=output,
        value_fn=lambda obj, field: getattr(obj, field.name),
    )
    return output

  @classmethod
  def schema(cls) -> pa.Schema:
    """Returns Arrow schema for the component.

    The schema has a flat structure - a list of pa.Field instances, where names
    correspond to column names (returned by `to_flatten_dict`)

    For MyComponent from the example in the class' docstring `schema()` will
    return the following Apache Arrow schema: pa.Schema([
       ('key.context_name', pa.string()),
       ('key.object_id', pa.string()),
       ('[MyComponent].bar[*].foo.name', pa.list_(pa.string())),
       ('[MyComponent].bar[*].foo.value', pa.list_(pa.int32())),
       ('[MyComponent].baz.value', pa.float32())
    ])

    For more examples see refer to component_test.py.

    Returns:
      An Apache Arrow schema object corresponding to the values returned by the
      `to_flatten_dict` method.
    """
    arrow_types = collections.OrderedDict()
    flatten_arrow_types(cls, parents=[], columns=arrow_types)
    return pa.schema(arrow_types.items())


class Key(metaclass=abc.ABCMeta):
  """Base class for components key.

  Field names and class names of child classes have to be descriptive and
  understandable without knowing which key class they belong to. Which leads to
  the convention:
    - class XyzKey(ParentKey):
        xyz_foo: str
  """


def create_column(
    arrow_type: Optional[pa.DataType] = None,
    is_repeated: Optional[bool] = False,
    **kwargs,
):
  """Creates a field/column for a Component or a dataclass used as a column.

  Refer to the docstring of the Component class for usage example.

  Args:
    arrow_type: arrow type, required for leaf fields, has to be None for others.
    is_repeated: if True the field will be marked as repeated and all its
      columns will have "[*]" suffix, e.g. "[MyComponent].bar[*].foo.name".
    **kwargs: extra arguments passed through to the `dataclasses.field` method.

  Returns:
    a dataclass field object.
  """
  return dataclasses.field(
      metadata={_META_ARROW_TYPE: arrow_type, _META_IS_REPEATED: is_repeated},
      **kwargs,
  )


def _column_is_repeated(field):
  return field.metadata.get(_META_IS_REPEATED, False)


def _column_arrow_type(field: dataclasses.Field[Any]) -> pa.DataType:
  return field.metadata.get(_META_ARROW_TYPE, None)


def _issubclass(instance_or_class: Union[Any, type], type_: type) -> bool:  # pylint: disable=g-bare-generic
  """An analog to issubclass method which works for instances as well."""
  # NOTE: `type[int]` is not a class, but an instance of the class
  # `types.GenericAlias` which is also an instance of `type`.
  if isinstance(instance_or_class, type) and not isinstance(
      instance_or_class, types.GenericAlias
  ):
    return issubclass(instance_or_class, type_)
  else:
    return isinstance(instance_or_class, type_)


def _get_type(type_: type) -> type:  # pylint: disable=g-bare-generic
  """Returns underlying type for Optional fields."""
  if typing.get_origin(type_) == typing.Union:
    return typing.get_args(type_)[0]
  else:
    return type_


def _get_value_or_type(
    instance_or_class: Union[Any, type], field: dataclasses.Field  # pylint: disable=g-bare-generic
) -> Any:
  if isinstance(instance_or_class, type):
    return _get_type(field.type)
  else:
    return getattr(instance_or_class, field.name)


def _component_column_prefix(instance_or_class: Union[Any, type]) -> str:  # pylint: disable=g-bare-generic
  """Returns component's name in square braces."""
  if isinstance(instance_or_class, type):
    return f'[{instance_or_class.__name__}]'
  else:
    return f'[{instance_or_class.__class__.__name__}]'


def _column_name(obj: Any, field: dataclasses.Field[Any]) -> str:
  """Returns name of the column.

  Examples of column names:
    "key.context_name"
    "[XyzComponent].foo.baz.flag"
    "[XyzComponent].foo.bar[*].location.x"

  Args:
    obj: a dataclass object.
    field: dataclass field

  Returns:
    a string column name.
  """
  if _column_is_repeated(field):
    name = f'{field.name}[*]'
  else:
    name = field.name
  value_or_type = _get_value_or_type(obj, field)
  if _issubclass(obj, Component) and not _issubclass(value_or_type, Key):
    return f'{_component_column_prefix(obj)}.{name}'
  else:
    return name


def _return_none(obj, field):
  del obj, field
  return None


# A function with the following signature:
#   value = fn(obj, field)
ValueFn = Callable[[Any, dataclasses.Field[Any]], Any]


def flatten_dict(
    dataclass: Component,
    parents: list[str],
    columns: dict[str, Any],
    value_fn: ValueFn,
) -> None:
  """Converts a dataclass into a flat dict recursively.

  Args:
    dataclass: any dataclass instance
    parents: list of column names, upper in the tree.
    columns: Output dictionary, where column values (output from the `value_fn`
      to be stored. For keys format refer to `Component.to_flatten_dict` method.
    value_fn: A function, which returns a value from the `obj` using its field
  """
  assert dataclasses.is_dataclass(dataclass), dataclass
  for field in dataclasses.fields(dataclass):
    field_path = parents + [_column_name(dataclass, field)]
    value_or_type = _get_value_or_type(dataclass, field)
    if dataclasses.is_dataclass(_get_type(field.type)):
      if value_or_type is None and not isinstance(dataclass, type):
        flatten_dict(_get_type(field.type), field_path, columns, _return_none)
      else:
        flatten_dict(value_or_type, field_path, columns, value_fn)
    else:
      columns['.'.join(field_path)] = value_fn(dataclass, field)


def flatten_arrow_types(
    dataclass: type[Component],
    parents: list[str],
    columns: dict[str, pa.DataType],
) -> None:
  """Returns a flat dictionary with arrow types for all fields.

  Args:
   dataclass: any dataclass instance
   parents: list of column names, upper in the tree.
   columns: Output dictionary, where values are arrow DataTypes annotated using
     `create_column` method. For keys format refer to
     `Component.to_flatten_dict` method.
  """
  assert dataclasses.is_dataclass(dataclass), dataclass
  for field in dataclasses.fields(dataclass):
    field_path = parents + [_column_name(dataclass, field)]
    type_ = _get_type(field.type)
    if dataclasses.is_dataclass(type_):
      flatten_arrow_types(type_, field_path, columns)
    else:
      columns['.'.join(field_path)] = _column_arrow_type(field)


def _remove_prefix(prefix: str, columns: dict[str, _T]) -> dict[str, _T]:
  """Removes specified prefix from all keys in the input dictionary."""
  return {k.removeprefix(prefix): v for k, v in columns.items()}


def unflatten_dict(class_type: type[_T], columns: dict[str, Any]) -> _T:
  """Creates an instance of the component and populates it with column values.

  Args:
    class_type: a class of the component to create.
    columns: a flat dictionary with key in the format described in
      `Component.to_flatten_dict`.

  Returns:
    An instance of the given class type.
  """
  assert dataclasses.is_dataclass(class_type), class_type
  field_values = {}
  for field in dataclasses.fields(class_type):
    column_name = _column_name(class_type, field)
    if column_name in columns:
      column_value = columns[column_name]
    else:
      type_ = _get_type(field.type)
      assert dataclasses.is_dataclass(type_), type_
      column_value = unflatten_dict(
          type_, _remove_prefix(f'{column_name}.', columns)
      )
    if _issubclass(class_type, Component):
      column_name = column_name.removeprefix(
          f'{_component_column_prefix(class_type)}.'
      )
    if _column_is_repeated(field):
      column_name = column_name.removesuffix('[*]')
    field_values[column_name] = column_value
  return class_type(**field_values)  # pytype: disable=bad-return-type  # re-none
