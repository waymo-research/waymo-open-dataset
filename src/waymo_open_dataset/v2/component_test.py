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
"""Tests for waymo_open_dataset.v2.component."""

import dataclasses
from typing import Optional

from absl.testing import absltest
import pyarrow as pa

from waymo_open_dataset.v2 import component as _lib


@dataclasses.dataclass(frozen=True)
class RecordWithOptionalField:
  field: Optional[int] = _lib.create_column(arrow_type=pa.int8(), default=None)


@dataclasses.dataclass(frozen=True)
class NestedRecordWithOptionalField:
  foo: Optional[RecordWithOptionalField] = None


@dataclasses.dataclass(frozen=True)
class TypedRecord:
  foo: Optional[int] = _lib.create_column(arrow_type=pa.int8(), default=None)


@dataclasses.dataclass(frozen=True)
class RecordWithRepeatedField:
  bar: Optional[TypedRecord] = _lib.create_column(
      is_repeated=True, default=None
  )
  baz: Optional[int] = _lib.create_column(arrow_type=pa.int8(), default=None)


@dataclasses.dataclass(frozen=True)
class MyKey(_lib.Key):
  name: str = _lib.create_column(arrow_type=pa.string())


@dataclasses.dataclass(frozen=True)
class MyComponent(_lib.Component):
  key: MyKey
  field: str = _lib.create_column(arrow_type=pa.string())


@dataclasses.dataclass(frozen=True)
class ComponentWithPrimitiveField(_lib.Component):
  key: MyKey
  values: list[int] = _lib.create_column(arrow_type=pa.list_(pa.int32()))


def _field_value(obj, field):
  return getattr(obj, field.name)


class FlattenDictTest(absltest.TestCase):

  def test_returns_value_for_pain_type_field(self):
    obj = RecordWithOptionalField(field=1)

    output = {}
    _lib.flatten_dict(obj, [], output, _field_value)

    self.assertEqual({'field': 1}, output)

  def test_passes_through_none_for_plain_type_fields(self):
    obj = RecordWithOptionalField()

    output = {}
    _lib.flatten_dict(obj, [], output, _field_value)

    self.assertEqual({'field': None}, output)

  def test_returns_value_for_nested_type_field(self):
    obj = NestedRecordWithOptionalField(foo=RecordWithOptionalField(field=1))

    output = {}
    _lib.flatten_dict(obj, [], output, _field_value)

    self.assertEqual({'foo.field': 1}, output)

  def test_passes_through_none_for_nested_type_fields_not_none(self):
    obj = NestedRecordWithOptionalField(foo=RecordWithOptionalField())

    output = {}
    _lib.flatten_dict(obj, [], output, _field_value)

    self.assertEqual({'foo.field': None}, output)

  def test_passes_through_none_for_nested_type_fields_none(self):
    obj = NestedRecordWithOptionalField()

    output = {}
    _lib.flatten_dict(obj, [], output, _field_value)

    self.assertEqual({'foo.field': None}, output)

  def test_added_asterix_for_repeated_fields(self):
    obj = RecordWithRepeatedField(bar=TypedRecord(foo=1))

    output = {}
    _lib.flatten_dict(obj, [], output, _field_value)

    self.assertEqual({'bar[*].foo': 1, 'baz': None}, output)

  def test_added_prefix_for_all_but_key_fields_of_componets(self):
    obj = MyComponent(key=MyKey(name='key_name'), field='field_value')

    output = {}
    _lib.flatten_dict(obj, [], output, _field_value)

    self.assertEqual(
        {'[MyComponent].field': 'field_value', 'key.name': 'key_name'}, output
    )


class FlattenArrowTypesTest(absltest.TestCase):

  def test_returns_type_for_nested_class_with_optional_fields(self):
    dataclass = NestedRecordWithOptionalField

    output = {}
    _lib.flatten_arrow_types(dataclass, [], output)

    self.assertEqual({'foo.field': pa.int8()}, output)

  def test_added_prefix_for_all_but_key_fields_of_componets(self):
    dataclass = MyComponent

    output = {}
    _lib.flatten_arrow_types(dataclass, [], output)

    self.assertEqual(
        {'[MyComponent].field': pa.string(), 'key.name': pa.string()}, output
    )

  def test_schema_for_a_component_with_a_primitive_column_is_correct(self):
    schema = ComponentWithPrimitiveField.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    self.assertEqual(
        schema_dict,
        {
            'key.name': pa.string(),
            '[ComponentWithPrimitiveField].values': pa.list_(pa.int32()),
        },
    )


class UnflattenDict(absltest.TestCase):

  def test_populates_fields_of_a_flat_dataclass(self):
    columns = {'field': 42}
    obj = _lib.unflatten_dict(RecordWithOptionalField, columns)
    self.assertEqual(obj, RecordWithOptionalField(field=42))

  def test_populates_fields_of_a_nested_dataclass(self):
    columns = {'foo.field': 42}

    obj = _lib.unflatten_dict(NestedRecordWithOptionalField, columns)

    self.assertEqual(
        obj,
        NestedRecordWithOptionalField(foo=RecordWithOptionalField(field=42)),
    )

  def test_supports_repeated_fields(self):
    columns = {'bar[*].foo': 1, 'baz': None}

    obj = _lib.unflatten_dict(RecordWithRepeatedField, columns)

    self.assertEqual(obj, RecordWithRepeatedField(bar=TypedRecord(foo=1)))

  def test_populates_top_level_component_fields(self):
    columns = {'[MyComponent].field': 'field_value', 'key.name': 'key_name'}

    obj = _lib.unflatten_dict(MyComponent, columns)

    self.assertEqual(
        obj, MyComponent(key=MyKey(name='key_name'), field='field_value')
    )


if __name__ == '__main__':
  absltest.main()
