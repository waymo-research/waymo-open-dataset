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
"""Base classes to define utils for compatibility with v1.

Naming convention:
  XyzFrameExtractor - a class which extracts Xyz components from a Frame proto.
  XyzFrameInjector - a class which injects Xyz components into a Frame proto.
"""

import abc
import dataclasses
from typing import Generic, Iterable, Sequence, TypeVar

import pyarrow as pa

from google.protobuf import message
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.v2 import component
from waymo_open_dataset.v2 import typing_utils
from waymo_open_dataset.v2.perception import base


class ComponentSource:
  """Base class for all sources for extraction.

  This class has no methods and serves for type annotation and documentation
  purposes.

  A child class is supposed to have all fields required to create corresponding
  component.

  For example, `Frame` has a repeated field `camera_labels`, which has repeated
  field `labels` of the type `Label` with actual camera labels. A method to
  extract a single camera label would need to access all levels of this nested
  heierarchy of fields (see `CameraLabelComponentSrc` below):
    - frame - to extract context information
    - frame.camera_labels[i] - to get a camera name
    - frame.camera_labels[i].labels[j] - to get actual label.
  A `ComponentSource` object should have references to all levels. This will
  allow an extractor pipeline to iterate over `i` and `j`, create a single
  instance of the source class with `frame`, `camera_label` and `label` fields
  and call all extractors for the corresponding type of components.

  If you are familiar with SQL syntax, a `ComponentSource` represents a row in a
  query like this:
    SELECT
      T.frame, camera_label, label
    FROM v1_dataset AS T
    CROSS JOIN UNNEST(T.frame.camera_labels) AS camera_label
    CROSS JOIN UNEST(camera_label.labels) AS label;

  See below `ComponentSource` child classes which define collections of fields
  for the corresponding extractors to use to create their components.

  This design allows to standardize interfaces for all components and implement
  a general purpose extraction pipeline parametrized by extractor types.
  """


class ExtractionError(Exception):
  """Base class for all extraction specific errors."""


def assert_has_fields(proto: message.Message, fields: Sequence[str]) -> None:
  """Raises an extraction error if any of the fields is missing in the proto."""
  for field in fields:
    if not proto.HasField(field):
      raise ExtractionError(f'{proto} has missing "{field}" field.')


_S = TypeVar('_S', bound=ComponentSource)
_C = TypeVar('_C', bound=component.Component)


class ComponentExtractor(Generic[_S, _C], metaclass=abc.ABCMeta):
  """Base class for all extractors.

  Generic parameters:
    _S: A subset of Frame proto fields (v1 format) to create a component (v2
      format).
    _C: Output component created by the extractor.
  """

  @abc.abstractmethod
  def __call__(self, src: _S) -> Iterable[_C]:
    """Extracts all components from the frame.

    Args:
      src: an instance of a `ComponentSource` for the component.

    Returns:
      an instance of the component class or None if data for the component is
      completely missing.

    Raises:
      ExtractionError: if there is data for the component, but some of required
      fields are missing or data is inconsistent and the component can't be
      extracted.
    """

  @classmethod
  def component(cls) -> type[_C]:
    return typing_utils.iterable_item_type(cls.__call__)

  @classmethod
  def schema(cls) -> pa.Schema:
    """Returns schema for the extracted components."""
    return cls.component().schema()


@dataclasses.dataclass(frozen=True)
class FrameComponentSrc(ComponentSource):
  """Subfields of the Frame proto relevant to FrameComponents."""

  frame: dataset_pb2.Frame


class SegmentComponentExtractor:
  """Interface for extracting SegmentComponent from FrameComponentSrc.

  This class has no methods and serves for type annotation and documentation
  purposes.
  """


class FrameComponentExtractor(
    ComponentExtractor[FrameComponentSrc, base.FrameComponent]
):
  """Interface for extracting FrameComponent from FrameComponentSrc."""

  @abc.abstractmethod
  def __call__(self, src: FrameComponentSrc) -> Iterable[base.FrameComponent]:
    """See base class."""


@dataclasses.dataclass(frozen=True)
class CameraLabelComponentSrc(ComponentSource):
  """Subfields of the Frame proto relevant to all CameraLabelComponents.

  Refer to `ComponentSource` for usage details.
  """

  frame: dataset_pb2.Frame
  camera_labels: dataset_pb2.CameraLabels
  label: label_pb2.Label


class CameraLabelComponentExtractor(
    ComponentExtractor[CameraLabelComponentSrc, base.CameraLabelComponent]
):
  """Interface for extracting CameraLabelComponent from CameraLabelComponentSrc."""

  @abc.abstractmethod
  def __call__(
      self, src: CameraLabelComponentSrc
  ) -> Iterable[base.CameraLabelComponent]:
    """See base class."""


@dataclasses.dataclass(frozen=True)
class LiDARComponentSrc(ComponentSource):
  """Subfields of the Frame proto relevant to all LiDARComponents.

  Refer to `ComponentSource` for usage details.
  """
  frame: dataset_pb2.Frame
  laser: dataset_pb2.Laser


@dataclasses.dataclass(frozen=True)
class LiDARComponentExtractor(
    ComponentExtractor[LiDARComponentSrc, base.LaserComponent]
):
  """Interface for extracting LiDARComponent from LiDARComponentSrc."""

  @abc.abstractmethod
  def __call__(self, src: LiDARComponentSrc) -> Iterable[base.LaserComponent]:
    """See base class."""


@dataclasses.dataclass(frozen=True)
class CameraImageComponentSrc(ComponentSource):
  """Subfields of the Frame proto relevant to CameraImageComponent.

  Refer to `ComponentSource` for usage details.
  """
  frame: dataset_pb2.Frame
  camera_image: dataset_pb2.CameraImage


@dataclasses.dataclass(frozen=True)
class CameraImageComponentExtractor(
    ComponentExtractor[CameraImageComponentSrc, base.CameraComponent]
):
  """Interface for extracting CameraImageComponent from CameraImageComponentSrc."""

  @abc.abstractmethod
  def __call__(
      self, src: CameraImageComponentSrc
  ) -> Iterable[base.CameraComponent]:
    """See base class."""


@dataclasses.dataclass(frozen=True)
class LiDARLabelComponentSrc(ComponentSource):
  frame: dataset_pb2.Frame
  lidar_label: label_pb2.Label


@dataclasses.dataclass(frozen=True)
class LiDARLabelComponentExtractor(
    ComponentExtractor[LiDARLabelComponentSrc, base.LaserLabelComponent]
):

  @abc.abstractmethod
  def __call__(
      self, src: LiDARLabelComponentSrc
  ) -> Iterable[base.LaserLabelComponent]:
    """See base class."""


@dataclasses.dataclass(frozen=True)
class CameraCalibrationComponentSrc(FrameComponentSrc):
  camera_calibration: dataset_pb2.CameraCalibration


@dataclasses.dataclass(frozen=True)
class CameraCalibrationComponentExtractor(
    ComponentExtractor[
        CameraCalibrationComponentSrc,
        base.SegmentCameraComponent,
    ],
):

  @abc.abstractmethod
  def __call__(
      self, src: CameraCalibrationComponentSrc
  ) -> Iterable[base.SegmentCameraComponent]:
    """See base class."""


@dataclasses.dataclass(frozen=True)
class LiDARCalibrationComponentSrc(FrameComponentSrc):
  lidar_calibration: dataset_pb2.LaserCalibration


@dataclasses.dataclass(frozen=True)
class LiDARCalibrationComponentExtractor(
    ComponentExtractor[
        LiDARCalibrationComponentSrc,
        base.SegmentLaserComponent,
    ],
):

  @abc.abstractmethod
  def __call__(
      self, src: LiDARCalibrationComponentSrc
  ) -> Iterable[base.SegmentLaserComponent]:
    """See base class."""
