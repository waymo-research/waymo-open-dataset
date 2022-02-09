# pytype
load("//waymo_open_dataset/utils:pytype.bzl", "pytype_strict_library")

# strict
load("//waymo_open_dataset/utils:pytype.bzl", "py_strict_test")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["BUILD"])

# Py libs.

pytype_strict_library(
    name = "box_utils",
    srcs = ["box_utils.py"],
    deps = [
        ":transform_utils",
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:label_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "range_image_utils",
    srcs = ["range_image_utils.py"],
    deps = [
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "transform_utils",
    srcs = ["transform_utils.py"],
    deps = [
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    deps = [
        ":transform_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "frame_utils",
    srcs = ["frame_utils.py"],
    deps = [
        ":range_image_utils",
        ":transform_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:dataset_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "keypoint_utils",
    srcs = ["keypoint_utils.py"],
    deps = [
        # Implicit immutabledict dependency.
        # Implicit matplotlib dependency.
        # Implicit numpy dependency.
        # Implicit plotly dependency.
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoint_proto_py_pb2",
    ],
)

# Tests
py_strict_test(
    name = "box_utils_test",
    srcs = ["box_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":box_utils",
        ":test_utils",
        ":transform_utils",
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "range_image_utils_test",
    srcs = ["range_image_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":range_image_utils",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "transform_utils_test",
    srcs = ["transform_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":transform_utils",
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "keypoint_utils_test",
    srcs = ["keypoint_utils_test.py"],
    deps = [
        ":keypoint_utils",
        # Implicit matplotlib dependency.
        # Implicit numpy dependency.
        # Implicit plotly dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoint_proto_py_pb2",
    ],
)
