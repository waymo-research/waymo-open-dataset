# buildifier: disable=out-of-order-load
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

load("@wod_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

py_library(
    name = "config_util_py",
    srcs = ["config_util_py.py"],
    deps = [
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

py_library(
    name = "detection_metrics",
    srcs = ["detection_metrics.py"],
    deps = [
        requirement('tensorflow'),
        ":config_util_py",
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
    ],
)

py_test(
    name = "detection_metrics_test",
    srcs = ["detection_metrics_test.py"],
    python_version = "PY3",
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":detection_metrics",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
    ],
)

py_library(
    name = "motion_metrics",
    srcs = ["motion_metrics.py"],
    deps = [
        requirement('tensorflow'),
        ":config_util_py",
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
    ],
)

py_test(
    name = "motion_metrics_test",
    srcs = ["motion_metrics_test.py"],
    python_version = "PY3",
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":motion_metrics",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:motion_metrics_proto_py_pb2",
    ],
)

py_library(
    name = "tracking_metrics",
    srcs = ["tracking_metrics.py"],
    deps = [
        requirement('tensorflow'),
        ":config_util_py",
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
    ],
)

py_test(
    name = "tracking_metrics_test",
    srcs = ["tracking_metrics_test.py"],
    python_version = "PY3",
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":tracking_metrics",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
    ],
)

py_library(
    name = "keypoint_metrics",
    srcs = ["keypoint_metrics.py"],
    deps = [
        requirement('immutabledict'),
        requirement('scipy'),
        requirement('tensorflow'),
        ":matcher",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoint_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
        "//waymo_open_dataset/utils:keypoint_data",
    ],
)

py_test(
    name = "keypoint_metrics_test",
    srcs = ["keypoint_metrics_test.py"],
    data = [
        ":testdata/oks_testdata.json",
    ],
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        requirement('tensorflow'),
        ":keypoint_metrics",
        "//waymo_open_dataset/utils:keypoint_data",
    ],
)

py_library(
    name = "wod_detection_evaluator",
    srcs = ["wod_detection_evaluator.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":config_util_py",
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
    ],
)

py_test(
    name = "wod_detection_evaluator_test",
    srcs = ["wod_detection_evaluator_test.py"],
    python_version = "PY3",
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":wod_detection_evaluator",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
    ],
)

py_library(
    name = "matcher",
    srcs = ["matcher.py"],
    deps = [
        requirement('tensorflow'),
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
    ],
)

py_test(
    name = "matcher_test",
    srcs = ["matcher_test.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":matcher",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
    ],
)
