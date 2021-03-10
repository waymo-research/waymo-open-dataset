package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

# Libraries.
cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/common:integral_types",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "//waymo_open_dataset/protos:motion_metrics_cc_proto",
        "//waymo_open_dataset/protos:motion_submission_cc_proto",
        "//waymo_open_dataset/protos:scenario_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:optional",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "py_metrics_ops",
    srcs = ["py_metrics_ops.py"],
    data = [
        ":metrics_ops.so",
    ],
)

# All ops defined in metrics_ops.cc must be compiled in one target.
cc_binary(
    name = "metrics_ops.so",
    srcs = [
        "detection_metrics_ops.cc",
        "metrics_ops.cc",
        "motion_metrics_ops.cc",
        "tracking_metrics_ops.cc",
    ],
    copts = [
        "-pthread",
    ],
    linkshared = 1,
    deps = [
        ":utils",
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/metrics:config_util",
        "//waymo_open_dataset/metrics:detection_metrics",
        "//waymo_open_dataset/metrics:motion_metrics",
        "//waymo_open_dataset/metrics:tracking_metrics",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

# Tests
py_test(
    name = "detection_metrics_ops_test",
    srcs = ["detection_metrics_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":py_metrics_ops",
        # Implicit python proto dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
    ],
)

py_test(
    name = "motion_metrics_ops_test",
    srcs = ["motion_metrics_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":py_metrics_ops",
        # Implicit python proto dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:motion_metrics_proto_py_pb2",
    ],
)

py_test(
    name = "tracking_metrics_ops_test",
    srcs = ["tracking_metrics_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":py_metrics_ops",
        # Implicit python proto dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
    ],
)
