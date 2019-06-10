package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "config_util_py",
    srcs = ["config_util_py.py"],
    deps = [
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",
    ],
)

py_library(
    name = "detection_metrics",
    srcs = ["detection_metrics.py"],
    deps = [
        ":config_util_py",
        # Implicit tensorflow dependency.
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
    ],
)

py_test(
    name = "detection_metrics_test",
    srcs = ["detection_metrics_test.py"],
    python_version = "PY3",
    deps = [
        ":detection_metrics",
        # Implicit python proto dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
    ],
)

