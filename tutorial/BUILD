# Implicitly load Jupyter build rules.

package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "framedata",
    srcs = [
        "3d_point_cloud.png",
        "frames",
    ],
)

py_binary(
    name = "jupyter_kernel",
    srcs = ["jupyter_kernel.py"],
    data = [
        "tutorial_local.ipynb",
        ":framedata",
        "//waymo_open_dataset/metrics/tools:compute_detection_metrics_main",
        "//waymo_open_dataset/metrics/tools:fakedata",
    ],
    python_version = "PY3",
    deps = [
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
        "//waymo_open_dataset/metrics/python:config_util_py",
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",
        "//waymo_open_dataset/protos:map_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:motion_metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:motion_submission_proto_py_pb2",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
        "//waymo_open_dataset/protos:submission_proto_py_pb2",
        "//waymo_open_dataset/utils:box_utils",
        "//waymo_open_dataset/utils:frame_utils",
        "//waymo_open_dataset/utils:range_image_utils",
        "//waymo_open_dataset/utils:transform_utils",
    ],
)
