package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "jupyter_kernel",
    srcs = ["jupyter_kernel.py"],
    python_version = "PY3",
    deps = [
        "//waymo_open_dataset:dataset_proto_py_pb2",  # build_cleaner: keep
        "//waymo_open_dataset:label_proto_py_pb2",  # build_cleaner: keep
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",  # build_cleaner: keep
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",  # build_cleaner: keep
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",  # build_cleaner: keep
        "//waymo_open_dataset/utils:box_utils",  # build_cleaner: keep
        "//waymo_open_dataset/utils:range_image_utils",  # build_cleaner: keep
        "//waymo_open_dataset/utils:transform_utils",  # build_cleaner: keep
    ],
)
