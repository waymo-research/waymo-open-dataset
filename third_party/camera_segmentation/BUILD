package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

# Libs
py_library(
    name = "camera_segmentation_metrics",
    srcs = ["camera_segmentation_metrics.py"],
    deps = [
        "//waymo_open_dataset/protos:camera_segmentation_metrics_proto_py_pb2",
        "@deeplab2",
    ],
)

# Tests
py_test(
    name = "camera_segmentation_metrics_test",
    srcs = ["camera_segmentation_metrics_test.py"],
    deps = [":camera_segmentation_metrics"],
)
