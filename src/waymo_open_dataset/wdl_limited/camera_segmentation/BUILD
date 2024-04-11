load("@wod_deps//:requirements.bzl", "requirement")

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
        requirement('numpy'),
        "//waymo_open_dataset/protos:camera_segmentation_metrics_proto_py_pb2",
        "@deeplab2",
    ],
)

# Tests
py_test(
    name = "camera_segmentation_metrics_test",
    srcs = ["camera_segmentation_metrics_test.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":camera_segmentation_metrics",
    ],
)
