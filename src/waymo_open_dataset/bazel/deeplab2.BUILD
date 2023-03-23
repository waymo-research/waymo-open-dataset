load("@wod_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

# We need only several pure python files from the deeplab2 library for camera segmentation metric.
py_library(
    name = "deeplab2",
    srcs = [
        "data/ade20k_constants.py",
        "data/dataset.py",
        "data/waymo_constants.py",
        "evaluation/segmentation_and_tracking_quality.py",
    ],
    deps = [
        requirement("immutabledict"),
    ],
)
