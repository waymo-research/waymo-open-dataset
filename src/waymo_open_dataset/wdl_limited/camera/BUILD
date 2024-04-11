load("@wod_deps//:requirements.bzl", "requirement")
package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

# Libs

cc_library(
    name = "camera_model",
    srcs = ["camera_model.cc"],
    hdrs = ["camera_model.h"],
    deps = [
        "//waymo_open_dataset:dataset_cc_proto",
        "//waymo_open_dataset/math:vec2d",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
        "@com_google_glog//:glog",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

filegroup(
    name = "libcamera_model.so",
    srcs = [":camera_model"],
    output_group = "dynamic_library",
)

# Tests
cc_test(
    name = "camera_model_test",
    srcs = ["camera_model_test.cc"],
    deps = [
        ":camera_model",
        "//waymo_open_dataset:dataset_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tensorflow_cc_deps//:cc_library",  # build_cleaner: keep
    ],
)
