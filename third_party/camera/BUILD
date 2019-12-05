package(
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
        "//waymo_open_dataset/common:integral_types",
        "//waymo_open_dataset/math:vec2d",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
        "@com_google_glog//:glog",
        "@eigen_archive//:eigen",
    ],
)

# Tests
cc_test(
    name = "camera_model_test",
    srcs = ["camera_model_test.cc"],
    deps = [
        ":camera_model",
        # Implicit proto dependency.
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "//waymo_open_dataset:dataset_cc_proto",
    ],
)
