package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "breakdown_generator",
    srcs = ["breakdown_generator.cc"],
    hdrs = ["breakdown_generator.h"],
    deps = [
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "config_util",
    srcs = ["config_util.cc"],
    hdrs = ["config_util.h"],
    deps = [
        ":breakdown_generator",
        ":metrics_utils",
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mot",
    srcs = ["mot.cc"],
    hdrs = ["mot.h"],
    deps = [
        ":matcher",
        ":metrics_utils",
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "iou",
    srcs = ["iou.cc"],
    hdrs = ["iou.h"],
    deps = [
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/math:aabox2d",
        "//waymo_open_dataset/math:box2d",
        "//waymo_open_dataset/math:polygon2d",
        "//waymo_open_dataset/math:vec2d",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "hungarian",
    srcs = ["hungarian.cc"],
    hdrs = ["hungarian.h"],
    deps = [
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "matcher",
    srcs = ["matcher.cc"],
    hdrs = ["matcher.h"],
    deps = [
        ":hungarian",
        ":iou",
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
    ],
)

cc_library(
    name = "metrics_utils",
    srcs = ["metrics_utils.cc"],
    hdrs = ["metrics_utils.h"],
    deps = [
        ":breakdown_generator",
        ":matcher",
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "detection_metrics",
    srcs = ["detection_metrics.cc"],
    hdrs = ["detection_metrics.h"],
    deps = [
        ":matcher",
        ":metrics_utils",
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "tracking_metrics",
    srcs = ["tracking_metrics.cc"],
    hdrs = ["tracking_metrics.h"],
    deps = [
        ":breakdown_generator",
        ":matcher",
        ":metrics_utils",
        ":mot",
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "iou_test",
    srcs = ["iou_test.cc"],
    deps = [
        ":iou",
        ":test_utils",
        "//waymo_open_dataset:label_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "breakdown_generator_test",
    srcs = ["breakdown_generator_test.cc"],
    deps = [
        ":breakdown_generator",
        ":test_utils",
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "matcher_test",
    srcs = ["matcher_test.cc"],
    deps = [
        ":iou",
        ":matcher",
        ":test_utils",
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "hungarian_test",
    srcs = ["hungarian_test.cc"],
    deps = [
        ":hungarian",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "metrics_utils_test",
    srcs = ["metrics_utils_test.cc"],
    deps = [
        ":matcher",
        ":metrics_utils",
        ":test_utils",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "detection_metrics_test",
    srcs = ["detection_metrics_test.cc"],
    deps = [
        ":detection_metrics",
        ":test_utils",
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "mot_test",
    srcs = ["mot_test.cc"],
    deps = [
        ":mot",
        ":test_utils",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tracking_metrics_test",
    srcs = ["tracking_metrics_test.cc"],
    deps = [
        ":test_utils",
        ":tracking_metrics",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
