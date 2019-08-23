package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

# Libs

cc_library(
    name = "aabox2d",
    srcs = [
        "aabox2d.h",
    ],
    hdrs = [
        "aabox2d.h",
    ],
)

cc_library(
    name = "box2d",
    srcs = [
        "box2d.cc",
    ],
    hdrs = [
        "box2d.h",
    ],
    deps = [
        ":segment2d",
        ":vec2d",
    ],
)

cc_library(
    name = "exactfloat",
    srcs = ["exactfloat.cc"],
    hdrs = ["exactfloat.h"],
    deps = [
        "//waymo_open_dataset/common:integral_types",
        "@boringssl//:crypto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "math_util",
    srcs = [
        "math_util.h",
    ],
    hdrs = [
        "math_util.h",
    ],
)

cc_library(
    name = "polygon2d",
    srcs = [
        "polygon2d.cc",
    ],
    hdrs = [
        "polygon2d.h",
    ],
    deps = [
        ":box2d",
        ":exactfloat",
        ":segment2d",
        ":vec2d",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "segment2d",
    srcs = [
        "segment2d.cc",
    ],
    hdrs = [
        "segment2d.h",
    ],
    deps = [
        ":math_util",
        ":vec2d",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "vec2d",
    srcs = [
        "vec2d.h",
    ],
    hdrs = [
        "vec2d.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "aabox2d_test",
    srcs = [
        "aabox2d_test.cc",
    ],
    deps = [
        ":aabox2d",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "box2d_test",
    srcs = ["box2d_test.cc"],
    deps = [
        ":box2d",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "exactfloat_test",
    size = "large",
    srcs = ["exactfloat_test.cc"],
    deps = [
        ":exactfloat",
        "//waymo_open_dataset/common:integral_types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "polygon2d_test",
    srcs = ["polygon2d_test.cc"],
    deps = [
        ":polygon2d",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "vec2d_test",
    srcs = ["vec2d_test.cc"],
    deps = [
        ":vec2d",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
