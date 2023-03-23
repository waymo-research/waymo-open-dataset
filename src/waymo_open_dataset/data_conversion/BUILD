package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

# Libraries.
#===============================================================================

cc_library(
    name = "scenario_conversion",
    srcs = [
        "scenario_conversion.cc",
    ],
    hdrs = [
        "scenario_conversion.h",
    ],
    deps = [
        "//waymo_open_dataset/math:vec3d",
        "//waymo_open_dataset/protos:conversion_config_cc_proto",
        "//waymo_open_dataset/protos:map_cc_proto",
        "//waymo_open_dataset/protos:scenario_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@com_google_protobuf//:protobuf_lite",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

# Tests.
#===============================================================================

cc_test(
    name = "scenario_conversion_test",
    srcs = ["scenario_conversion_test.cc"],
    data = [
    ],
    deps = [
        ":scenario_conversion",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "//waymo_open_dataset/protos:scenario_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@tensorflow_cc_deps//:cc_library",
    ],
)
