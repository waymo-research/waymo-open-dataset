load("@wod_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

py_library(
    name = "delta_encoder",
    srcs = ["delta_encoder.py"],
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        "//waymo_open_dataset/protos:compressed_lidar_proto_py_pb2",
    ],
)

py_test(
    name = "delta_encoder_test",
    srcs = ["delta_encoder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("absl-py"),
        requirement("numpy"),
        ":delta_encoder",
    ],
)
