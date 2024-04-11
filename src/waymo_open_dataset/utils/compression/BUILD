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
        requirement('numpy'),
        "//waymo_open_dataset/protos:compressed_lidar_proto_py_pb2",
    ],
)

py_test(
    name = "delta_encoder_test",
    srcs = ["delta_encoder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        ":delta_encoder",
    ],
)

py_library(
    name = "tiling",
    srcs = ["tiling.py"],
    srcs_version = "PY3",
    deps = [
        requirement('tensorflow'),
    ],
)

py_test(
    name = "tiling_test",
    srcs = ["tiling_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement('absl-py'),
        requirement('tensorflow'),
        ":tiling",
    ],
)

py_library(
    name = "pca_codec",
    srcs = ["pca_codec.py"],
    srcs_version = "PY3",
    deps = [
        requirement('dacite'),
        requirement('numpy'),
        requirement('scikit-learn'),
        requirement('tensorflow'),
        ":tiling",
    ],
)

py_test(
    name = "pca_codec_test",
    srcs = ["pca_codec_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        requirement('tensorflow'),
        ":pca_codec",
    ],
)
