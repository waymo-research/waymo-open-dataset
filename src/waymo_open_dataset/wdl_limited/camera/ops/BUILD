load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

load("@wod_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_binary(
    name = "camera_model_ops.so",
    srcs = [
        "camera_model_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "//waymo_open_dataset/wdl_limited/camera:camera_model",
        "//waymo_open_dataset:dataset_cc_proto",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

py_library(
    name = "py_camera_model_ops",
    srcs = ["py_camera_model_ops.py"],
    deps = [
        requirement('tensorflow')
    ],
    data = [
        ":camera_model_ops.so"
    ],
)

py_test(
    name = "camera_model_ops_test",
    srcs = ["camera_model_ops_test.py"],
    python_version = "PY3",
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":py_camera_model_ops",
        "//waymo_open_dataset:dataset_proto_py_pb2",
    ],
    data = [
        ":camera_model_ops.so"
    ],
)
