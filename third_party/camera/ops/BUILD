package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

py_library(
    name = "py_camera_model_ops",
    srcs = ["py_camera_model_ops.py"],
    data = [
        ":camera_model_ops.so",
    ],
)

cc_binary(
    name = "camera_model_ops.so",
    srcs = [
        "camera_model_ops.cc",
    ],
    copts = [
        "-pthread",
    ],
    linkshared = 1,
    deps = [
        "//third_party/camera:camera_model",
        "//waymo_open_dataset:dataset_cc_proto",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

# Tests
py_test(
    name = "camera_model_ops_test",
    srcs = ["camera_model_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":py_camera_model_ops",
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:dataset_proto_py_pb2",
    ],
)
