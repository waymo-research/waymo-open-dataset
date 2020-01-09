package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["BUILD"])

# Py libs.

py_library(
    name = "box_utils",
    srcs = ["box_utils.py"],
    deps = [
        ":transform_utils",
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "range_image_utils",
    srcs = ["range_image_utils.py"],
    deps = [
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "transform_utils",
    srcs = ["transform_utils.py"],
    deps = [
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    deps = [
        ":transform_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "frame_utils",
    srcs = ["frame_utils.py"],
    deps = [
        ":range_image_utils",
        ":transform_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:dataset_proto_py_pb2",
    ],
)

# Tests
py_test(
    name = "box_utils_test",
    srcs = ["box_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":box_utils",
        ":test_utils",
        ":transform_utils",
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "range_image_utils_test",
    srcs = ["range_image_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":range_image_utils",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "transform_utils_test",
    srcs = ["transform_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":transform_utils",
        # Implicit tensorflow dependency.
    ],
)
