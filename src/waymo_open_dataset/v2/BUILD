load("@wod_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "v2",
    srcs = ["__init__.py"],
    deps = [
        ":column_types",
        ":component",
        ":dataframe_utils",
        "//waymo_open_dataset/v2/perception:box",
        "//waymo_open_dataset/v2/perception:camera_image",
        "//waymo_open_dataset/v2/perception:context",
        "//waymo_open_dataset/v2/perception:keypoints",
        "//waymo_open_dataset/v2/perception:lidar",
        "//waymo_open_dataset/v2/perception:object_asset",
        "//waymo_open_dataset/v2/perception:pose",
        "//waymo_open_dataset/v2/perception:segmentation",
        "//waymo_open_dataset/v2/perception/utils:lidar_utils",
        "//waymo_open_dataset/v2/perception/utils:object_asset_codec",
        "//waymo_open_dataset/v2/perception/utils:object_asset_utils",
    ],
)

py_library(
    name = "column_types",
    srcs = ["column_types.py"],
    deps = [
        requirement('numpy'),
        requirement('pyarrow'),
        ":component",
    ],
)

py_test(
    name = "column_types_test",
    srcs = ["column_types_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        ":column_types",
    ],
)

py_library(
    name = "component",
    srcs = ["component.py"],
    deps = [requirement('pyarrow')],
)

py_test(
    name = "component_test",
    srcs = ["component_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('pyarrow'),
        ":component",
    ],
)

py_library(
    name = "typing_utils",
    srcs = ["typing_utils.py"],
)

py_test(
    name = "typing_utils_test",
    srcs = ["typing_utils_test.py"],
    deps = [
        requirement('absl-py'),
        ":typing_utils",
    ],
)

py_library(
    name = "dataframe_utils",
    srcs = ["dataframe_utils.py"],
    deps = [
        requirement('dask'),
        requirement('pandas'),
    ],
)

py_test(
    name = "dataframe_utils_test",
    srcs = ["dataframe_utils_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('dask'),
        requirement('pandas'),
        ":dataframe_utils",
    ],
)
