load("@wod_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "lidar_utils",
    srcs = ["lidar_utils.py"],
    deps = [
        requirement('tensorflow'),
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/utils:range_image_utils",
        "//waymo_open_dataset/utils:transform_utils",
        "//waymo_open_dataset/v2/perception:base",
        "//waymo_open_dataset/v2/perception:context",
        "//waymo_open_dataset/v2/perception:lidar",
        "//waymo_open_dataset/v2/perception:pose",
        "//waymo_open_dataset/v2/perception:segmentation",
        "//waymo_open_dataset/v2/perception/compat_v1:interfaces",
    ],
)

py_library(
    name = "object_asset_utils",
    srcs = ["object_asset_utils.py"],
    deps = [
        requirement('numpy'),
    ],
)

py_test(
    name = "lidar_utils_test",
    srcs = ["lidar_utils_test.py"],
    data = [
        "//waymo_open_dataset/v2/perception/compat_v1:test_data",
    ],
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        requirement('tensorflow'),
        ":lidar_utils",
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/v2:column_types",
        "//waymo_open_dataset/v2/perception:base",
        "//waymo_open_dataset/v2/perception:context",
        "//waymo_open_dataset/v2/perception:lidar",
        "//waymo_open_dataset/v2/perception:pose",
        "//waymo_open_dataset/v2/perception:segmentation",
        "//waymo_open_dataset/v2/perception/compat_v1:interfaces",
    ],
)

py_test(
    name = "object_asset_utils_test",
    srcs = ["object_asset_utils_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        ":object_asset_utils",
    ],
)

py_library(
    name = "object_asset_codec",
    srcs = ["object_asset_codec.py"],
    deps = [
        requirement('numpy'),
        requirement('pandas'),
        requirement('tensorflow'),
        "//waymo_open_dataset/utils/compression:pca_codec",
        "//waymo_open_dataset/v2/perception:object_asset",
    ],
)

py_test(
    name = "object_asset_codec_test",
    srcs = ["object_asset_codec_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        requirement('pandas'),
        ":object_asset_codec",
        "//waymo_open_dataset/utils/compression:pca_codec",
        "//waymo_open_dataset/v2/perception:base",
        "//waymo_open_dataset/v2/perception:object_asset",
    ],
)
