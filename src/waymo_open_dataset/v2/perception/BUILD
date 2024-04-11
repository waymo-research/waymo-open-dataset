load("@wod_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        requirement('pyarrow'),
        "//waymo_open_dataset/v2:component",
    ],
)

py_library(
    name = "pose",
    srcs = ["pose.py"],
    deps = [
        ":base",
        "//waymo_open_dataset/v2:column_types",
    ],
)

py_test(
    name = "pose_test",
    srcs = ["pose_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('pyarrow'),
        ":base",
        ":pose",
        "//waymo_open_dataset/v2:column_types",
    ],
)

py_library(
    name = "keypoints",
    srcs = ["keypoints.py"],
    deps = [
        requirement('pyarrow'),
        ":base",
        "//waymo_open_dataset/v2:column_types",
        "//waymo_open_dataset/v2:component",
    ],
)

py_library(
    name = "lidar",
    srcs = ["lidar.py"],
    deps = [
        requirement('pyarrow'),
        requirement('tensorflow'),
        ":base",
        "//waymo_open_dataset/v2:component",
    ],
)

py_library(
    name = "camera_image",
    srcs = ["camera_image.py"],
    deps = [
        requirement('pyarrow'),
        ":base",
        "//waymo_open_dataset/v2:column_types",
        "//waymo_open_dataset/v2:component",
    ],
)

py_library(
    name = "box",
    srcs = ["box.py"],
    deps = [
        requirement('pyarrow'),
        ":base",
        "//waymo_open_dataset/v2:column_types",
        "//waymo_open_dataset/v2:component",
    ],
)

py_library(
    name = "context",
    srcs = ["context.py"],
    deps = [
        requirement('pyarrow'),
        ":base",
        "//waymo_open_dataset/v2:column_types",
        "//waymo_open_dataset/v2:component",
    ],
)

py_library(
    name = "segmentation",
    srcs = ["segmentation.py"],
    deps = [
        requirement('pyarrow'),
        requirement('tensorflow'),
        ":base",
        "//waymo_open_dataset/v2:component",
    ],
)

py_library(
    name = "object_asset",
    srcs = ["object_asset.py"],
    deps = [
        requirement('numpy'),
        requirement('pyarrow'),
        requirement('tensorflow'),
        ":base",
        "//waymo_open_dataset/v2:column_types",
        "//waymo_open_dataset/v2:component",
    ],
)

py_test(
    name = "object_asset_test",
    srcs = ["object_asset_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        requirement('pyarrow'),
        requirement('tensorflow'),
        ":base",
        ":camera_image",
        ":object_asset",
        "//waymo_open_dataset/v2:column_types",
    ],
)

py_test(
    name = "keypoints_test",
    srcs = ["keypoints_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('pyarrow'),
        requirement('tensorflow'),  # build_cleaner: keep
        ":base",
        ":keypoints",
        "//waymo_open_dataset/v2:column_types",
    ],
)

py_test(
    name = "lidar_test",
    srcs = ["lidar_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('pyarrow'),
        requirement('tensorflow'),
        ":base",
        ":lidar",
    ],
)

py_test(
    name = "camera_image_test",
    srcs = ["camera_image_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('pyarrow'),
        ":base",
        ":camera_image",
        "//waymo_open_dataset/v2:column_types",
    ],
)

py_test(
    name = "box_test",
    srcs = ["box_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('pyarrow'),
        ":base",
        ":box",
        "//waymo_open_dataset/v2:column_types",
    ],
)

py_test(
    name = "context_test",
    srcs = ["context_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('pyarrow'),
        ":base",
        ":context",
        "//waymo_open_dataset/v2:column_types",
    ],
)

py_test(
    name = "segmentation_test",
    srcs = ["segmentation_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('pyarrow'),
        ":base",
        ":segmentation",
        "//waymo_open_dataset/v2:column_types",
    ],
)
