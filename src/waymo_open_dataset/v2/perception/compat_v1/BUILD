load("@wod_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "test_data",
    srcs = glob(["testdata/*.tfrecord"]),
)

py_library(
    name = "interfaces",
    srcs = ["interfaces.py"],
    deps = [
        requirement('pyarrow'),
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/v2:component",
        "//waymo_open_dataset/v2:typing_utils",
        "//waymo_open_dataset/v2/perception:base",
    ],
)

py_test(
    name = "interfaces_test",
    srcs = ["interfaces_test.py"],
    deps = [
        requirement('absl-py'),
        ":interfaces",
        "//waymo_open_dataset/v2/perception:base",
    ],
)

py_library(
    name = "keypoints",
    srcs = ["keypoints.py"],
    deps = [
        ":interfaces",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/v2:column_types",
        "//waymo_open_dataset/v2/perception:base",
        "//waymo_open_dataset/v2/perception:keypoints",
    ],
)

py_test(
    name = "keypoints_test",
    srcs = ["keypoints_test.py"],
    deps = [
        requirement('absl-py'),
        ":interfaces",
        ":keypoints",
        "//waymo_open_dataset:dataset_proto_py_pb2",
    ],
)

py_library(
    name = "pose",
    srcs = ["pose.py"],
    deps = [
        ":interfaces",
        "//waymo_open_dataset/v2:column_types",
        "//waymo_open_dataset/v2/perception:base",
        "//waymo_open_dataset/v2/perception:pose",
    ],
)

py_library(
    name = "lidar",
    srcs = ["lidar.py"],
    deps = [
        ":interfaces",
        "//waymo_open_dataset/v2/perception:lidar",
        "//waymo_open_dataset/v2/perception/utils:lidar_utils",
    ],
)

py_library(
    name = "camera_image",
    srcs = ["camera_image.py"],
    deps = [
        ":interfaces",
        "//waymo_open_dataset/v2:column_types",
        "//waymo_open_dataset/v2/perception:base",
        "//waymo_open_dataset/v2/perception:camera_image",
    ],
)

py_library(
    name = "box",
    srcs = ["box.py"],
    deps = [
        ":interfaces",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/v2:column_types",
        "//waymo_open_dataset/v2/perception:base",
        "//waymo_open_dataset/v2/perception:box",
        "//waymo_open_dataset/v2/perception:camera_image",
    ],
)

py_library(
    name = "context",
    srcs = ["context.py"],
    deps = [
        ":interfaces",
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/v2:column_types",
        "//waymo_open_dataset/v2/perception:base",
        "//waymo_open_dataset/v2/perception:context",
    ],
)

py_library(
    name = "segmentation",
    srcs = ["segmentation.py"],
    deps = [
        ":interfaces",
        "//waymo_open_dataset/v2/perception:base",
        "//waymo_open_dataset/v2/perception:segmentation",
        "//waymo_open_dataset/v2/perception/utils:lidar_utils",
    ],
)

py_test(
    name = "pose_test",
    srcs = ["pose_test.py"],
    deps = [
        requirement('absl-py'),
        ":interfaces",
        ":pose",
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/v2/perception:pose",
    ],
)

py_test(
    name = "lidar_test",
    srcs = ["lidar_test.py"],
    data = [
        ":test_data",
    ],
    deps = [
        requirement('absl-py'),
        requirement('tensorflow'),
        ":interfaces",
        ":lidar",
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/v2/perception:lidar",
        "//waymo_open_dataset/v2/perception/utils:lidar_utils",
    ],
)

py_test(
    name = "camera_image_test",
    srcs = ["camera_image_test.py"],
    data = [
        ":test_data",
    ],
    deps = [
        requirement('absl-py'),
        requirement('tensorflow'),
        ":camera_image",
        ":interfaces",
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/v2/perception:camera_image",
    ],
)

py_test(
    name = "box_test",
    srcs = ["box_test.py"],
    data = [
        ":test_data",
    ],
    deps = [
        requirement('absl-py'),
        requirement('tensorflow'),
        ":box",
        ":interfaces",
        "//waymo_open_dataset:dataset_proto_py_pb2",
    ],
)

py_test(
    name = "context_test",
    srcs = ["context_test.py"],
    data = [
        ":test_data",
    ],
    deps = [
        requirement('absl-py'),
        requirement('tensorflow'),
        ":context",
        ":interfaces",
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/v2/perception:camera_image",
        "//waymo_open_dataset/v2/perception:lidar",
    ],
)

py_test(
    name = "segmentation_test",
    srcs = ["segmentation_test.py"],
    data = [
        ":test_data",
    ],
    deps = [
        requirement('absl-py'),
        requirement('tensorflow'),
        ":interfaces",
        ":segmentation",
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/v2/perception:camera_image",
        "//waymo_open_dataset/v2/perception:lidar",
        "//waymo_open_dataset/v2/perception/utils:lidar_utils",
    ],
)
