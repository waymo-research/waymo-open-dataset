package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE"])

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")

cc_proto_library(
    name = "metrics_cc_proto",
    srcs = ["metrics.proto"],
    deps = [
        ":breakdown_cc_proto",
        "//waymo_open_dataset:label_cc_proto",
    ],
)

py_proto_library(
    name = "metrics_proto_py_pb2",
    srcs = ["metrics.proto"],
    deps = [
        ":breakdown_proto_py_pb2",
        "//waymo_open_dataset:label_proto_py_pb2",
    ],
)

cc_proto_library(
    name = "breakdown_cc_proto",
    srcs = ["breakdown.proto"],
    deps = [
        "//waymo_open_dataset:label_cc_proto",
    ],
)

py_proto_library(
    name = "breakdown_proto_py_pb2",
    srcs = ["breakdown.proto"],
    deps = [
        "//waymo_open_dataset:label_proto_py_pb2",
    ],
)
