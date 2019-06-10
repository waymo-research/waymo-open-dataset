# Waymo open dataset

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE"])

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")

cc_proto_library(
    name = "dataset_cc_proto",
    srcs = ["dataset.proto"],
    deps = [":label_cc_proto"],
)

py_proto_library(
    name = "dataset_proto_py_pb2",
    srcs = ["dataset.proto"],
    deps = [":label_proto_py_pb2"],
)

cc_proto_library(
    name = "label_cc_proto",
    srcs = ["label.proto"],
)

py_proto_library(
    name = "label_proto_py_pb2",
    srcs = ["label.proto"],
)
