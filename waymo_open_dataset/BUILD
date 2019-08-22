# Waymo open dataset

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE"])

load("//tf:build_config.bzl", "all_proto_library")


all_proto_library(
    src = "dataset.proto",
    deps = [":label_proto"],
)


all_proto_library(
    src = "label.proto",
)

