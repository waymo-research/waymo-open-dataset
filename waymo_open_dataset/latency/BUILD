package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_binary(
    name = "compare_objects_file_to_submission_main",
    srcs = ["compare_objects_file_to_submission_main.cc"],
    deps = [
        "//waymo_open_dataset:dataset_cc_proto",
        "//waymo_open_dataset/common:integral_types",
        "//waymo_open_dataset/metrics:matcher",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "//waymo_open_dataset/protos:submission_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_glog//:glog",
    ],
)

py_binary(
    name = "make_objects_file_from_latency_results",
    srcs = ["make_objects_file_from_latency_results.py"],
    python_version = "PY3",
    deps = [
        # Implicit numpy dependency.
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
    ],
)
