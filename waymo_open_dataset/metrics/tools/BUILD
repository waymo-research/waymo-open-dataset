package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "fakedata",
    srcs = [
        "fake_ground_truths.bin",
        "fake_predictions.bin",
    ],
)

cc_binary(
    name = "compute_detection_metrics_main",
    srcs = ["compute_detection_metrics_main.cc"],
    deps = [
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/common:integral_types",
        "//waymo_open_dataset/metrics:config_util",
        "//waymo_open_dataset/metrics:detection_metrics",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "compute_tracking_metrics_main",
    srcs = ["compute_tracking_metrics_main.cc"],
    deps = [
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/common:integral_types",
        "//waymo_open_dataset/metrics:config_util",
        "//waymo_open_dataset/metrics:tracking_metrics",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
    ],
)

cc_binary(
    name = "create_submission",
    srcs = ["create_submission.cc"],
    deps = [
        "@com_google_glog//:glog",
        # Implicit proto dependency.
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//waymo_open_dataset:dataset_cc_proto",
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/common:integral_types",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "//waymo_open_dataset/protos:submission_cc_proto",
    ],
)

py_binary(
    name = "create_prediction_file_example",
    srcs = ["create_prediction_file_example.py"],
    python_version = "PY3",
    deps = [
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
    ],
)
