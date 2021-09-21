"""Install dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def load_tf_version_specific_dependencies():
    """Load TF version specific dependencies."""
    http_archive(
        name = "com_google_absl",
        sha256 = "35f22ef5cb286f09954b7cc4c85b5a3f6221c9d4df6b8c4a1e9d399555b366ee",
        strip_prefix = "abseil-cpp-997aaf3a28308eba1b9156aa35ab7bca9688e9f6",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz",
        ],
    )

    http_archive(
        name = "zlib",
        build_file = "//third_party:zlib.BUILD",
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        urls = [
            "http://mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
    )

    PROTOBUF_URLS = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
        "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
    ]

    PROTOBUF_SHA256 = "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b"

    PROTOBUF_STRIP_PREFIX = "protobuf-3.9.2"

    # We need to import the protobuf library under the names com_google_protobuf
    # and com_google_protobuf_cc to enable proto_library support in bazel.
    http_archive(
        name = "com_google_protobuf",
        sha256 = PROTOBUF_SHA256,
        strip_prefix = PROTOBUF_STRIP_PREFIX,
        urls = PROTOBUF_URLS,
    )

    http_archive(
        name = "com_google_protobuf_cc",
        sha256 = PROTOBUF_SHA256,
        strip_prefix = PROTOBUF_STRIP_PREFIX,
        urls = PROTOBUF_URLS,
    )
