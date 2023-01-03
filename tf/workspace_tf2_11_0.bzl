"""Install dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def load_tf_version_specific_dependencies():
    """Load TF version specific dependencies."""
    http_archive(
        name = "com_google_absl",
        sha256 = "94aef187f688665dc299d09286bfa0d22c4ecb86a80b156dff6aabadc5a5c26d",
        strip_prefix = "abseil-cpp-273292d1cfc0a94a65082ee350509af1d113344d",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/273292d1cfc0a94a65082ee350509af1d113344d.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/273292d1cfc0a94a65082ee350509af1d113344d.tar.gz",
        ],
    )

    http_archive(
        name = "zlib",
        build_file = "//third_party:zlib.BUILD",
        sha256 = "91844808532e5ce316b3c010929493c0244f3d37593afd6de04f71821d5136d9",
        strip_prefix = "zlib-1.2.12",
        urls = [
            "http://mirror.tensorflow.org/zlib.net/zlib-1.2.12.tar.gz",
            "https://zlib.net/zlib-1.2.12.tar.gz",
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
