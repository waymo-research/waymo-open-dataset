"""Build rule to build cc and py protos."""

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")

_SUFFIXES = {
    "proto": "_proto",
    "py": "_proto_py_pb2",
    "cc": "_cc_proto",
}

def _target_name(lang, base_name):
    return base_name + _SUFFIXES[lang]

def _get_basename(name):
    suffix = _SUFFIXES["proto"]
    if not name.endswith(suffix):
        fail("Unexpected build rule name: %s" % name)
    return name[:-len(suffix)]

def _deps(lang, deps):
    return [_target_name(lang, _get_basename(d)) for d in deps]

def all_proto_library(src, deps = None):
    """Adds build rulles to build python and c++ proto libraries.

    Args:
      src: proto file.
      deps: optional list with proto dependensies.
    """
    if deps == None:
        deps = []
    base_name = src[:-len(".proto")]
    py_proto_library(name = _target_name("py", base_name), srcs = [src], deps = _deps("py", deps))
    cc_proto_library(name = _target_name("cc", base_name), srcs = [src], deps = _deps("cc", deps))
