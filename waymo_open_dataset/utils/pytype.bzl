"""Placeholders for pytype build targets until bazel supports them."""

# Placeholder to use until bazel supports py_strict_binary.
def py_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_library.
def py_strict_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_binary.
def pytype_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_library.
def pytype_strict_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_test.
def py_strict_test(name, **kwargs):
    native.py_test(name = name, **kwargs)
