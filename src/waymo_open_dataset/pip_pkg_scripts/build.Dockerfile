# We Keep it to a bare minumum to run bazel build to build the pip package (wheel).
# An official docker image to build manylinux2014_x86_64 wheels.
FROM quay.io/pypa/manylinux_2_28_x86_64

# 1. Update yum and install essential build tools + NumPy dependencies
RUN yum update -y && \
    yum install -y gcc-gfortran openblas-devel && \
    yum clean all # Clean up yum cache to reduce image size

# go is required to use bazelisk
RUN yum -y install sudo golang clang

# Almost all dependencies are defined via `requirements.in`.
# Dependencies which can't be installed via pip.
RUN yum -y install OpenEXR-devel.x86_64

# === Select Python Version ===
# Set the PATH to include the desired Python version (3.11) from /opt/python
# This makes `python3`, `pip3` etc. point to the 3.11 versions.
ENV CPYTHON_PATH="/opt/python/cp311-cp311/bin"
ENV PATH="${CPYTHON_PATH}:${PATH}"

# === Upgrade build tools for the selected Python ===
RUN pip install --upgrade pip setuptools wheel

# Create a non-root user.
# Recent version of bazelbuild/rules_python will fails if you bazel build as a root.
# https://github.com/bazelbuild/rules_python/pull/713
ARG USERNAME=package-manager
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME

# Install bazelisk
RUN go install github.com/bazelbuild/bazelisk@latest
ENV PATH="$PATH:/home/${USERNAME}/go/bin"


COPY --chown=${USERNAME}:${USERNAME} . /tmp/repo
WORKDIR /tmp/repo

ENTRYPOINT ["waymo_open_dataset/pip_pkg_scripts/build.sh"]

# The default parameters for the build.sh
CMD []
