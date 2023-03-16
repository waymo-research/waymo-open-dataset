FROM tensorflow/tensorflow:2.3.0-custom-op-ubuntu16

ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION=""
ENV PIP_MANYLINUX2010="1"
ENV TF_VERSION="2.6.0"
# There are some problems with the python3 installation from custom-op-ubuntu16.
# Remove it and install new ones.
RUN apt-get remove --purge -y python3.5 python3.6
RUN rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list
RUN apt-key del F06FC659
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
ENV PIP_ROOT_USER_ACTION=ignore

# Deadsnakes PPA no longer supports 16.04
# https://github.com/deadsnakes/issues/issues/195
# So we build all python versions here
RUN mkdir /tmp/python
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libssl-dev zlib1g-dev openssl libffi-dev openexr libopenexr-dev

RUN for v in 3.7.12 3.8.12 3.9.10; do \
    wget "https://www.python.org/ftp/python/$v/Python-${v}.tar.xz" && \
    tar xvf "Python-${v}.tar.xz" -C /tmp/python && \
    cd "/tmp/python/Python-${v}" && \
    ./configure && \
    make -j8 altinstall; \
  done


RUN BAZEL_VERSION="$(wget https://raw.githubusercontent.com/tensorflow/tensorflow/v2.9.0/.bazelversion -O -)"&& \
  wget "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" > /dev/null && \
  bash "bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN for python in python3.7 python3.8 python3.9; do \
      $python get-pip.py && \
      $python -m pip install --upgrade pip setuptools auditwheel && \
      $python -m pip install --upgrade grpcio>=1.24.3; \
      $python -m pip install --upgrade matplotlib plotly scikit-image immutabledict scipy sklearn absl-py pandas==1.4 numpy pyarrow dask[array]; \
      $python -m pip install --upgrade tensorflow==${TF_VERSION} OpenEXR==1.3.2 tensorflow_graphics; \
    done

VOLUME /tmp/artifacts
VOLUME /root
COPY . /tmp/repo
WORKDIR /tmp/repo

ENTRYPOINT ["pip_pkg_scripts/build.sh"]

# The default parameters for the build.sh
CMD []
