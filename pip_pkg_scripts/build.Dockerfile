FROM tensorflow/tensorflow:custom-op-ubuntu16

ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION=""
ENV PIP_MANYLINUX2010="1"
ENV TF_VERSION="2.6.0"
RUN wget https://github.com/bazelbuild/bazel/releases/download/4.0.0/bazel-4.0.0-installer-linux-x86_64.sh > /dev/null
RUN bash bazel-4.0.0-installer-linux-x86_64.sh
# There are some problems with the python3 installation from custom-op-ubuntu16.
# Remove it and install new ones.
RUN apt-get remove --purge -y python3.5 python3.6
RUN rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list
RUN apt-key del F06FC659

# Deadsnakes PPA no longer supports 16.04
# https://github.com/deadsnakes/issues/issues/195
# So we build all python versions here
RUN mkdir /tmp/python
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libssl-dev zlib1g-dev openssl libffi-dev

RUN for v in 3.6.15 3.7.12 3.8.12 3.9.10; do \
    wget "https://www.python.org/ftp/python/$v/Python-${v}.tar.xz" && \
    tar xvf "Python-${v}.tar.xz" -C /tmp/python && \
    cd "/tmp/python/Python-${v}" && \
    ./configure && \
    make -j8 altinstall; \
  done


RUN apt-get install -y libopenexr-dev
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN for python in python3.6 python3.7 python3.8 python3.9; do \
      $python get-pip.py && \
      $python -m pip install --upgrade pip setuptools auditwheel && \
      $python -m pip install --upgrade grpcio>=1.24.3; \
      $python -m pip install --upgrade matplotlib plotly scikit-image immutabledict; \
      $python -m pip install --upgrade OpenEXR tensorflow_graphics; \
      $python -m pip install --upgrade tensorflow==${TF_VERSION}; \
    done

VOLUME /tmp/artifacts
VOLUME /root
COPY . /tmp/repo
WORKDIR /tmp/repo

ENTRYPOINT ["pip_pkg_scripts/build.sh"]

# The default parameters for the build.sh
CMD []
