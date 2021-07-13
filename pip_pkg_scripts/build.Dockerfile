FROM tensorflow/tensorflow:custom-op-ubuntu16

ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION=""
ENV PIP_MANYLINUX2010="1"
ENV TF_VERSION="2.5.0"
RUN wget https://github.com/bazelbuild/bazel/releases/download/4.0.0/bazel-4.0.0-installer-linux-x86_64.sh > /dev/null
RUN bash bazel-4.0.0-installer-linux-x86_64.sh
# There are some problems with the python3 installation from custom-op-ubuntu16.
# Remove it and install new ones.
RUN apt-get remove --purge -y python3.5 python3.6
RUN rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list
RUN apt-key del F06FC659
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776
RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main" > /etc/apt/sources.list.d/deadsnakes-ppa-xenial.list
RUN apt-get update && apt-get install -y python3.6 python3.7 python3.8 python3.8-distutils python3.9 python3.9-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN for python in python3.6 python3.7 python3.8 python3.9; do \
      $python get-pip.py && \
      $python -m pip install --upgrade pip setuptools auditwheel && \
      $python -m pip install --upgrade grpcio>=1.24.3; \
    done

VOLUME /tmp/artifacts
COPY . /tmp/repo
WORKDIR /tmp/repo

ENTRYPOINT ["pip_pkg_scripts/build.sh"]

# The default parameters for the build.sh
CMD []
