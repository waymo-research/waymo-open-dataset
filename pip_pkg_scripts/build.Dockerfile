FROM tensorflow/tensorflow:custom-op-ubuntu16

ENV GITHUB_BRANCH="master"
ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION=""
ENV PIP_MANYLINUX2010="1"

RUN apt-get update && apt-get install -y && \
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && apt-get install -y bazel && \
    rm -rf /usr/local/bin/bazel && hash -r

RUN apt-get install python3.5
RUN apt-get install python3.6

RUN pip install --upgrade setuptools
RUN pip3 install --upgrade setuptools

# Install tensorflow
RUN pip install tensorflow==1.14.0
RUN pip3 install tensorflow==1.14.0

RUN pip3 install --upgrade auditwheel
COPY pip_pkg_scripts/build.sh /

VOLUME /tmp/pip_pkg_build

ENTRYPOINT ["/build.sh"]

# The default parameters for the build.sh
CMD []
