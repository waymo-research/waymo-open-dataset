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

# Install python3.6
RUN mkdir /build_from_source && cd /build_from_source
RUN wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tgz \
  && tar -xvf Python-3.6.3.tgz \
  && cd Python-3.6.3 \
  && ./configure \
  && make -j 16 \
  && make altinstall \
  && cd ../../

RUN wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz \
  && tar -xf Python-3.7.4.tgz \
  && cd Python-3.7.4 \
  && ./configure \
  && make -j 16 \
  && make altinstall \
  && cd ../../

# Install python 3.8
RUN wget https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz \
  && tar -xf Python-3.8.1.tgz \
  && cd Python-3.8.1 \
  && ./configure \
  && make -j 16 \
  && make altinstall \
  && cd ../../

# Install python 3.9
RUN wget https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz \
  && tar -xf Python-3.9.1.tgz \
  && cd Python-3.9.1 \
  && ./configure \
  && make -j 16 \
  && make altinstall \
  && cd ../../

# Update pip and tools on python 3.6
RUN python3.6 -m pip install --upgrade pip setuptools auditwheel grpcio>=1.24.3

# Install pip and tools for other python(s)
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN for python in python3.7 python3.8 python3.9; do \
      $python get-pip.py && \
      $python -m pip install --upgrade pip && \
      $python -m pip install --upgrade setuptools auditwheel && \
      $python -m pip install --upgrade grpcio>=1.24.3; \
    done

VOLUME /tmp/artifacts
COPY . /tmp/repo
WORKDIR /tmp/repo

ENTRYPOINT ["pip_pkg_scripts/build.sh"]

# The default parameters for the build.sh
CMD []
