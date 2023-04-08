FROM tensorflow/tensorflow:latest-py3

RUN apt-get update && apt-get install -y \
  git build-essential wget vim findutils curl \
  pkg-config zip g++ zlib1g-dev unzip python3 python3-pip

RUN apt-get install -y wget golang
RUN go install github.com/bazelbuild/bazelisk@latest


RUN pip3 install jupyter matplotlib jupyter_http_over_ws &&\
  jupyter serverextension enable --py jupyter_http_over_ws

RUN git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
WORKDIR /waymo-od/src

RUN pip_pkg_scripts/build.sh

EXPOSE 8888
RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && bazel run -c opt //waymo_open_dataset/tutorial:jupyter_kernel"]
