FROM tensorflow/tensorflow:latest-py3

RUN apt-get update && apt-get install -y \
  git build-essential wget vim bazel \
  pkg-config zip g++ zlib1g-dev unzip python3 python3-pip

RUN pip3 install jupyter matplotlib jupyter_http_over_ws &&\
  jupyter serverextension enable --py jupyter_http_over_ws

RUN git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
WORKDIR /waymo-od

RUN bash ./configure.sh && bazel build waymo_open_dataset/metrics:all
RUN bash bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
RUN bash bazel test ...

EXPOSE 8888
RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/waymo-od --ip 0.0.0.0 --port 8888 --no-browser --allow-root"]
