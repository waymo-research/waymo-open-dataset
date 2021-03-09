FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# The submission directory contains the code and associated libraries that are
# clonein future steps.
RUN mkdir -p /code/submission
WORKDIR /code/submission

# Get packages needed for builds below.
RUN apt-get update \
  && apt-get install -y wget unzip g++ git python3-pip libgl1-mesa-glx python3 \
                        libboost-all-dev llvm-10 libsm6 libxrender-dev \
  && rm -rf /var/lib/apt/lists/*

# Install the specific version of PyTorch recommended for the libraries below.
RUN pip3 install --upgrade pip \
  && pip3 install torch==1.3.0 torchvision==0.4.1

# CMake is required for builds.
RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.4/cmake-3.19.4-Linux-x86_64.tar.gz \
  && tar xf cmake-3.19.4-Linux-x86_64.tar.gz \
  && rm -f cmake-3.19.4-Linux-x86_64.tar.gz

ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    PATH=$PATH:/code/submission/cmake-3.19.4-Linux-x86_64/bin/ \
    LLVM_CONFIG=/usr/bin/llvm-config-10 \
    PYTHONPATH=/code/submission/lib:$PYTHONPATH \
    TF_CPP_MIN_LOG_LEVEL=1

# Install dependency.
RUN git clone --recursive https://github.com/traveller59/spconv.git \
  && cd spconv \
  && python3 setup.py bdist_wheel \
  && cd dist \
  && pip3 install *

# Install prediction library used by this PV-RCNN implementation.
RUN git clone https://github.com/open-mmlab/OpenPCDet.git \
  && cd OpenPCDet \
  && pip3 install -r requirements.txt \
  && pip3 install vtk \
  && pip3 install mayavi \
  && python3 setup.py develop

# Python should default to python3 for this code.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0

# Copy in the folder with the submission module and download the model weights
# from https://github.com/open-mmlab/OpenPCDet/blob/master/README.md.
COPY submission /code/submission/
RUN wget -O /code/submission/lib/wod_latency_submission/WAYMO_MODEL_WEIGHTS.pth \
  'https://drive.google.com/u/1/uc?id=1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-&export=download'

# Set the working directly correctly so to ensure access to some config files.
WORKDIR /code/submission/OpenPCDet/tools
