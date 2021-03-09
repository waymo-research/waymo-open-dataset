FROM tensorflow/tensorflow:2.4.1-gpu

# Install apt dependencies.
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    wget

# Download the repo.
RUN git clone https://github.com/tensorflow/models.git
# Compile protos
RUN cd models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    python -m pip install .


# Download the EfficientDet model.
RUN wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz && \
    tar xf efficientdet_d5_coco17_tpu-32.tar.gz && \
    mv efficientdet_d5_coco17_tpu-32 models/research/object_detection/test_data/ && \
    rm efficientdet_d5_coco17_tpu-32.tar.gz

# Copy the use rmodule code to the /user_model/ directory and put that
# directory in the PYTHONPATH so that wod_latency_submission can be imported
# from anywhere.
ENV PYTHONPATH=/user_model/
COPY wod_latency_submission/ /user_model/wod_latency_submission/
