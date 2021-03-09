FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y curl && \
    mkdir -p /root/.cache/torch/hub/checkpoints/ && \
    curl https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth > /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

# Copy the use rmodule code to the /user_model/ directory and put that
# directory in the PYTHONPATH so that wod_latency_submission can be imported
# from anywhere.
ENV PYTHONPATH=/user_model/
COPY wod_latency_submission/ /user_model/wod_latency_submission/
