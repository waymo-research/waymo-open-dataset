# Dockerfile for Waymo Open Dataset Local Tutorial with Jupyter Notebook

# Base image with TensorFlow 2.11.0 (CPU-only, pinned to ensure reproducibility)
FROM tensorflow/tensorflow:2.11.0  

# Install system dependencies required for building and running the tutorial
RUN apt-get update && apt-get install -y \
  git build-essential wget vim findutils curl \
  pkg-config zip g++ zlib1g-dev unzip python3 python3-pip && \
  apt-get clean  # Clean up to reduce image size

# Install Python packages with pinned versions for stability
# jupyter: Main package for notebook support
# notebook: Specific version for compatibility
# matplotlib: For visualization in the tutorial
RUN pip3 install jupyter==1.0.0 notebook==7.0.8 matplotlib==3.7.5

# Clone the Waymo Open Dataset repository for tutorial files
RUN git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od

# Upgrade pip to ensure compatibility with the latest wheel installation
RUN pip3 install --upgrade pip

# Install the Waymo Open Dataset wheel from PyPI
# --target specifies where to install the package for Jupyter to find it
RUN pip3 install waymo-open-dataset-tf-2-11-0 --target /usr/local/lib/python3.8/site-packages

# Expose port 8888 for Jupyter notebook access
EXPOSE 8888  
RUN python3 -m ipykernel.kernelspec

# Set PYTHONPATH to include the wheel location
ENV PYTHONPATH=/usr/local/lib/python3.8/site-packages:/usr/local/lib/python3.8/dist-packages

# Command to start Jupyter notebook
# --ip=0.0.0.0: Allows access from any IP (use with caution, local only recommended)
# --port=8888: Matches the exposed port
# --no-browser: Disables browser launch (no GUI in container)
# --NotebookApp.token='' and --NotebookApp.password='': Disables authentication (security risk, use locally)
# --NotebookApp.disable_check_xsrf=True: Improves performance by disabling CSRF check
# --notebook-dir=/waymo-od/tutorial: Sets the starting directory to the tutorial folder
# --allow-root: Allows running as root (required since no user switch is implemented)
CMD ["python3", "-m", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.disable_check_xsrf=True", "--notebook-dir=/waymo-od/tutorial", "--allow-root"]