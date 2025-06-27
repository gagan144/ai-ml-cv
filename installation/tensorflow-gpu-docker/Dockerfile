FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    python3-pip curl git wget unzip nano \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/local/bin/pip3 /usr/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install cuDNN 9.3 (manual install from NVIDIA)
# NOTE: You must manually download cuDNN 9.3 tar from NVIDIA website and copy it here.
# Example assumes you've placed the tar in the same folder as the Dockerfile.

COPY cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz /tmp/.
RUN tar -xJf /tmp/cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz -C /usr/local && \
    rm /tmp/cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz && \
    ldconfig

# Install TensorFlow 2.19
RUN pip install --no-cache-dir tensorflow==2.19.0

# Copy verification script
COPY verify_tensorflow.py /verify_tensorflow.py

# Run the script by default
CMD ["python", "/verify_tensorflow.py"]
