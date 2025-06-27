# Tensorflow GPU Docker Image

## REQUIREMENTS
- Download CuDNN source code: https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz

## BUILD
```shell
docker build -t tensorflow-gpu-219-cuda125-cudnn9-py311 .
```

## RUN
```shell
docker run --rm --gpus all tensorflow-gpu-219-cuda125-cudnn9-py311
```

## SETUP NVIDIA CONTAINER TOOLKIT
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

```shell
# Get repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update 
sudo apt-get update

# Install
sudo apt-get install -y nvidia-container-toolkit
# OR
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure

# Restart Docker
sudo systemctl restart docker
```