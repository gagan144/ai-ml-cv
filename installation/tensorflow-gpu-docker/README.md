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