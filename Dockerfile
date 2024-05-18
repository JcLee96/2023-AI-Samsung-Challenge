# For more information, please refer to https://aka.ms/vscode-docker-python
#FROM ubuntu:18.04
ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update
ADD . /samsung2023_iqa_solution
WORKDIR /samsung2023_iqa_solution

#RUN python -m pip install --upgrate pip
#RUN python -m pip install -r requirements.txt

# COPY requirements.txt .
# RUN pip install -r requirements.txt

## Keeps Python from generating .pyc files in the container
#ENV PYTHONDONTWRITEBYTECODE=1
#
## Turns off buffering for easier container logging
#ENV PYTHONUNBUFFERED=1

#COPY requirements.txt .
#RUN pip install -r /requirements.txt

## Install pip requirements
#COPY requirements.txt .
#RUN python -m pip install -r requirements.txt

#WORKDIR /app
#COPY . /app
#
## Creates a non-root user with an explicit UID and adds permission to access the /app folder
## For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
#RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
#USER appuser
#
## During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#CMD ["python", "train.py"]
