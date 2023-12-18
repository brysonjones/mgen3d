FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Install apt dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git pip ffmpeg libsm6 libxext6 wget unzip

# Pip dependencies
RUN pip install black pre-commit

ADD . /mgen3d
WORKDIR /mgen3d
RUN pip install -e .
RUN pip install --default-timeout=900 git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
RUN pre-commit install