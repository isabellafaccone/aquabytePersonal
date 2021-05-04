# Based upon https://github.com/daisukekobayashi/darknet-docker/blob/master/Dockerfile.gpu-cv
# NB: we need cuda 9.0 for use on the Aquabyte lambda Quad machines
ARG BASE_IMAGE=nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
FROM $BASE_IMAGE AS builder

ENV DEBIAN_FRONTEND noninteractive

# Identify this image as the darknet training image
RUN touch /opt/i_am_darknet_trainer

# Create an empty scratch dir; on the lambda quads use the 8TB drive
RUN mkdir -p /opt/mft-pg-scratch

RUN apt-get update \
      && apt-get install --no-install-recommends --no-install-suggests -y gnupg2 ca-certificates \
            git build-essential libopencv-dev \
      && rm -rf /var/lib/apt/lists/*

# This script will help build darknet with the right dependencies & for the right CUDA
COPY configure.sh /tmp/

#ARG SOURCE_BRANCH=unspecified
#ENV SOURCE_BRANCH $SOURCE_BRANCH

# darknet_yolo_v3  tag 
ARG SOURCE_COMMIT=5e850c24897a5eb65941703059a85ead2ea5ff8c
ENV SOURCE_COMMIT $SOURCE_COMMIT

ARG CONFIG

RUN mkdir -p /opt && cd /opt && \
      git clone https://github.com/AlexeyAB/darknet && cd darknet \
      && git checkout $SOURCE_BRANCH \
      && git reset --hard $SOURCE_COMMIT \
      && /tmp/configure.sh $CONFIG && make -j `nproc` \
      && cp darknet /usr/local/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/darknet/libdarknet.so
ENV PYTHONPATH $PYTHONPATH:/opt/darknet

RUN \
  echo "need to edit darknet.py to have /opt/darknet/libdarknet.so instead of ./libdarknet.so :(" && \
  sed -i 's/\.\/libdarknet\.so/\/opt\/darknet\/libdarknet\.so/g' /opt/darknet/darknet.py


RUN apt-get update \
      && apt-get install -y \
            python-opencv \
            libopencv-dev \
            python3-dev \
            vim less curl wget \
            python3-pip

RUN curl -sSL https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py && \
      python3 get-pip.py "pip < 21.0" "setuptools < 50.0" "wheel < 1.0"
RUN pip3 install \
      imageio \
      pandas \
      ipython \
      imageio-ffmpeg \
      click \
      mlflow==1.14.1

# Thanks alot click
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ENV PYTHONPATH $PYTHONPATH:/opt/mft-pg/mft_utils
