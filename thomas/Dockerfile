FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && \
    apt-get install -y \
        git \
        git-core \
        libfontconfig1 \
        libgconf-2-4 \
        libglu1 \
        libsm6 \
        libxext6 \
        libxi6 \
        libxrender1 \
        unrar \
        vim \
        wget

COPY /src/requirements.txt /root/
RUN pip install -r /root/requirements.txt

COPY src /root/aquabyte_trainer
RUN cd /root/aquabyte_trainer/fish_detection/aquabyte_retinanet/ && \
    pip install .

RUN pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

WORKDIR /root/