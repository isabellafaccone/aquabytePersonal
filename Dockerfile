FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && \
    apt-get install -y && \
        git \
        git-core \
        opencv-python \
        vim \
        zip


RUN mkdir /root/models/
RUN git clone https://github.com/fizyr/keras-retinanet.git /root/models/
RUN git clone https://github.com/matterport/Mask_RCNN.git /root/models/

RUN pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

COPY /src/requirements.txt /root/
RUN pip install -r /root/requirements.txt

WORKDIR /root/