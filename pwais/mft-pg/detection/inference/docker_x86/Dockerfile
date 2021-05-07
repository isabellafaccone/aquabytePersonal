FROM nvcr.io/nvidia/tensorrt:20.08-py3

# Install Nvidia's extras to get ONNX, numpy, etc
RUN \
  /opt/tensorrt/install_opensource.sh && /opt/tensorrt/python/python_setup.sh

ENV DEBIAN_FRONTEND noninteractive
RUN \
  apt-get update && apt-get install -y \
    sudo wget curl vim git \
    python3-pip python3-opencv
RUN python3 -m pip install --upgrade pip

# Install TRT demo stuff for yolo
RUN \
  mkdir -p /opt && cd /opt && \
  git clone https://github.com/jkjung-avt/tensorrt_demos && \
  cd tensorrt_demos && git checkout 0016973315d1f3f6eaed70a5abd03d6309fe4730
RUN \
  cd /opt/tensorrt_demos && \
  cd ssd && ./install_pycuda.sh && cd - && \
  pip3 install onnx==1.4.1 && \
  cd plugins && make && cd -
ENV PYTHONPATH $PYTHONPATH:/opt/tensorrt_demos

RUN \
  echo "need to edit tensorrt_demos have /opt/tensorrt_demos/plugins/libyolo_layer.so instead of ./plugins/libyolo_layer.so :(" && \
  sed -i 's/\.\/plugins\/libyolo_layer\.so/\/opt\/tensorrt_demos\/plugins\/libyolo_layer\.so/g' /opt/tensorrt_demos/utils/yolo_with_plugins.py && \
  sed -i 's/\.\.\/plugins\/libyolo_layer\.so/\/opt\/tensorrt_demos\/plugins\/libyolo_layer\.so/g' /opt/tensorrt_demos/yolo/plugins.py
