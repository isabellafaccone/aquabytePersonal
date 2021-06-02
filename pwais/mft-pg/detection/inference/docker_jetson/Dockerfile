FROM nvcr.io/nvidia/l4t-ml:r32.4.3-py3

ENV DEBIAN_FRONTEND noninteractive

# NOTE:
# Your host needs things like: libnvinfer
# If you can't build this container, it might actually be your host
# that is missing dependencies.

# Doh, the base image is missing OpenCV
RUN \
  apt-get update && \
  apt-get install -y --no-install-recommends  python3-opencv

# # Doh, the base image is missing libnvToolsExt.so
# # FMI https://forums.developer.nvidia.com/t/how-can-i-install-the-pytorch/108038/9
# RUN \


RUN \
  apt-get update && apt-get install -y \
    sudo wget curl vim git less
RUN python3 -m pip install --upgrade pip

RUN pip3 install \
      imageio \
      pandas==1.1.5 \
      pycuda \
      ipython \
      imageio-ffmpeg \
      click \
      jupyter \
      mlflow==1.14.1

# Thanks alot click
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN \
  mkdir -p /opt && cd /opt && \
  git clone https://github.com/jkjung-avt/tensorrt_demos && \
  cd tensorrt_demos && git checkout 0016973315d1f3f6eaed70a5abd03d6309fe4730
RUN \
  cd /opt/tensorrt_demos && \
  cd ssd && ./install_pycuda.sh && cd - && \
  pip3 install onnx==1.4.1

# Doh, the base image is missing python tensorRT headers as well as the 
# python tensorrt package
# RUN \
#   apt-get update && apt-get install -y software-properties-common && \
#   apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
#   apt-get update && \
#   echo "deb https://repo.download.nvidia.com/jetson/common r32.4 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
#   echo "deb https://repo.download.nvidia.com/jetson/t186 r32.4 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
#   apt-get update && \
#   apt-get install -y \
#     nvidia-container-csv-cuda \
#     nvidia-container-csv-cudnn \
#     nvidia-container-csv-tensorrt \
#     nvidia-container-csv-visionworks \
#     cuda-curand-dev-10-2 \
#     cuda-toolkit-10-2 \
#     cuda-tools-10-2 \
#     python3-libnvinfer-dev \
#     python3-opencv
  # (apt-get install -y --no-install-recommends python3-libnvinfer-dev || true)
# blarg the above dies with
# unable to remove newly-extracted version of '/usr/lib/python3.6/dist-packages/tensorrt/__init__.py': Read-only file system

# Note this won't work for arm / jetson :(
# RUN \
#   pip3 install nvidia-pyindex && \
#   pip3 install --upgrade nvidia-tensorrt

# RUN \
#   cd /opt/tensorrt_demos && \
#   cd plugins && make && cd -

RUN \
  echo "need to edit tensorrt_demos have /opt/tensorrt_demos/plugins/libyolo_layer.so instead of ./plugins/libyolo_layer.so :(" && \
  sed -i 's/\.\/plugins\/libyolo_layer\.so/\/opt\/tensorrt_demos\/plugins\/libyolo_layer\.so/g' /opt/tensorrt_demos/utils/yolo_with_plugins.py && \
  sed -i 's/\.\.\/plugins\/libyolo_layer\.so/\/opt\/tensorrt_demos\/plugins\/libyolo_layer\.so/g' /opt/tensorrt_demos/yolo/plugins.py

ENV PYTHONPATH $PYTHONPATH:/opt/tensorrt_demos


# Install tracker stuff
RUN cd /opt/ && \
  git clone https://github.com/adipandas/multi-object-tracker  && \
    cd multi-object-tracker && \
      git checkout 501605262a120f1ab47658f423fa180a6f36117c && \
        # pip3 install -r requirements.txt && \
        pip3 install ipyfilechooser
        #&& \
        # pip3 install -e .
ENV PYTHONPATH $PYTHONPATH:/opt/multi-object-tracker

# RUN \
#   echo "somebody breaks numpy" && \
#   rm -rf /usr/local/lib/python3.6/dist-packages/numpy/ && \
#   apt-get install --reinstall python3-numpy
