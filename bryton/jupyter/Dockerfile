FROM tensorflow/tensorflow:latest

RUN apt-get update && \
    apt-get install -y \
	build-essential \
	cmake \
        gfortran \
        git \
        git-core \
	libatlas-base-dev \
        libfontconfig1 \
        libgconf-2-4 \
        libglu1 \
	libgtk2.0-dev \
        libhdf5-dev \
	libjasper-dev \
        libjpeg8-dev \
	libpng12-dev \
        libsm6 \
        libxext6 \
        libxi6 \
        libxrender1 \
        pkg-config \
        python3 \
	python3-dev \
        python3-tk \
        python3-pip \
	screen \
        ssh \
        unzip \
        vim \
        wget \
        zip


WORKDIR /root
COPY requirements.txt /root/requirements.txt
RUN pip3 install -r requirements.txt

COPY run_jupyter.sh /
EXPOSE 8889
CMD ["/run_jupyter.sh", "--ip=0.0.0.0", "--allow-root", "--port=8889"]

