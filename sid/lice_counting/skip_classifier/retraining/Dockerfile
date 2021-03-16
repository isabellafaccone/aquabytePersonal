FROM anibali/pytorch:cuda-9.0
USER root
COPY requirements.txt /root/
COPY research-0.1.1.tar.gz /app/
RUN apt-get update && \
    apt-get install -y \
        gcc \
        git \
        git-core \
        libfontconfig1 \
        libgconf-2-4 \
        libglu1 \
        libpq-dev \
        libsm6 \
        libxext6 \
        libxi6 \
        libxrender1 \
        python-dev \
        unrar \
        vim \
        wget \
	cron
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Copy hello-cron file to the cron.d directory
COPY crontab /etc/cron.d/crontab
 
# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/crontab

# Apply cron job
RUN crontab /etc/cron.d/crontab
 
# Create the log file to be able to run tail
RUN touch /var/log/cron.log

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r /root/requirements.txt
WORKDIR /root/

CMD ["cron", "-f"]