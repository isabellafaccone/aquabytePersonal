FROM python:3

RUN apt-get update && \
    apt-get install -y \
        vim

COPY src /app/
WORKDIR /app

RUN pip install -r requirements.txt

