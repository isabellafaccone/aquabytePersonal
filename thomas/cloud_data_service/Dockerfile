FROM python:2.7

RUN apt-get update && \
    apt-get install -y \
        vim

COPY src /root/
WORKDIR /root

RUN pip install -r requirements.txt

RUN pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

COPY deploy/credentials.json /root/

ENTRYPOINT ["/root/entrypoint.sh"]