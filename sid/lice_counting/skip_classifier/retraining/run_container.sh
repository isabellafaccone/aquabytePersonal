# docker run -p 7777:7777  -v /home/ubuntu/bryton/research-exploration/sid/lice_counting/skip_classifier:/root/skip_classifier -v /data:/root/data --gpus all --ipc=host --shm-size=256m --name skip-classifier-retraining skip-classifier-retraining:v1 sleep 1000h
docker rm -f skip-classifier-retraining
docker-compose up
