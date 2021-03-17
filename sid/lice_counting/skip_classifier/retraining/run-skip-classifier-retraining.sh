#!/bin/bash

TODAY=$(date "+%Y-%m-%d")

echo "Writing logs to /root/data/skip-classifier/logs/$TODAY"

time -p /home/user/miniconda/envs/py36/bin/python3 /root/skip_classifier/retraining/skip-classifier-retraining.py >> /root/data/skip-classifier/logs/$TODAY 2>&1

