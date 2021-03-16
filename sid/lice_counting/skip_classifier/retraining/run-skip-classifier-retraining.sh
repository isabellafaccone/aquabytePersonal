#!/bin/bash

TODAY=$(date "+%Y-%m-%d")

echo "Writing logs to /root/data/skip-classifier/logs/$TODAY"

python3 skip-classifier-retraining.py >> /root/data/skip-classifier/logs/$TODAY 2>&1

