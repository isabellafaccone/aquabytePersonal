#!/bin/bash

python3 generate_dataframe.py
python3 data_new_bryton.py
python3 train_new_bryton.py --fname 01152020_bodyparts

