#!/bin/bash

docker run --name=research -v /Users/bryton/Documents/aquabyte/data/:/root/data/ -v /Users/bryton/Documents/aquabyte/:/root/bryton/ -p 8889:8889 -ti algorithms
