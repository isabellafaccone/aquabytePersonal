#!/bin/bash

INSTANCE_TYPE=m5.12xlarge
CODE_LOCATION=s3://aquabyte-research/template-matching/main.py

#
NUM_INSTANCES=4
CORES=48
MEM=192
NUM_CORES=$(expr $(expr $CORES-1)\*$NUM_INSTANCES)
AVAIL_EXECUTORS=$(expr $NUM_CORES/5)
NUM_EXECUTORS=$(expr $AVAIL_EXECUTORS-1)
EXECUTOR_MEM=$(expr $MEM/$AVAIL_EXECUTORS)

aws emr create-cluster --name "Aloks Spicy Cluster" --release-label emr-5.27.0 --applications Name=Spark \
    --instance-type $INSTANCE_TYPE --instance-count $NUM_INSTANCES \
    --steps Type=Spark,Name="template matching",ActionOnFailure=TERMINATE_CLUSTER,Args=[--deploy-mode,cluster,--master,yarn,--conf,spark.yarn.submit.waitAppCompletion=false,--num-executors,$NUM_EXECUTORS,--executor-cores,5,--executor-memory,${EXECUTOR_MEM}g,$CODE_LOCATION] --use-default-roles --auto-terminate --configurations https://aquabyte-research.s3.amazonaws.com/emr/config.json --bootstrap-actions Path=s3://aquabyte-research/emr/bootstrap.sh
