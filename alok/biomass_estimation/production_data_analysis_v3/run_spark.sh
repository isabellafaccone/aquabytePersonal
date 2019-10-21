#!/bin/bash

EXEC_FILE=pyspark_template_matching.py
INSTANCE_TYPE=r5.2xlarge
#INSTANCE_TYPE=c5.4xlarge
#INSTANCE_TYPE=m5.4xlarge
#INSTANCE_TYPE=c5.9xlarge
CODE_LOCATION=s3://aquabyte-research/template-matching/code/main_repartition.py

#
NUM_INSTANCES=9
let NUM_INSTANCES_M1=$NUM_INSTANCES-1
CORES=16
MEM=64
NUM_EXEC=5
let CORES_M1=$CORES-1
let NUM_CORES=$CORES_M1*$NUM_INSTANCES_M1
let AVAIL_EXECUTORS=$NUM_CORES/$NUM_EXEC
let NUM_EXECUTORS=$AVAIL_EXECUTORS-1
let EXECUTOR_MEM=$MEM/$AVAIL_EXECUTORS

#aws s3 cp $EXEC_FILE $CODE_LOCATION

aws emr create-cluster --name "Aloks Spicy Cluster" --release-label emr-5.27.0 --applications Name=Spark \
    --instance-type $INSTANCE_TYPE --instance-count $NUM_INSTANCES \
    --steps Type=Spark,Name="template matching",ActionOnFailure=TERMINATE_CLUSTER,Args=[--deploy-mode,cluster,--master,yarn,$CODE_LOCATION] \
    --use-default-roles --auto-terminate  --configurations https://aquabyte-research.s3.amazonaws.com/emr/config.json \
    --bootstrap-actions Path=s3://aquabyte-research/emr/bootstrap.sh --log-uri s3://aws-logs-286712564201-eu-west-1/elasticmapreduce/ --enable-debugging

#aws emr create-cluster --name "Aloks Spicy Cluster" --release-label emr-5.27.0 --applications Name=Spark \
#    --instance-type $INSTANCE_TYPE --instance-count $NUM_INSTANCES \
#    --steps Type=Spark,Name="template matching",ActionOnFailure=TERMINATE_CLUSTER,Args=[--deploy-mode,cluster,--master,yarn,--num-executors,$NUM_EXECUTORS,--executor-cores,$NUM_EXEC,--executor-memory,${EXECUTOR_MEM}g,$CODE_LOCATION] --use-default-roles --auto-terminate --configurations https://aquabyte-research.s3.amazonaws.com/emr/config.json --bootstrap-actions Path=s3://aquabyte-research/emr/bootstrap.sh --log-uri s3://aws-logs-286712564201-eu-west-1/elasticmapreduce/ --enable-debugging

#spark.executor.memory = (yarn.scheduler.maximum-allocation-mb – 1g) -spark.yarn.executor.memoryOverhead
#spark.executor.instances = [this is set to the initial number of core nodes plus the number of task nodes in the cluster]
#spark.executor.cores = yarn.nodemanager.resource.cpu-vcores
#spark.default.parallelism = spark.executor.instances * spark.executor.cores
