#!/bin/bash

EXEC_FILE=pyspark_template_matching.py
INSTANCE_TYPE=m5.2xlarge
CODE_LOCATION=s3://aquabyte-research/template-matching/code/main_repartition.py

#
NUM_INSTANCES=17

aws s3 cp $EXEC_FILE $CODE_LOCATION

aws emr create-cluster --name "Aloks Spicy Cluster" --release-label emr-5.27.0 --applications Name=Spark \
    --instance-type $INSTANCE_TYPE --instance-count $NUM_INSTANCES \
    --steps Type=Spark,Name="template matching",ActionOnFailure=TERMINATE_CLUSTER,Args=[--deploy-mode,cluster,--master,yarn,$CODE_LOCATION] \
    --use-default-roles --auto-terminate  --configurations https://aquabyte-research.s3.amazonaws.com/emr/config.json \
    --bootstrap-actions Path=s3://aquabyte-research/emr/bootstrap.sh --log-uri s3://aws-logs-286712564201-eu-west-1/elasticmapreduce/ --enable-debugging

