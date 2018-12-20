# Cloud data service

This micro-service downloads data from the cloud and organize it in specific folders on the local server. It downloads data for GSTF,
body parts segmentation and lice annotations.

## Getting Started


### Prerequisites

To test and deploy this service, docker and docker-compose needs to be installed.

## Deployment

Place the SQL and S3 json credentials in the deploy folder.
 
The SQL credentials must be under the following format:
```
{
  "host": "aquabyte-dev.cfwlu7jbdcqj.eu-west-1.rds.amazonaws.com",
  "port": "5432",
  "user": "aquabyte_ro",
  "password": "XXXXX",
  "database": "aquabyte_dev"
}
```

The S3 credentials must be under the following format:

```
{
  "aws_secret_access_key": "XXXXXXXXXXXXXXXXXX",
  "aws_access_key_id": "XXXXXXXXXXXXXXXXXX"
}
```

Launch the service with ```docker-compose up -d --build --force-recreate```. You can look at the logs do check if the 
service deployed correctly by running ```journalctl -f CONTAINER_NAME=cloud_data_service```


## Authors

* **Thomas Hossler** - thomas@aquabyte.ai
