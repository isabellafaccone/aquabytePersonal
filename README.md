# Aquabyte CV research

This repository is used to train various models for the aquabyte-ml pipeline. As of now, Aquabyte trainer can train a 
fish detection model using Retinanet and a fish segmentation model using Mask-RCNN. 

## Getting Started


### Prerequisites

Docker and docker-compose needs to be installed. The docker daemon runtimes (/etc/docker/daemon.json) must be set up so 
the container will use GPUs.


### Installing

Start by modifying the deploy/docker-compose.yml file.
```
...
services:
  aquabyte-trainer: # change this name to the name you want to give to your container.
...
volumes:
  - /home/ubuntu/thomas:/root/thomas/ # change the mount to your home directory
ports:
  - "1234:1234" # change the port forwarding rule to the port wanted
 ```

Then deploy the container by running in the deploy folder:

```
docker-compose up --build -d --force-recreate
```

Once the container is deployed, log into it by running the line below, where container name has been defined above:

```
docker exec -ti {CONTAINER_NAME} bash
```

And in the container, where the port has been defined in the docker-compose file:

```
jupyter notebook --port={PORT} --allow-root --NotebookApp.token=
```

## Using

On your local machine, in the browser, enter:
```
localhost:PORT
```

## Authors

* **Thomas Hossler** - thomas@aquabyte.ai

