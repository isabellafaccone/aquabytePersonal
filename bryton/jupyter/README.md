docker run -it -p 8888:8888 tensorflow/tensorflow:latest 

 ```docker build -t algorithms .``` 


bryton [5:48 PM]
awesome let me try this.

thomas [5:48 PM]
 ```docker run --name=research -v /home/ubuntu/:/root/ -p 8889:8889 -ti algorithms```
(edited)
the docker run will automatically launch the notebook, just click on the link in the logs or go to localhost:8889
I am assumed you were mounting /home/ubuntu to root
but if you need to mount anything else, just change left the part after the -v flag