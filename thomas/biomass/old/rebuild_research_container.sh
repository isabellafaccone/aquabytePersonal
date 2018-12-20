#docker run --name bryton_research -p 9898:9898 -v /aquabyte_data/:/aquabyte_data/ -v /data/:/root/data/ -v /home/ubuntu/bryton/:/root/bryton/ -ti gcr.io/tensorflow/tensorflow:latest-gpu bash
#docker run --name bryton_research_wnet -p 9899:9899 -v /aquabyte_data/:/aquabyte_data/ -v /data/:/root/data/ -v /home/ubuntu/bryton/:/root/bryton/ -ti bryton_biomass bash

docker stop bryton_research_wnet

docker rm bryton_research_wnet

echo "jupyter notebook --no-browser --port=9899 --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/root'"

docker run --name bryton_research_wnet -p 9899:9899 -v /aquabyte_data/:/aquabyte_data/ -v /data/:/root/data/ -v /home/ubuntu/bryton/:/root/bryton/ -v /home/ubuntu/thomas/:/root/thomas/ -v /home/ubuntu/alok/:/root/alok/ -ti bryton_biomass bash

docker ps | grep bryton
