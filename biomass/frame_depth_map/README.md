To port forward to your local machine:
ssh -NfL localhost:9899:localhost:9899 ubuntu@research.aquabyte.ai -p 22

To start the jupyter process
Old:

cd /root/ && jupyter notebook --port=9898 --allow-root --NotebookApp.token=

screen

jupyter notebook --no-browser --port=9899 --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/root'