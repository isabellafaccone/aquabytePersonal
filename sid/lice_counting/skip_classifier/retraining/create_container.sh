cd ~/bryton/repos/research-library
python3 setup.py sdist
cd -
cp ~/bryton/repos/research-library/dist/research-0.1.1.tar.gz .
docker build . -t skip-classifier-retraining:v1
