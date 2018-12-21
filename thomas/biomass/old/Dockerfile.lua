FROM ufoym/deepo:all-py36-jupyter

RUN apt-get update && apt-get install -y emacs libpng++-dev pkg-config

COPY lib /root/bryton/aquabyte_biomass/lib

WORKDIR /root/bryton/aquabyte_biomass/lib/mc-cnn

RUN make

