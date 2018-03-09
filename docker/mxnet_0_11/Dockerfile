
FROM    mxnet/python:gpu_0.11.0

RUN     apt-get update && apt-get install -y \
        nano \
        wget \
        graphviz \
        python-tk


RUN     pip install ipython jupyter matplotlib scipy graphviz tensorboard future

WORKDIR /mxnet/example/ssd
