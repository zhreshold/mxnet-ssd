#!/usr/bin/env bash

nvidia-docker run -it --rm  -p 0.0.0.0:6006:6006 \
-e MXNET_CUDNN_AUTOTUNE_DEFAULT=0 \
-v /home/oper/Datasets:/root/mxnet-ssd/data \
-v /home/oper/david/mxnet-ssd/model:/root/mxnet-ssd/model \
-v /home/oper/david/mxnet-ssd/config:/root/mxnet-ssd/config \
-v /home/oper/david/mxnet-ssd/output:/root/mxnet-ssd/output \
-v /home/oper/david/mxnet-ssd/dataset:/root/mxnet-ssd/dataset \
-v /home/oper/david/mxnet-ssd/train:/root/mxnet-ssd/train \
-v /home/oper/david/mxnet-ssd/tools:/root/mxnet-ssd/tools \
-v /home/oper/david/mxnet-ssd/symbol:/root/mxnet-ssd/symbol \
-v /home/oper/david/mxnet-ssd/detect:/root/mxnet-ssd/detect \
-v /home/oper/david/mxnet-ssd/evaluate:/root/mxnet-ssd/evaluate \
-v /home/oper/david/mxnet-ssd/scripts:/root/mxnet-ssd/scripts \
-v /home/oper/david/mxnet-ssd/deploy.py:/root/mxnet-ssd/deploy.py \
-v /home/oper/david/mxnet-ssd/evaluate.py:/root/mxnet-ssd/evaluate.py \
-v /home/oper/david/mxnet-ssd/train.py:/root/mxnet-ssd/train.py \
-v /home/oper/david/mxnet-ssd/demo.py:/root/mxnet-ssd/demo.py \
mxnet/ssd:gpu_0.12.0_cuda9
