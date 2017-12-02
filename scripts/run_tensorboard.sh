#!/usr/bin/env bash

nvidia-docker run -it --rm -p 0.0.0.0:6006:6006 \
-e MXNET_CUDNN_AUTOTUNE_DEFAULT=0 \
-v /home/oper/Datasets:/mxnet/example/ssd/data \
-v /home/oper/david/mxnet-ssd/model:/mxnet/example/ssd/model \
-v /home/oper/david/mxnet-ssd/config:/mxnet/example/ssd/config \
-v /home/oper/david/mxnet-ssd/output:/mxnet/example/ssd/output \
-v /home/oper/david/mxnet-ssd/dataset:/mxnet/example/ssd/dataset \
-v /home/oper/david/mxnet-ssd/train:/mxnet/example/ssd/train \
-v /home/oper/david/mxnet-ssd/tools:/mxnet/example/ssd/tools \
-v /home/oper/david/mxnet-ssd/symbol:/mxnet/example/ssd/symbol \
-v /home/oper/david/mxnet-ssd/detect:/mxnet/example/ssd/detect \
-v /home/oper/david/mxnet-ssd/evaluate:/mxnet/example/ssd/evaluate \
-v /home/oper/david/mxnet-ssd/scripts:/mxnet/example/ssd/scripts \
-v /home/oper/david/mxnet-ssd/deploy.py:/mxnet/example/ssd/deploy.py \
-v /home/oper/david/mxnet-ssd/evaluate.py:/mxnet/example/ssd/evaluate.py \
-v /home/oper/david/mxnet-ssd/train.py:/mxnet/example/ssd/train.py \
-v /home/oper/david/mxnet-ssd/demo.py:/mxnet/example/ssd/demo.py \
mxnet/ssd:gpu_0.12.0_cuda9
