#!/usr/bin/env bash

# this is a training script
# defining global parameters
TRAIN_REC_PATH=./data/train.rec
VAL_REC_PATH=./data/val.rec
NETWORK=vgg16_reduced
BATCH_SIZE=32
DATA_SHAPE=300

python ./scripts/train.py \
    --train-path ${TRAIN_REC_PATH} \
    --val-path= ${VAL_REC_PATH} \
    --network ${NETWORK} \
    --batch-size ${BATCH_SIZE} \
    --data-shape ${DATA_SHAPE}

