#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target DIR/../data/train.lst
python prepare_dataset.py --dataset pascal --year 2007 --set test --target $DIR/../data/val.lst --shuffle False
