#!/usr/bin/env bash

CONFIG=$1

python -m torch.distributed.launch --nproc_per_node=4 --master_port=4322 basicsr/train_crop.py -opt $CONFIG --launcher pytorch