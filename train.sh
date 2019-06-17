#! /bin/bash
set -euo pipefail

# Which script to run
script=$(echo "$1")

# Where to store results
experiments=$(echo "$2")
rm -rf $experiments && mkdir -p $experiments

python $script --experiments $experiments --train-set ./data/isic2018/224/train/train.npz --epochs 500 --batch-size 64
