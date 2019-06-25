#! /bin/bash
set -euo pipefail

# Which script to run
script=$(echo "$1")

# Where to store results
name=$(basename "$1" .py)
experiments="experiments_${name}"
rm -rf $experiments && mkdir -p $experiments

python $script --experiments $experiments --train-set ./data/isic2018/224/train/train.npz --epochs 1000 --batch-size 64
