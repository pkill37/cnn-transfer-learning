#! /bin/bash
set -euo pipefail

# Which script to run
script=$(echo "$1")

# Where to store results
timestamp=$(echo $(($(date +%s%N)/1000000)))
name=$(echo $2)
experiments="experiments_${timestamp}_${name}"
echo $experiments
rm -rf $experiments && mkdir -p $experiments

python $script --experiments $experiments --train-set ./data/isic2018/224/train/train.npz --epochs 500 --batch-size 64
