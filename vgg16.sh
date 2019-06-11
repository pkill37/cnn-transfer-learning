#! /bin/bash
set -euo pipefail
. ./env/bin/activate

experiment=$(echo ./experiments/vgg16)
rm -rf $experiment && mkdir -p $experiment

python ./src/vgg16.py --experiment $experiment \
                      --train ./data/isic2018/224/train/train.npz \
                      --epochs 500 \
                      --batch-size 64
