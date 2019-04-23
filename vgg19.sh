#! /bin/bash
set -euo pipefail
. ./env/bin/activate

timestamp=$(date +%s)
mkdir -p ./experiments/vgg19_$timestamp/

python ./src/train.py --experiments-path ./experiments/vgg19_$timestamp/ \
                      --train ./data/isic2018/vgg19/train/train.npz \
                      --pretrained-model vgg19 \
                      --extract-until 21 \
                      --freeze-until 21 \
                      --epochs 1000 \
                      --batch-size 64 \
                      --lr 0.0001 \
                      --l1 0.00001 \
                      --l2 0.00001
