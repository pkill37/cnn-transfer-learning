#! /bin/bash

set -euo pipefail

. ./env/bin/activate

timestamp=$(date +%s)
mkdir -p ./experiments/test_$timestamp/

python ./src/train.py --experiments-path ./experiments/test_$timestamp/ \
                      --train ./data/train_vgg16.npz \
                      --validation ./data/validation_vgg16.npz \
                      --pretrained-model vgg16 \
                      --extract-until 18 \
                      --freeze-until 18 \
                      --epochs 1000 \
                      --batch-size 32 \
                      --lr 0.01 \
                      --l1 0.01 \
                      --l2 0.01
