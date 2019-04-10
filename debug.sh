#! /bin/bash
set -euo pipefail
. ./env/bin/activate

timestamp=$(date +%s)
mkdir -p ./experiments/debug_$timestamp/

python ./src/train.py --experiments-path ./experiments/debug_$timestamp/ \
                      --train ./data/train_vgg19/train_vgg19.npz \
                      --validation ./data/validation_vgg19/validation_vgg19.npz \
                      --pretrained-model vgg19 \
                      --extract-until 21 \
                      --freeze-until 21 \
                      --epochs 1000 \
                      --batch-size 32 \
                      --lr 0.001 \
                      --dropout 0.4
