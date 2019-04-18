#! /bin/bash
set -euo pipefail
. ./env/bin/activate

timestamp=$(date +%s)
mkdir -p ./experiments/debug_$timestamp/

python ./src/train.py --experiments-path ./experiments/debug_$timestamp/ \
                      --train ./data/vgg19_train/vgg19_train.npz \
                      --validation ./data/vgg19_validation/vgg19_validation.npz \
                      --pretrained-model vgg19 \
                      --extract-until 21 \
                      --freeze-until 21 \
                      --epochs 1000 \
                      --batch-size 64 \
                      --lr 0.0001 \
                      --l1 0.00001 \
                      --l2 0.00001 \
                      --dropout 0.4
