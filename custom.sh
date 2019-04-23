#! /bin/bash
set -euo pipefail
. ./env/bin/activate

timestamp=$(date +%s)
mkdir -p ./experiments/debug_$timestamp/

python ./src/custom.py --experiments ./experiments/debug_$timestamp/ \
                       --train ./data/vgg19_train/vgg19_train.npz \
                       --validation ./data/vgg19_validation/vgg19_validation.npz \
                       --epochs 5 \
                       --bs 32
