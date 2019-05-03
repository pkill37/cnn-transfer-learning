#! /bin/bash
set -euo pipefail
. ./env/bin/activate

timestamp=$(date +%s)
mkdir -p ./experiments/custom_$timestamp/

python ./src/custom.py --experiments ./experiments/custom_$timestamp/ \
                       --train ./data/isic2018/224/train/train.npz \
                       --epochs 5 \
                       --bs 64
