#! /bin/bash
set -euo pipefail
. ./env/bin/activate

timestamp=$(date +%s)
mkdir -p ./experiments/custom/$timestamp/

python ./src/custom.py --experiment ./experiments/custom/$timestamp/ \
                       --train ./data/isic2018/224/train/train.npz \
                       --epochs 3
