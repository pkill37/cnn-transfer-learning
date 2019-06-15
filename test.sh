#! /bin/bash
set -euo pipefail
. ./env/bin/activate

experiments=$(echo "$1")
mkdir -p $experiments

for experiment in $(find $experiments -mindepth 1 -maxdepth 1 -type d); do
    python ./src/test.py --model $experiment/model.h5 --test-set ./data/isic2018/224/test/test.npz
    python ./src/plot.py --experiment $experiment
done
