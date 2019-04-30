#! /bin/bash
set -euo pipefail
. ./env/bin/activate

for extract in 21 16 11 6 3; do
    for freeze in 21 16 11 6 3; do
        experiment=$(echo ./experiments/vgg19_"$extract"_"$freeze"/)
        rm -rf $experiment && mkdir -p $experiment && echo $experiment

        python ./src/vgg19.py --experiment $experiment \
                              --train ./data/isic2018/224/train/train.npz \
                              --extract-until $extract \
                              --freeze-until $freeze \
                              --lr 0.0001 \
                              --l2 0.0001 \
                              --epochs 300 \
                              --bs 64

        python ./src/test.py --model $experiment/model.hdf5 \
                             --test ./data/isic2018/224/test/test.npz
    done
done
