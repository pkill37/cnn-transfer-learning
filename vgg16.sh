#! /bin/bash
set -euo pipefail
. ./env/bin/activate

vgg16=$(echo ./experiments/vgg16)
rm -rf $vgg16

for extract in 18 14 10 6 3; do
    for freeze in 18 14 10 6 3 0; do
        if [ "$extract" -lt "$freeze" ]; then
            continue
        fi

        experiment=$(echo $vgg16/extract"$extract"_freeze"$freeze")
        mkdir -p $experiment && echo $experiment

        python ./src/vgg16.py --experiment $experiment \
                              --train ./data/isic2018/224/train/train.npz \
                              --extract-until $extract \
                              --freeze-until $freeze \
                              --epochs 500 \
                              --bs 64

        python ./src/test.py --model $experiment/model.h5 \
                             --test ./data/isic2018/224/test/test.npz
    done
done
