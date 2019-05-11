#! /bin/bash
set -euo pipefail
. ./env/bin/activate

vgg19=$(echo ./experiments/vgg19)
rm -rf $vgg19

for extract in 21 16 11 6 3; do
    for freeze in 21 16 11 6 3 0; do
        if [ "$extract" -lt "$freeze" ]; then
            continue
        fi

        experiment=$(echo $vgg19/extract"$extract"_freeze"$freeze")
        mkdir -p $experiment && echo $experiment

        python ./src/vgg19.py --experiment $experiment \
                              --train ./data/isic2018/224/train/train.npz \
                              --extract-until $extract \
                              --freeze-until $freeze \
                              --epochs 500 \
                              --bs 64

        python ./src/test.py --model $experiment/model.h5 \
                             --test ./data/isic2018/224/test/test.npz
    done
done
