#! /bin/bash

set -euxo pipefail

for model in vgg16 inceptionv3 resnet152; do
    rm -rf ./out/experiment_$model/ && mkdir -p ./out/experiment_$model/ && mkdir -p ./out/experiment_$model/tensorboard/
    python ./src/train.py --experiment ./out/experiment_/ \
                          --images-path ./data/images/ \
                          --descriptions-path ./data/descriptions/ \
                          --img-height 224 \
                          --img-width 224 \
                          --model $model \
                          --nb-layers 0 \
                          --epochs 100 \
                          --batch-size 32 \
                          --lr 0.01
done
