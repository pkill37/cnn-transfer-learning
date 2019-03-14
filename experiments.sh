#! /bin/bash

set -euxo pipefail

pretrained_models=("inceptionv3" "vgg16" "resnet50")
image_sizes=(299 224 224)

for ((i=0;i<${#pretrained_models[@]};++i)); do
    rm -rf ./out/experiment_${pretrained_models[i]}/ && mkdir -p ./out/experiment_${pretrained_models[i]}/ && mkdir -p ./out/experiment_${pretrained_models[i]}/tensorboard/
    python ./src/train.py --experiment ./out/${pretrained_models[i]}_0_100_32_001/ \
                          --images-path ./data/images/ \
                          --descriptions-path ./data/descriptions/ \
                          --pretrained-model ${pretrained_models[i]} \
                          --extract-until 0 \
                          --freeze-until 0 \
                          --epochs 100 \
                          --batch-size 32 \
                          --lr 0.01
done
