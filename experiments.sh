#! /bin/bash

set -euxo pipefail

rm -rf ./out/experiment_inceptionv3/ && mkdir -p ./out/experiment_inceptionv3/ && mkdir -p ./out/experiment_inceptionv3/tensorboard/
python ./src/train.py --experiment ./out/experiment_/ --images-path ./data/images/ --descriptions-path ./data/descriptions/ --img-height 299 --img-width 299 --model inceptionv3 --nb-layers 0 --epochs 100 --batch-size 32 --lr 0.01

rm -rf ./out/experiment_vgg16/ && mkdir -p ./out/experiment_vgg16/ && mkdir -p ./out/experiment_vgg16/tensorboard/
python ./src/train.py --experiment ./out/experiment_/ --images-path ./data/images/ --descriptions-path ./data/descriptions/ --img-height 224 --img-width 224 --model vgg16 --nb-layers 0 --epochs 100 --batch-size 32 --lr 0.01

rm -rf ./out/experiment_resnet50/ && mkdir -p ./out/experiment_resnet50/ && mkdir -p ./out/experiment_resnet50/tensorboard/
python ./src/train.py --experiment ./out/experiment_/ --images-path ./data/images/ --descriptions-path ./data/descriptions/ --img-height 224 --img-width 224 --model resnet50 --nb-layers 0 --epochs 100 --batch-size 32 --lr 0.01
