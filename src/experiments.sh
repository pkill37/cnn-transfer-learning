#! /bin/bash

set -euxo pipefail

for i in 21 17 13 9 5 2; do
    rm -rf ../out/experiment_$i/
    mkdir -p ../out/experiment_$i/
    mkdir -p ../out/experiment_$i/tensorboard/
    python ./train.py --images_path ../data/images/ --descriptions_path ../data/descriptions/ --experiment ../out/experiment_$i/ --nb_layers $i --augmentation
done
