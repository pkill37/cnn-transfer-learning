#! /bin/bash

set -euxo pipefail

for i in 21 17 13 9 5 2; do
    rm -rf $(pwd)/out/experiment_$i/
    mkdir -p $(pwd)/out/experiment_$i/tensorboard
    python src/train.py --images_path /Volumes/data/images/ --descriptions_path /Volumes/data/descriptions/ --experiment $(pwd)/out/experiment_$i/ --nb_layers $i --augmentation
done
