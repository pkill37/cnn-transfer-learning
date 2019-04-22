#! /bin/bash
set -euo pipefail
. ./env/bin/activate
echo "Please make sure to download the ISIC 2017 data from https://challenge.kitware.com/#challenge/n/ISIC_2017%3A_Skin_Lesion_Analysis_Towards_Melanoma_Detection to ./data/isic2017/"

for pretrained_model in vgg19 inceptionv3; do
    echo "Processing train data for $pretrained_model networks..."
    rm -rf ./data/"$pretrained_model"_train/ && mkdir -p ./data/"$pretrained_model"_train
    python ./src/data.py --images ./data/isic2017/ISIC-2017_Training_Data/ \
                         --descriptions ./data/isic2017/ISIC-2017_Training_Part3_GroundTruth.csv \
                         --output ./data/"$pretrained_model"_train \
                         --pretrained-model $pretrained_model \
                         --total-samples 8000

    echo "Processing validation data for $pretrained_model networks..."
    rm -rf ./data/"$pretrained_model"_validation/ && mkdir -p ./data/"$pretrained_model"_validation
    python ./src/data.py --images ./data/isic2017/ISIC-2017_Validation_Data/ \
                         --descriptions ./data/isic2017/ISIC-2017_Validation_Part3_GroundTruth.csv \
                         --output ./data/"$pretrained_model"_validation \
                         --pretrained-model $pretrained_model

    echo "Processing test data for $pretrained_model networks..."
    rm -rf ./data/"$pretrained_model"_test/ && mkdir -p ./data/"$pretrained_model"_test
    python ./src/data.py --images ./data/isic2017/ISIC-2017_Test_v2_Data/ \
                         --descriptions ./data/isic2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv \
                         --output ./data/"$pretrained_model"_test \
                         --pretrained-model $pretrained_model
done
