#! /bin/bash
set -euxo pipefail
#. ./env/bin/activate

# Download and unzip
rm -rf ./data/isic2018/ && mkdir -p ./data/isic2018/
curl -sSL https://challenge.kitware.com/api/v1/item/5ac20fc456357d4ff856e139/download > ./data/isic2018/ISIC2018_Task3_Training_Input.zip
curl -sSL https://challenge.kitware.com/api/v1/item/5ac20eeb56357d4ff856e136/download > ./data/isic2018/ISIC2018_Task3_Training_GroundTruth.zip
unzip ./data/isic2018/ISIC2018_Task3_Training_Input.zip -d ./data/isic2018
unzip ./data/isic2018/ISIC2018_Task3_Training_GroundTruth.zip -d ./data/isic2018

# Process and compress
for pretrained_model in vgg19; do
    echo "Processing data for $pretrained_model networks..."
    rm -rf ./data/isic2018/"$pretrained_model"
    mkdir -p ./data/isic2018/"$pretrained_model"/{train,test}
    python ./src/data.py --images ./data/isic2018/ISIC2018_Task3_Training_Input \
                         --descriptions ./data/isic2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv \
                         --output ./data/isic2018/"$pretrained_model" \
                         --pretrained-model $pretrained_model
done
