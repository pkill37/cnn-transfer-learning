#! /bin/bash
set -euo pipefail
. ./env/bin/activate

# Download and unzip
rm -rf ./data/isic2018/ && mkdir -p ./data/isic2018/
curl -sSL https://challenge.kitware.com/api/v1/item/5ac20fc456357d4ff856e139/download > ./data/isic2018/ISIC2018_Task3_Training_Input.zip
curl -sSL https://challenge.kitware.com/api/v1/item/5ac20eeb56357d4ff856e136/download > ./data/isic2018/ISIC2018_Task3_Training_GroundTruth.zip
unzip ./data/isic2018/ISIC2018_Task3_Training_Input.zip -d ./data/isic2018
unzip ./data/isic2018/ISIC2018_Task3_Training_GroundTruth.zip -d ./data/isic2018

# Process and compress
for target_size in 224; do
    echo "Processing data for target size "$target_size"x"$target_size"..."
    rm -rf ./data/isic2018/"$target_size"
    mkdir -p ./data/isic2018/"$target_size"/{train,test}
    python ./src/data.py --images ./data/isic2018/ISIC2018_Task3_Training_Input \
                         --descriptions ./data/isic2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv \
                         --target-size "$target_size" \
                         --target-samples 16000 \
                         --output ./data/isic2018/"$target_size"
done
