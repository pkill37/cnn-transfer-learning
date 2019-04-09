#! /bin/bash
set -euo pipefail
. ./env/bin/activate

for pretrained_model in vgg16 inceptionv3; do
	echo "Processing training data for $pretrained_model networks..."
	rm -f ./data/train_$pretrained_model.npz
	python ./src/data.py --images ../isic2017/ISIC-2017_Training_Data/ --descriptions ../isic2017/ISIC-2017_Training_Part3_GroundTruth.csv --output ./data/train_$pretrained_model.npz --pretrained-model $pretrained_model --total-samples 8000

	echo "Processing validation data for $pretrained_model networks..."
	rm -f ./data/validation_$pretrained_model.npz
	python ./src/data.py --images ../isic2017/ISIC-2017_Validation_Data/ --descriptions ../isic2017/ISIC-2017_Validation_Part3_GroundTruth.csv --output ./data/validation_$pretrained_model.npz --pretrained-model $pretrained_model

	echo "Processing test data for $pretrained_model networks..."
	rm -f ./data/test_$pretrained_model.npz
	python ./src/data.py --images ../isic2017/ISIC-2017_Test_v2_Data/ --descriptions ../isic2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv --output ./data/test_$pretrained_model.npz --pretrained-model $pretrained_model
done
