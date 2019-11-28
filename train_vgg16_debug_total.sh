#! /bin/bash
set -euxo pipefail

experiments=$(echo ./experiments/vgg16_debug_total)
mkdir -p $experiments


for fraction in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
	python ./src/train_vgg16.py \
		--experiments "$experiments" \
		--train ./data/isic2018/224/train/train.npz \
		--epochs 1000 \
		--batch-size 32 \
		--extract-until 18 \
		--freeze-until 18 \
		--units 512 \
		--l2 0.000879 \
		--dropout 0.0 \
		--patience 20 \
		--lr 10e-4 \
		--m-fraction "$fraction"
done
