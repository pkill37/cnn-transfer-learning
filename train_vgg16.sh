#! /bin/bash
set -euxo pipefail

experiments=$(echo ./experiments_fixed/)
mkdir -p $experiments

#for extract in 18 14 10 06 03; do
	#for freeze in 18 14 10 06 03 00; do
for extract in 18; do
	for freeze in 18; do
		if [ "$extract" -lt "$freeze" ]; then
			continue
		fi

		for lambda in 1.00000000e-04 2.78255940e-04 7.74263683e-04 2.15443469e-03 5.99484250e-03 1.66810054e-02 4.64158883e-02 1.29154967e-01 3.59381366e-01 1.00000000e+00; do
			python ./src/train_vgg16.py \
				--experiments "$experiments" \
				--train ./data/isic2018/224/train/train.npz \
				--epochs 1000 \
				--batch-size 32 \
				--extract-until "$extract" \
				--freeze-until "$freeze" \
				--units 512 \
				--l2 "$lambda" \
				--dropout 0.0 \
				--patience 10
		done
	done
done
