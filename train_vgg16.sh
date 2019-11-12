#! /bin/bash
set -euxo pipefail

experiments=$(echo ./experiments/vgg16)
mkdir -p $experiments

for extract in 18 14 10 06 03; do
	for freeze in 00 10 06 14 03 18; do
		# ensure extract >= freeze
		if [ "$extract" -lt "$freeze" ]; then
			continue
		fi

		# np.geomspace(0.0001, 0.005, 10)
		for lambda in 0.0001 0.00015445 0.00023853 0.0003684 0.00056898 0.00087876 0.00135721 0.00209614 0.00323739 0.005; do
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
				--patience 20
		done
	done
done
