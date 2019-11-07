#! /bin/bash
set -euxo pipefail

experiments=$(echo ./experiments_story/0.2)
mkdir -p $experiments

for dropout in 0.0; do
	python ./src/0.py \
		--experiments "$experiments" \
		--train ./data/isic2018/224/train/train.npz \
		--epochs 1000 \
		--batch-size 32 \
		--extract-until 18 \
		--freeze-until 18 \
		--units 512 \
		--l2 0.3 \
		--dropout "$dropout"
done
