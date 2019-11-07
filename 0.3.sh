#! /bin/bash
set -euxo pipefail

experiments=$(echo ./experiments_story/0.3)
mkdir -p $experiments

for dropout in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
	python ./src/0.py \
		--experiments "$experiments" \
		--train ./data/isic2018/224/train/train.npz \
		--epochs 1000 \
		--batch-size 32 \
		--extract-until 18 \
		--freeze-until 14 \
		--units 512 \
		--l2 0.3 \
		--dropout "$dropout"
done
