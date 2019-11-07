#! /bin/bash
set -euxo pipefail

experiments=$(echo ./experiments_story/1)
mkdir -p $experiments

for units in 512 1024 2048 4096; do
	python ./src/0.py \
		--experiments "$experiments" \
		--train ./data/isic2018/224/train/train.npz \
		--epochs 1000 \
		--batch-size 32 \
		--extract-until 18 \
		--freeze-until 18 \
		--units "$units" \
		--l2 0.3 \
		--dropout 0.5
done
