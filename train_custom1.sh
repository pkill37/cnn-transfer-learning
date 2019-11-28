#! /bin/bash
set -euxo pipefail

experiments=$(echo ./experiments/custom1)
mkdir -p $experiments


# np.geomspace(0.0001, 0.005, 10)
for lambda in 0.0001 0.00015445 0.00023853 0.0003684 0.00056898 0.00087876 0.00135721 0.00209614 0.00323739 0.005; do
	python ./src/train_custom1.py \
		--experiments "$experiments" \
		--train ./data/isic2018/224/train/train.npz \
		--epochs 1000 \
		--batch-size 32 \
		--l2 "$lambda" \
		--unitsA 512 \
		--unitsB 512 \
		--dropoutA 0.0 \
		--dropoutB 0.0 \
		--patience 20
done
