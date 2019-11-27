#! /bin/bash
set -euxo pipefail

experiments=$(echo ./experiments/vgg16_debug_partial)
mkdir -p $experiments


# np.geomspace(10**0, 10**-10, 20)
for lr in 1.00000000e+00 2.97635144e-01 8.85866790e-02 2.63665090e-02 7.84759970e-03 2.33572147e-03 6.95192796e-04 2.06913808e-04 6.15848211e-05 1.83298071e-05 5.45559478e-06 1.62377674e-06 4.83293024e-07 1.43844989e-07 4.28133240e-08 1.27427499e-08 3.79269019e-09 1.12883789e-09 3.35981829e-10 1.00000000e-10; do
	python ./src/train_vgg16.py \
		--experiments "$experiments" \
		--train ./data/isic2018/224/train/train.npz \
		--epochs 1000 \
		--batch-size 32 \
		--extract-until 14 \
		--freeze-until 14 \
		--units 512 \
		--l2 0.000879 \
		--dropout 0.0 \
		--patience 20 \
		--lr "$lr" \
		--m-fraction 1.0
done
