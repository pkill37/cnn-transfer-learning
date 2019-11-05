#! /bin/bash
set -euxo pipefail

experiments=$(echo ./experiments/experiments_1_fixed)
rm -rf $experiments && mkdir -p $experiments

for extract in 18 14 10 06 03; do
	for freeze in 18 14 10 06 03 00; do
		if [ "$extract" -lt "$freeze" ]; then
			continue
		fi

		# np.geomspace(10**-4, 10**1, 20)
		for l2 in 1.00000000e-04 1.83298071e-04 3.35981829e-04 6.15848211e-04 1.12883789e-03 2.06913808e-03 3.79269019e-03 6.95192796e-03 1.27427499e-02 2.33572147e-02 4.28133240e-02 7.84759970e-02 1.43844989e-01 2.63665090e-01 4.83293024e-01 8.85866790e-01 1.62377674e+00 2.97635144e+00 5.45559478e+00 1.00000000e+01; do
			python ./src/1_vgg16.py --experiments "$experiments" --train ./data/isic2018/224/train/train.npz --epochs 1000 --batch-size 32 --extract-until "$extract" --freeze-until "$freeze" --l2 "$l2"
		done
	done
done
