#! /bin/bash
set -euxo pipefail

experiments=$(echo ./experiments_1)
rm -rf $experiments && mkdir $experiments

for extract in 18 14 10 06 03; do
	for freeze in 18 14 10 06 03 00; do
		if [ "$extract" -lt "$freeze" ]; then
			continue
		fi

		for l2 in 1e-10 2.1544346900318866e-09 4.641588833612773e-08 1e-06 2.1544346900318823e-05 0.00046415888336127724 0.01 0.21544346900318778 4.641588833612772 100.0; do
			python ./src/1_vgg16.py --experiments "$experiments" --train ./data/isic2018/224/train/train.npz --epochs 1000 --batch-size 32 --extract-until "$extract" --freeze-until "$freeze" --l2 "$l2"
		done
	done
done
