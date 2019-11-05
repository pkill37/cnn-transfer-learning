#! /bin/bash
set -euxo pipefail

experiments=$(echo ./experiments/experiments_2_fixed)
rm -rf $experiments && mkdir -p $experiments

# np.geomspace(10**-4, 10**1, 30)
for l2 in 1.00000000e-04 1.48735211e-04 2.21221629e-04 3.29034456e-04 4.89390092e-04 7.27895384e-04 1.08263673e-03 1.61026203e-03 2.39502662e-03 3.56224789e-03 5.29831691e-03 7.88046282e-03 1.17210230e-02 1.74332882e-02 2.59294380e-02 3.85662042e-02 5.73615251e-02 8.53167852e-02 1.26896100e-01 1.88739182e-01 2.80721620e-01 4.17531894e-01 6.21016942e-01 9.23670857e-01 1.37382380e+00 2.04335972e+00 3.03919538e+00 4.52035366e+00 6.72335754e+00 1.00000000e+01; do
	for units in 32 64 128 256 512 1024; do
		python ./src/2_custom1.py --experiments "$experiments" --train ./data/isic2018/224/train/train.npz --epochs 1000 --batch-size 32 --l2 "$l2" --units "$units"
	done
done
