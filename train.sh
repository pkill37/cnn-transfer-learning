#! /bin/bash

# Which script to run
script=$(echo "$1")

# Where to store results
name=$(basename "$1" .py)
experiments="experiments_${name}"
rm -rf $experiments && mkdir -p $experiments

# Stop when the script ran successfully (i.e. without running out of memory or GPUs)
while true; do
	python $script --experiments $experiments --train-set ./data/isic2018/224/train/train.npz --epochs 1000 --batch-size 8

	if [ $? -eq 0 ]; then
		exit
	fi
done
