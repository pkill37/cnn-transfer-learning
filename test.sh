#! /bin/bash
set -x

experiments=$(echo "$1")


for experiment in $(find "$experiments" -mindepth 1 -maxdepth 1 -type d); do
	rm -f $experiment/stats.json
	python ./src/test.py --experiment "$experiment" --test-set ./data/isic2018/224/test/test.npz
done

rm -f $experiments/stats.json
jq -s '.' $experiments/**/stats.json > $experiments/stats.json
