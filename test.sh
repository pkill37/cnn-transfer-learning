#! /bin/bash
set -euxo pipefail

experiments=$(echo "$1")
test_set=$(echo "$2")


for experiment in $(find "$experiments" -mindepth 1 -maxdepth 1 -type d); do
	rm -f $experiment/stats.json
	python ./src/test.py --experiment "$experiment" --test-set "$test_set"
done

rm -f $experiments/stats.json
jq -s '.' $experiments/**/stats.json > $experiments/stats.json
