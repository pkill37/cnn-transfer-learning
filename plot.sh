#! /bin/bash
set -euo pipefail
. ./env/bin/activate

for experiment in $(find ./experiments/ -mindepth 2 -maxdepth 2 -type d); do
    python ./src/plot.py --experiment $experiment
done
