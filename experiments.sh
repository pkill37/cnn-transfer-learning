#! /bin/bash
set -euo pipefail
. ./env/bin/activate

python ./src/experiments.py --experiments-path ./experiments/ \
                            --train-set ./data/train.npz \
                            --validation-set ./data/validation.npz
