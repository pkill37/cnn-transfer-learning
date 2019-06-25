#! /bin/bash
set -euo pipefail

experiments=$(echo "$1")

python ./src/test.py --experiments $experiments --test-set ./data/isic2018/224/test/test.npz
