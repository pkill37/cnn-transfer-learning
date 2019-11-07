#! /bin/bash
set -euxo pipefail

experiments=$(echo "$1")
python ./src/plot_vgg16.py --experiments "$experiments"
