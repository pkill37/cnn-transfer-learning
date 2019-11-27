#! /bin/bash
set -euxo pipefail

experiments=$(echo "$1")
python ./src/plot_custom.py --experiments "$experiments"
