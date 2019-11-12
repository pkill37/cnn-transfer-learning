#! /bin/bash
set -euo pipefail

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda.sh
bash miniconda.sh
rm miniconda.sh

exec $SHELL
conda env create -f src/environment.yml
