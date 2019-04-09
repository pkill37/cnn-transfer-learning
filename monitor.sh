#! /bin/bash
set -euo pipefail
. ./env/bin/activate

tensorboard --logdir=./experiments --port=1337
