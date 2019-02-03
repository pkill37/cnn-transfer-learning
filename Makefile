.PHONY: all data train test cleanall

SRC         := /tmp/src
OUT         := /tmp/out
DATA        := /tmp/ISIC-Archive-Downloader
PYTHON      := docker run -it              -v ${PWD}:/tmp -w /tmp tensorflow/tensorflow:latest-py3 python
TENSORBOARD := docker run -it -p 6006:6006 -v ${PWD}:/tmp -w /tmp tensorflow/tensorflow:latest-py3 tensorboard

all: train

data:
	rm -rf /Volumes/data/images
	rm -rf /Volumes/data/descriptions

	${PYTHON} ${DATA}/download_archive.py --num-images 1000 --filter benign    --images-dir /Volumes/data/images --descs-dir /Volumes/data/descriptions
	${PYTHON} ${DATA}/download_archive.py --num-images 1000 --filter malignant --images-dir /Volumes/data/images --descs-dir /Volumes/data/descriptions --offset 1000

train:
	tmux split-window -v ${TENSORBOARD} --logdir=${OUT}/tensorboard
	tmux split-window -h htop
	${PYTHON} ${SRC}/train.py

test:
	${PYTHON} ${SRC}/test.py

cleanall:
	rm -f ${OUT}/tensorboard/*
