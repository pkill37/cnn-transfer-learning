.PHONY: all data experiments test cleanall

SRC         := ${PWD}/src
OUT         := ${PWD}/out
DATA        := ${PWD}/data

all: train

data:
	tmux new -d python ISIC-Archive-Downloader/download_archive.py --images-dir ${DATA}/images --descs-dir ${DATA}/descriptions

experiments:
	tmux new -d bash experiments.sh
	tmux new -d tensorboard --logdir ${OUT}/

test:
	tmux new -d python ${SRC}/test.py

cleanall:
	rm -f ${OUT}/
