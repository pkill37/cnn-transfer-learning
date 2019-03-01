# msc

Binary classification of skin samples from ISIC-Archive.

## Environment

```
conda env create -f environment.yml
conda activate msc
conda env create -f environment-gpu.yml
conda activate msc-gpu
```

## Data

To download the entire dataset (+23000 samples, +50GB):

```
tmux new -d python ISIC-Archive-Downloader/download_archive.py --images-dir ~/data/images --descs-dir ~/data/descriptions
```

## Train

Train the model:

```
tmux new -d python src/train.py --images_path ~/data/images --descriptions_path ~/data/descriptions --augmentation --nb_layers 21
```

Monitor training:

```
tmux new -d tensorboard --logdir ./out/
```

## Evaluate

```
tmux new -d python src/test.py --images_path ~/data/images --descriptions_path ~/data/descriptions --model ./out/experiment_21/best.hdf5
```
