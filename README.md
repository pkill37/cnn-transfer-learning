# msc

Binary classification of skin samples from ISIC-Archive.

## Install

```
conda env create -f environment.yml
conda activate msc
```

## Data

To download the entire dataset (+23000 samples, +50GB):

```
python ISIC-Archive-Downloader/download_archive.py --images-dir /Volumes/data/images --descs-dir /Volumes/data/descriptions
```

For development it may be wise to consider only 150 samples of each class (malignant, benign):

```
python ISIC-Archive-Downloader/download_archive.py --images-dir /Volumes/data/images --descs-dir /Volumes/data/descriptions --num-images 150 --filter malignant
python ISIC-Archive-Downloader/download_archive.py --images-dir /Volumes/data/images --descs-dir /Volumes/data/descriptions --num-images 150 --filter benign --offset 150
```

## Train

Train the model:

```
mkdir -p ./out/models/ ./out/tensorboard
python src/train.py --images_path /Volumes/data/images --descriptions_path /Volumes/data/descriptions --augmentation --nb_layers 21
```

Monitor training:

```
tensorboard --logdir $(pwd)/out/experiment_21/tensorboard
```

## Evaluation

Evaluate a model:

```
python src/test.py --images_path /Volumes/data/images --descriptions_path /Volumes/data/descriptions --model $(pwd)/out/experiment_21/best.hdf5
```
