# MSc

The MSc thesis by Fábio Maia who developed

- a simple, pragmatic, Unix-y workflow for running experiments and storing their workflows
- a comprehensive study of transfer learning applied to binary classification of skin lesions

## Environment

We use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage a virtual Python 3.6 environment and dependencies (without the overhead of containerization tools like Docker or Podman).

Install Miniconda and create the virtual Python environment with all necessary dependencies with:

```
./conda.sh
```

Activate and enter the environment at any point with:

```
conda activate msc
```

## Data

Download the official ISIC2018 train set, preprocess it, and split it into our own internal train and test sets:

```
./isic2018.sh
```

## Train

Given a script `./src/5_resnet.py` that trains a model, you can run it and store the model in `./experiments_5_resnet/model.h5` using

```
python ./src/5_resnet.py --experiments ./experiments_5_resnet --train-set ./data/isic2018/224/train/train.npz --epochs 1000 --batch-size 32
```

## Test

Given a model residing in directory `./experiments_5_resnet` you can test the model against a test set and store results in `./experiments_5_resnet` using

```
python ./src/test.py --experiments ./experiments_5_resnet --test-set ./data/isic2018/224/test/test.npz
```

And then plot results using

```
python ./src/plot.py --experiments ./experiments_5_resnet --plots ./plots_5_resnet
```

## Dissertation

The LaTeX document in `doc/` is the dissertation for the MSc thesis which tells the story of the experiments.

```
cd doc
make
```
