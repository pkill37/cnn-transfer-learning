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

Given a script `./src/foo.py` that trains a model, you can run it and store results in `./foo` using

```
./train.sh ./src/foo.py
```

## Test

Given a directory of results `./foo` you can test the model against the test set using

```
./test.sh ./foo
```

## Dissertation

The LaTeX document in `doc/` is the dissertation for the MSc thesis which tells the story of the experiments.

```
cd doc
make
```
