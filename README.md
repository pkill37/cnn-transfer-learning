# MSc

The MSc thesis by Fábio Maia who developed a comprehensive study and comparison of transfer learning techniques applied to binary classification of skin lesions.

## Environment

We use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage a virtual Python 3.6 environment and dependencies.

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

Run the prepared scripts to train the VGG16 transfer learning models and the custom CNN end-to-end learning models:

```
./train_vgg16.sh
./train_custom1.sh
./train_custom2.sh
```

## Test

Run the prepared scripts to test the VGG16 transfer learning models and the custom CNN end-to-end learning models:

```
./test_vgg16.sh
./test_custom.sh
```

Then plot results accordingly:

```
./plot_vgg16.sh
./plot_custom.sh
```

## Dissertation

The LaTeX document in `doc/` is the dissertation for the MSc thesis which tells the story of the experiments.

```
cd doc
make
```
