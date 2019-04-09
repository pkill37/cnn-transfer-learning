# MSc

Comprehensive study of transfer learning applied to binary classification of skin lesions.

## Environment

```
python -m venv env
. ./env/bin/activate
pip install -r requirements.txt
```

## Data

Download the datasets from the [ISIC 2017](https://challenge.kitware.com/#challenge/n/ISIC_2017%3A_Skin_Lesion_Analysis_Towards_Melanoma_Detection) Part 3 challenge and preprocess them to obtain the final compressed datasets:

```
./data.sh
```

## Train

Run the experiments:

```
./experiments.sh
```

Monitor training:

```
./monitor.sh
```
