# MSc

Comprehensive study of transfer learning applied to binary classification of skin lesions.

## Environment

```
python -m venv env
. ./env/bin/activate
pip install -r requirements.txt
```

## Data

```
./isic2018.sh
```

## Train

```
./train.sh $1 $2
```

## Test

```
./test.sh ./experiments_vgg16_base/
```

## Thesis

```
cd doc
make
```
