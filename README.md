# MSc

Binary classification of skin samples from ISIC-Archive.

## Environment

```
conda env create -f environment.yml
conda activate msc
conda env create -f environment-gpu.yml
conda activate msc-gpu
```

## Data

Download the datasets from the [ISIC 2017](https://challenge.kitware.com/#challenge/n/ISIC_2017%3A_Skin_Lesion_Analysis_Towards_Melanoma_Detection) Part 3 challenge and preprocess them to obtain the final compressed datasets:

```
tmux new -d python ./src/data.py --images ~/Downloads/ISIC-2017_Training_Data/ --descriptions ~/Downloads/ISIC-2017_Training_Part3_GroundTruth.csv --dataset ./data/train.npz
tmux new -d python ./src/data.py --images ~/Downloads/ISIC-2017_Validation_Data/ --descriptions ~/Downloads/ISIC-2017_Validation_Part3_GroundTruth.csv --dataset ./data/validation.npz
tmux new -d python ./src/data.py --images ~/Downloads/ISIC-2017_Test_v2_Data/ --descriptions ~/Downloads/ISIC-2017_Test_v2_Part3_GroundTruth.csv --dataset ./data/test.npz
```

## Train

Run the experiments:

```
tmux new -d python ./src/experiments.py --experiments_path ./experiments --train-set ./data/train.npz --validation-set ./data/validation.npz
```

Monitor training:

```
tmux new -d tensorboard --logdir ./experiments/
```

## Test

```
tmux new -d python src/test.py --images_path ./data/images --descriptions_path ./data/descriptions --model ./out/experiment_21/best.hdf5
```
