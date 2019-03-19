import argparse
import train
import itertools
import helpers
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-path', type=str)
    parser.add_argument('--train-set', type=str)
    parser.add_argument('--validation-set', type=str)
    args = parser.parse_args()

    configs = [
        { 'model': ['vgg16'], 'extract_until': [3,6,10,14,18], 'freeze_until': [3,6,10,14,18], 'epochs': [1000], 'batch_size': [32], 'lr': [0.001], },
        { 'model': ['inceptionv3'], 'extract_until': [41,64,87,101,133,165,197,229,249], 'freeze_until': [41,64,87,101,133,165,197,229,249], 'epochs': [1000], 'batch_size': [32], 'lr': [0.001], },
    ]

    for config in configs:
        # Compute cartesian product of possible parameter values
        experiments = itertools.product(*config.values())

        # Filter such that extract_until >= extract_until
        experiments = list(filter(lambda e: e[1] >= e[2], experiments))

        # Run all experiments
        for i, parameters in enumerate(experiments):
            # Unpack parameters
            pretrained_model, extract_until, freeze_until, epochs, batch_size, lr = parameters

            # Run this particular experiment
            print(f'Experiment {i} (pretrained model {pretrained_model}, extract until {extract_until}, freeze until {freeze_until}, epochs {epochs}, batch_size {batch_size}, lr {lr})')
            experiment = os.path.join(args.experiments_path, f'{pretrained_model}_{extract_until}_{freeze_until}_{epochs}_{batch_size}_{lr}')
            helpers.create_or_recreate_dir(experiment)
            train.train(experiment, args.train_set, args.validation_set, pretrained_model, extract_until, freeze_until, epochs, batch_size, lr)
