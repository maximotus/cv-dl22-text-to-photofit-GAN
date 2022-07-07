import os
import shutil
import sys
import yaml

from datetime import datetime
from src.log import setup_logger
from src.training.training import train

LEARNING_RATE_KEY = 'learning_rate'
BATCH_SIZE_KEY = 'batch_size'
EPOCHS_KEY = 'epochs'
OPTIMIZER_KEY = 'optimizer'
NOISE_SIZE_KEY = 'noise_size'
IMAGE_SIZE_KEY = 'image_size'
DATASET_KEY = 'dataset'
MODEL_KEY = 'model'
DEVICE_KEY = 'device'
EXPERIMENT_PATH_KEY = 'experiment_path'
FREQUENCIES_KEY = 'frequencies'


def parse_config(argv):
    try:
        config_path = argv[1]
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
    except IndexError:
        print('Missing command line argument for the path to the configuration file. '
              'Please use this program like this: main.py PATH_TO_CONFIG_YAML_FILE')
        sys.exit()

    return config_path, config


def create_experiment_dir(file):
    config_name = file[10:-5]
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    base_path = os.path.join(config_dict.get(EXPERIMENT_PATH_KEY), config_name, now)

    paths = [os.path.join(base_path, 'stats'), os.path.join(base_path, 'model')]
    for path in paths:
        os.makedirs(path)
        print('Created directory', path)

    shutil.copy2(file, base_path)
    print('Copied config file', file, 'to', base_path)

    return base_path


if __name__ == '__main__':
    config_file, config_dict = parse_config(sys.argv)
    experiment_path = create_experiment_dir(config_file)
    logger = setup_logger(experiment_path)
    logger.info('Successfully read the given configuration file, created experiment directories and set up logger.')

    train()
