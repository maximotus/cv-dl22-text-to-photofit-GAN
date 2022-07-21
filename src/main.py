import os
import shutil
import sys
import yaml

from datetime import datetime

from misc.error import ConfigurationError
from misc.log import setup_logger
from training import Trainer, Evaluator, PhotofitGenerator

MODE_KEY = 'mode'
MODES = ['train', 'eval', 'gen']
MODEL_KEY = 'model'
MODEL_NAME_KEY = 'name'
MODEL_PARAMETERS_KEY = 'parameters'
DATASET_KEY = 'dataset'
EPOCHS_KEY = 'epochs'
BATCH_SIZE_KEY = 'batch_size'
OPTIMIZER_KEY = 'optimizer'
LEARNING_RATE_KEY = 'learning_rate'
CRITERION_KEY = 'criterion'
IMAGE_SIZE_KEY = 'image_size'
DEVICE_KEY = 'device'
EXPERIMENT_PATH_KEY = 'experiment_path'
FREQUENCIES_KEY = 'frequencies'
SAVE_FREQUENCY_KEY = 'save_freq'
GEN_FREQUENCY_KEY = 'gen_freq'


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


def create_experiment_dir(conf_file, exp_path):
    config_name = conf_file[10:-5]
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    base_path = os.path.join(exp_path, config_name, now)

    paths = [os.path.join(base_path, 'stats'),
             os.path.join(base_path, 'model'),
             os.path.join(base_path, 'results')]

    for path in paths:
        os.makedirs(path)
        print('Created directory', path)

    shutil.copy2(conf_file, base_path)
    print('Copied config file', conf_file, 'to', base_path)

    return base_path


if __name__ == '__main__':
    configuration_file, configuration = parse_config(sys.argv)
    mode = configuration.get(MODE_KEY)
    experiment_path = create_experiment_dir(configuration_file, configuration.get(EXPERIMENT_PATH_KEY))
    logger = setup_logger(experiment_path)
    logger.info('Successfully read the given configuration file, created experiment directories and set up logger.')

    try:
        if mode not in MODES:
            msg = 'mode has to be one of' + str(MODES)
            raise ConfigurationError(msg)
        if mode == MODES[0]:
            trainer = Trainer(configuration.get(MODEL_KEY).get(MODEL_NAME_KEY),
                              configuration.get(MODEL_KEY).get(MODEL_PARAMETERS_KEY),
                              configuration.get(DATASET_KEY),
                              configuration.get(EPOCHS_KEY),
                              configuration.get(BATCH_SIZE_KEY),
                              configuration.get(OPTIMIZER_KEY),
                              configuration.get(LEARNING_RATE_KEY),
                              configuration.get(CRITERION_KEY),
                              configuration.get(DEVICE_KEY),
                              configuration.get(FREQUENCIES_KEY).get(SAVE_FREQUENCY_KEY),
                              configuration.get(FREQUENCIES_KEY).get(GEN_FREQUENCY_KEY),
                              configuration.get(IMAGE_SIZE_KEY),
                              experiment_path)
            trainer.train()
        if mode == MODES[1]:
            evaluator = Evaluator()
            evaluator.evaluate()
        if mode == MODES[2]:
            generator = PhotofitGenerator()
            generator.generate()
    # except AttributeError as e:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     logger.error('Could not interpret configuration file properly or some other error occurred: ' + str(exc_type) + ' ' + str(file_name) +
    #                  ' ' + str(exc_tb.tb_lineno) + ' ' + str(e))
    except (ConfigurationError, NotImplementedError) as e:
        logger.exception(e)
