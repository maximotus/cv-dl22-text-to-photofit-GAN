import os
import shutil
import sys
import yaml
import misc.config as config

from datetime import datetime
from misc.error import ConfigurationError
from misc.log import setup_logger
from training import Trainer, Evaluator, PhotofitGenerator


def parse_config(argv):
    try:
        config_path = argv[1]
        with open(config_path, "r") as stream:
            conf = yaml.safe_load(stream)
    except IndexError:
        print('Missing command line argument for the path to the configuration file. '
              'Please use this program like this: main.py PATH_TO_CONFIG_YAML_FILE')
        sys.exit()

    return config_path, conf


def create_experiment_dir(conf_file, exp_path, run_mode):
    config_name = conf_file[10:-5]
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    base_path = os.path.join(exp_path, run_mode, config_name, now)

    paths = [os.path.join(base_path, 'results')]
    if run_mode in config.VALID_MODES[0:2]:
        paths.append(os.path.join(base_path, 'stats'))
    if run_mode == config.VALID_MODES[0]:
        paths.append(os.path.join(base_path, 'model'))

    for path in paths:
        os.makedirs(path)
        print('Created directory', path)

    shutil.copy2(conf_file, base_path)
    print('Copied config file', conf_file, 'to', base_path)

    return base_path


if __name__ == '__main__':
    configuration_file, configuration = parse_config(sys.argv)
    mode = configuration.get(config.MODE_KEY)

    if configuration.get(config.PRETRAINED_PATH_KEY) is not None:
        experiment_path = configuration.get(config.PRETRAINED_PATH_KEY)
    else:
        experiment_path = create_experiment_dir(configuration_file, configuration.get(config.EXPERIMENT_PATH_KEY), mode)
    logger = setup_logger(experiment_path)

    logger.info('Successfully read the given configuration file, created experiment directories and set up logger.')
    logger.info('Starting experiment in mode ' + mode + ' using configuration ' + configuration_file)

    try:
        if mode not in config.VALID_MODES:
            msg = 'Mode has to be one of' + str(config.VALID_MODES)
            raise ConfigurationError(msg)
        if mode == config.VALID_MODES[0]:
            trainer = Trainer(configuration.get(config.DEVICE_KEY),
                              experiment_path,
                              configuration.get(config.EPOCHS_KEY),
                              configuration.get(config.NUM_IMGS_KEY),
                              configuration.get(config.PREDEFINED_IMAGES_KEY),
                              configuration.get(config.FREQUENCIES_KEY),
                              configuration.get(config.MODEL_KEY),
                              configuration.get(config.DATALOADER_KEY))
            trainer.train()
        if mode == config.VALID_MODES[1]:
            # TODO
            raise NotImplementedError
            # evaluator = Evaluator()
            # evaluator.evaluate()
        if mode == config.VALID_MODES[2]:
            # TODO
            raise NotImplementedError
            # generator = PhotofitGenerator()
            # generator.generate()
    except (ConfigurationError, NotImplementedError) as e:
        logger.exception(e)
