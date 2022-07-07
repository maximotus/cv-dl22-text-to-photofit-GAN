# TODO argument parsing
"""
arguments:
    - learning rate
    - batch_size
    - epochs
    - optimizer
    - noise_size
    - image_size
    - use_spectral_norm
    - dataset
    - network / model
    - model-specific parameters (dropout, alpha, beta, ...)
    - device (gpu / cpu / auto)
    - experiment_path
        - experiment_path/stats
        - experiment_path/ckpt
        - experiment_path/logs
    - frequencies
        - save_freq (i.e. how often should the model checkpoint be saved)
        - gen_freq (i.e. how often should test images be generated)
"""

import yaml

if __name__ == '__main__':
    config = {}
    with open("../config/test-config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    print(config.get('learning_rate'))

    # TODO build experiment name
    # config_file_name + timestamp

    # TODO copy config to experiment folder

    # TODO create sub dirs
    # experiment_path / stats
    # experiment_path / ckpt
    # experiment_path / logs


