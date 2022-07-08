from error import ConfigurationError
from torch.optim import Adam, Adagrad, SGD

VALID_OPTIMIZER_NAMES = {'Adam': Adam, 'Adagrad': Adagrad, 'SGD': SGD}


class TediGAN:
    def __init__(self, model_params, optimizer_name, image_size, learning_rate, device_name):

        if optimizer_name not in VALID_OPTIMIZER_NAMES:
            raise ConfigurationError('Specified optimizer is not valid. Valid optimizers: ' + str(VALID_OPTIMIZER_NAMES))

        # TODO initialize optimizer
        # self.optimizer = VALID_OPTIMIZER_NAMES[optimizer_name](self.parameters(), lr=learning_rate)

        # TODO
        raise NotImplementedError
