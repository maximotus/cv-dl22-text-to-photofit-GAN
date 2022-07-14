import logging

from data.dataset import CelebA, CelebAHQ, LSW
from error import ConfigurationError
from model.cdcgan import CDCGAN
from model.tedi_gan import TediGAN
from tqdm.auto import tqdm

logger = logging.getLogger('root')
VALID_MODEL_NAMES = {'DCGAN': CDCGAN, 'tediGAN': TediGAN}
VALID_DATASET_NAMES = {'celebA': CelebA, 'celebA_HQ': CelebAHQ, 'LSW': LSW}


class Trainer:
    def __init__(self, model_name, model_params, dataset_name, epochs, batch_size, optimizer_name, learning_rate,
                 criterion_name, device_name, save_freq, gen_freq, image_size, experiment_path):

        logger.info('Initializing trainer...')

        if model_name not in VALID_MODEL_NAMES:
            raise ConfigurationError('Specified model.name is not valid. Valid names: ' + str(VALID_MODEL_NAMES))

        if dataset_name not in VALID_DATASET_NAMES:
            raise ConfigurationError('Specified dataset is not valid. Valid datasets: ' + str(VALID_DATASET_NAMES))

        self.dataset = VALID_DATASET_NAMES[dataset_name](image_size, batch_size)
        self.model = VALID_MODEL_NAMES[model_name](model_params, optimizer_name, learning_rate, criterion_name,
                                                   device_name, self.dataset.class_mapping)

        self.model_name = model_name
        self.model_params = model_params
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.save_freq = save_freq
        self.gen_freq = gen_freq
        self.device_name = device_name
        self.experiment_path = experiment_path

    def train(self):
        logger.info('Starting training...')

        for epoch in tqdm(range(self.epochs), desc='Epoch'):
            for step, batch in enumerate(tqdm(self.dataset.data_loader, desc='Batch')):
                # TODO this only works with the implemented celebA configuration
                images = batch[0]
                attributes = batch[1][0]
                identities = batch[1][1]
                bboxes = batch[1][2]
                landmarks = batch[1][3]

                # TODO remove the following prints (they are for development purposes only)
                print(images.shape)
                print(attributes.shape)
                print(identities.shape)
                print(bboxes.shape)
                print(landmarks.shape)

                # TODO self.model.fit()
                # in fit(), the forward pass, loss calculation and backpropagation should be done
                raise NotImplementedError

            if epoch % self.save_freq == 0:
                # TODO save model
                raise NotImplementedError

            if epoch % self.gen_freq == 0:
                # TODO generate samples (save them in folder results)
                raise NotImplementedError
