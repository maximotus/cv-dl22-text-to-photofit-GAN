import logging
import misc.config as config
import torch

from misc.error import ConfigurationError
from tqdm.auto import tqdm

logger = logging.getLogger('root')


class Creator:
    def __init__(self, device_name, experiment_path, num_imgs, predefined_images, model, dataloader):

        logger.info('Initializing photofit creator...')

        if dataloader.get(config.DATASET_KEY) not in config.VALID_DATASET_NAMES:
            raise ConfigurationError('Specified dataset is not valid. Valid datasets: ' + str(config.VALID_DATASET_NAMES))

        if model.get(config.NAME_KEY) not in config.VALID_MODEL_NAMES:
            raise ConfigurationError('Specified model.name is not valid. Valid names: ' + str(config.VALID_MODEL_NAMES))

        if device_name == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_name)

        logger.info('Using device ' + str(device))

        self.dataset = config.VALID_DATASET_NAMES[dataloader.get(config.DATASET_KEY)](
            dataloader.get(config.IMAGE_SIZE_KEY),
            dataloader.get(config.BATCH_SIZE_KEY),
            dataloader.get(config.SIZE_FRACTION_KEY))

        self.model = config.VALID_MODEL_NAMES[model.get(config.NAME_KEY)](
            model.get(config.MODEL_PARAMETERS_KEY),
            model.get(config.OPTIMIZER_KEY),
            model.get(config.LEARNING_RATE_KEY),
            model.get(config.CRITERION_KEY),
            len(self.dataset.attribute_to_idx),
            device,
            model.get(config.PRETRAINED_PATH_KEY),
            model.get(config.START_EPOCH_KEY))

        self.num_imgs = num_imgs
        self.predefined_images = predefined_images
        self.experiment_path = experiment_path
        self.start_epoch = model.get(config.START_EPOCH_KEY)

        logger.info('Successfully initialized photofit creator')

    def create(self, epoch=None):
        logger.info('Creating images...')
        epoch = self.start_epoch if epoch is None else epoch
        self.model.save_fixed_img(self.experiment_path, epoch)
        self.model.save_random_img(self.num_imgs, self.experiment_path, epoch)
        self.model.save_predefined_img(self.num_imgs, self.experiment_path, epoch, self.predefined_images)
        logger.info('Successfully created images')


class Trainer(Creator):
    def __init__(self, device_name, experiment_path, epochs, num_imgs, predefined_images, frequencies, model,
                 dataloader):

        logger.info('Initializing trainer...')

        super().__init__(device_name, experiment_path, num_imgs, predefined_images, model, dataloader)

        self.epochs = epochs
        self.save_freq = frequencies.get(config.SAVE_FREQUENCY_KEY)
        self.gen_freq = frequencies.get(config.GEN_FREQUENCY_KEY)

        logger.info('Successfully initialized trainer')

    def train(self):
        if self.start_epoch is None:
            self.start_epoch = 0
            logger.info('Starting training...')
        else:
            self.start_epoch += 1
            logger.info('Continuing training from epoch ' + str(self.start_epoch))

        for epoch in tqdm(range(self.start_epoch, self.start_epoch + self.epochs), desc='Epoch'):
            logger.info('Starting epoch ' + str(epoch))
            for step, batch in enumerate(tqdm(self.dataset.data_loader, desc='Batch')):
                images = batch[0]
                attributes = batch[1][0]
                self.model.fit(images, attributes)

            self.model.after_epoch()

            if epoch % self.save_freq == 0:
                self.model.save_ckpt(self.experiment_path, epoch)
                self.model.save_stats(self.experiment_path, epoch)

            if epoch % self.gen_freq == 0:
                self.create(epoch)


class Evaluator(Creator):
    def __init__(self, device_name, experiment_path, num_imgs, predefined_images, model, dataloader):
        super().__init__(device_name, experiment_path, num_imgs, predefined_images, model, dataloader)
        # TODO
        raise NotImplementedError

    def evaluate(self):
        # TODO
        raise NotImplementedError
