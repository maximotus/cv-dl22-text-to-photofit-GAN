import logging
import os

import misc.config as config
import torch
from torchvision.utils import save_image

from misc.error import ConfigurationError
from tqdm.auto import tqdm

logger = logging.getLogger('root')


class Creator:
    def __init__(self, device_name, experiment_path, num_imgs, predefined_images, model, dataloader):

        logger.info('Initializing creator...')

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

        logger.info('Successfully initialized creator')

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
    def __init__(self, device_name, experiment_path, num_imgs, predefined_images, model, dataloader, metrics):

        logger.info('Initializing evaluator...')

        super().__init__(device_name, experiment_path, num_imgs, predefined_images, model, dataloader)

        self.metrics = metrics
        for metric in self.metrics:
            if metric not in config.VALID_METRIC_NAMES:
                raise ConfigurationError('Specified metric is not valid. Valid metrics: ' + str(config.VALID_METRIC_NAMES))           

        logger.info('Successfully initialized evaluator')

    def evaluate(self):
        paths = [self.experiment_path + '/img-fake', self.experiment_path + '/img-real']

        batch = next(iter(self.dataset.data_loader))
        batch_size = batch[0].size(0)
        if batch_size < self.num_imgs:
            raise ConfigurationError(
                'Cannot use num_imgs=' + str(self.num_imgs) + ' out of a batch of batch_size=' + str(batch_size) +
                '. Please adapt the configuration so batch_size >= num_imgs.')

        real_attributes = batch[1][0][0:self.num_imgs]
        real_images = batch[0][0:self.num_imgs]
        real_images = torch.tensor_split(real_images, self.num_imgs, dim=0)
        fake_images = self.model.generate_images(n=self.num_imgs, c=real_attributes).detach()
        fake_images = torch.tensor_split(fake_images, self.num_imgs, dim=0)

        for i in range(self.num_imgs):
            fake_image = fake_images[i]
            real_image = real_images[i]

            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(path)
                    print('Created directory', path)

            save_image(fake_image, paths[0] + '/' + str(i) + '.jpg')
            save_image(real_image, paths[1] + '/' + str(i) + '.jpg')

        scores = {metric_name: 0 for metric_name in self.metrics}

        for metric_name in self.metrics:
            if metric_name == 'BRISQUE':
                scores[metric_name] = config.VALID_METRIC_NAMES[metric_name](paths[0] + '/0.jpg')
            elif metric_name == 'LPIPS':
                scores[metric_name] = config.VALID_METRIC_NAMES[metric_name](paths[0] + '/0.jpg', paths[1] + '/0.jpg',)
            else:
                scores[metric_name] = config.VALID_METRIC_NAMES[metric_name](paths[0], paths[1])

        logger.info(scores)
