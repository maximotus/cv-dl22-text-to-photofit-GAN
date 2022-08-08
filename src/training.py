import logging
import torch

from dataset import CelebA, CelebAHQ, LSW
from misc.error import ConfigurationError
from model.cdcgan import CDCGAN
from model.tedi_gan import TediGAN
from tqdm.auto import tqdm

logger = logging.getLogger('root')
VALID_MODEL_NAMES = {'CDCGAN': CDCGAN, 'TEDIGAN': TediGAN}
VALID_DATASET_NAMES = {'celebA': CelebA, 'celebA_HQ': CelebAHQ, 'LSW': LSW}


class Creator:
    def __init__(self, model_name, model_params, dataset_name, dataset_size_factor, batch_size, optimizer_name,
                 learning_rate, criterion_name, device_name, image_size, num_imgs, predefined_images, experiment_path,
                 current_epoch, pretrained_path):

        logger.info('Initializing photofit creator...')

        if model_name not in VALID_MODEL_NAMES:
            raise ConfigurationError('Specified model.name is not valid. Valid names: ' + str(VALID_MODEL_NAMES))

        if dataset_name not in VALID_DATASET_NAMES:
            raise ConfigurationError('Specified dataset is not valid. Valid datasets: ' + str(VALID_DATASET_NAMES))

        if device_name == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_name)
        logger.info('Using device ' + str(device))

        self.dataset = VALID_DATASET_NAMES[dataset_name](image_size, batch_size, dataset_size_factor)
        self.model = VALID_MODEL_NAMES[model_name](model_params, optimizer_name, learning_rate, criterion_name,
                                                   len(self.dataset.attribute_to_idx), device, pretrained_path,
                                                   current_epoch)

        self.num_imgs = num_imgs
        self.predefined_images = predefined_images
        self.experiment_path = experiment_path
        self.current_epoch = current_epoch


class Trainer(Creator):
    def __init__(self, model_name, model_params, dataset_name, dataset_size_factor, batch_size, optimizer_name,
                 learning_rate, criterion_name, device_name, image_size, num_imgs, predefined_images, experiment_path,
                 epochs, save_freq, gen_freq, current_epoch, pretrained_path):

        super().__init__(model_name, model_params, dataset_name, dataset_size_factor, batch_size, optimizer_name,
                         learning_rate, criterion_name, device_name, image_size, num_imgs, predefined_images,
                         experiment_path, current_epoch, pretrained_path)

        self.epochs = epochs
        self.save_freq = save_freq
        self.gen_freq = gen_freq

    def train(self):
        if self.current_epoch is None:
            self.current_epoch = 0
            logger.info('Starting training...')
        else:
            self.current_epoch += 1
            logger.info('Continuing training from epoch ' + str(self.current_epoch))

        for epoch in tqdm(range(self.current_epoch, self.current_epoch + self.epochs), desc='Epoch'):
            for step, batch in enumerate(tqdm(self.dataset.data_loader, desc='Batch')):
                images = batch[0]
                attributes = batch[1][0]
                self.model.fit(images, attributes)

            self.model.after_epoch()

            if epoch % self.save_freq == 0:
                self.model.save_ckpt(self.experiment_path, epoch)
                self.model.save_stats(self.experiment_path, epoch)

            if epoch % self.gen_freq == 0:
                self.model.save_fixed_img(self.experiment_path, epoch)
                self.model.save_random_img(self.num_imgs, self.experiment_path, epoch)
                self.model.save_predefined_img(self.num_imgs, self.experiment_path, epoch, self.predefined_images)


class Evaluator(Creator):
    def __init__(self, model_name, model_params, dataset_name, dataset_size_factor, batch_size, optimizer_name,
                 learning_rate, criterion_name, device_name, image_size, num_imgs, predefined_images, experiment_path,
                 current_epoch, pretrained_path):
        super().__init__(model_name, model_params, dataset_name, dataset_size_factor, batch_size, optimizer_name,
                         learning_rate, criterion_name, device_name, image_size, num_imgs, predefined_images,
                         experiment_path, current_epoch, pretrained_path)
        # TODO
        raise NotImplementedError

    def evaluate(self):
        # TODO
        raise NotImplementedError


class PhotofitGenerator(Creator):
    def __init__(self, model_name, model_params, dataset_name, dataset_size_factor, batch_size, optimizer_name,
                 learning_rate, criterion_name, device_name, image_size, num_imgs, predefined_images, experiment_path,
                 current_epoch, pretrained_path):
        super().__init__(model_name, model_params, dataset_name, dataset_size_factor, batch_size, optimizer_name,
                         learning_rate, criterion_name, device_name, image_size, num_imgs, predefined_images,
                         experiment_path, current_epoch, pretrained_path)
        # TODO
        raise NotImplementedError

    def generate(self):
        # TODO
        raise NotImplementedError
