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


class Trainer:
    def __init__(self, model_name, model_params, dataset_name, dataset_size_factor, epochs, batch_size, optimizer_name, learning_rate,
                 criterion_name, device_name, save_freq, gen_freq, image_size, num_imgs, predefined_images, experiment_path):

        logger.info('Initializing trainer...')

        if model_name not in VALID_MODEL_NAMES:
            raise ConfigurationError('Specified model.name is not valid. Valid names: ' + str(VALID_MODEL_NAMES))

        if dataset_name not in VALID_DATASET_NAMES:
            raise ConfigurationError('Specified dataset is not valid. Valid datasets: ' + str(VALID_DATASET_NAMES))

        if device_name == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_name)
        logger.info('Using device ' + str(self.device))

        self.dataset = VALID_DATASET_NAMES[dataset_name](image_size, batch_size, dataset_size_factor)
        self.model = VALID_MODEL_NAMES[model_name](model_params, optimizer_name, learning_rate, criterion_name,
                                                   len(self.dataset.attribute_to_idx), self.device)

        self.model_name = model_name
        self.model_params = model_params
        self.dataset_name = dataset_name
        self.dataset_size_factor = dataset_size_factor
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.save_freq = save_freq
        self.gen_freq = gen_freq
        self.num_imgs = num_imgs
        self.experiment_path = experiment_path
        self.predefined_images = predefined_images

    def train(self):
        logger.info('Starting training...')
        # TODO use subset of celeba

        for epoch in tqdm(range(self.epochs), desc='Epoch'):
            for step, batch in enumerate(tqdm(self.dataset.data_loader, desc='Batch')):
                images = batch[0]
                attributes = batch[1][0]
                # identities = batch[1][1]
                # bboxes = batch[1][2]
                # landmarks = batch[1][3]

                # in fit(), the forward pass, loss calculation and backpropagation is done
                self.model.fit(images, attributes)

            self.model.after_epoch()

            if epoch % self.save_freq == 0:
                self.model.save_ckpt(self.experiment_path, epoch)
                self.model.save_stats(self.experiment_path, epoch)

            if epoch % self.gen_freq == 0:
                self.model.save_fixed_img(self.experiment_path, epoch)
                self.model.save_random_img(self.num_imgs, self.experiment_path, epoch)
                self.model.save_predefined_img(self.num_imgs, self.experiment_path, epoch, self.predefined_images)


class Evaluator:
    def __init__(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class PhotofitGenerator:
    def __init__(self):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError

class Continuation:
    def __init__(self, model_name, model_params, dataset_name, dataset_size_factor, current_epoch, epochs, batch_size, optimizer_name, learning_rate,
                 criterion_name, device_name, save_freq, gen_freq, image_size, num_imgs, predefined_images, experiment_path):
        logger.info('Initializing Continuation of Training...')
        
        if model_name not in VALID_MODEL_NAMES:
            raise ConfigurationError('Specified model.name is not valid. Valid names: ' + str(VALID_MODEL_NAMES))

        if dataset_name not in VALID_DATASET_NAMES:
            raise ConfigurationError('Specified dataset is not valid. Valid datasets: ' + str(VALID_DATASET_NAMES))

        if device_name == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_name)
        logger.info('Using device ' + str(self.device))

        self.dataset = VALID_DATASET_NAMES[dataset_name](image_size, batch_size, dataset_size_factor)
        self.model = VALID_MODEL_NAMES[model_name](model_params, optimizer_name, learning_rate, criterion_name,
                                                   len(self.dataset.attribute_to_idx), self.device)

        # load pretrained gen and disc
        self.model.generator.load_state_dict(torch.load(experiment_path + '\model\generator\epoch-' + str(0) + '.pt'), strict=False)#str(current_epoch)
        self.model.discriminator.load_state_dict(torch.load(experiment_path + '\model\discriminator\epoch-' + str(0) + '.pt'), strict=False)#str(current_epoch)

        self.model_name = model_name
        self.model_params = model_params
        self.dataset_name = dataset_name
        self.dataset_size_factor = dataset_size_factor
        self.current_epoch = current_epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.save_freq = save_freq
        self.gen_freq = gen_freq
        self.num_imgs = num_imgs
        self.experiment_path = experiment_path
        self.predefined_images = predefined_images        
         
    def train(self):
        logger.info('Continue training ...')

        for epoch in tqdm(range(self.current_epoch + 1, self.epochs), desc='Epoch'):
            for step, batch in enumerate(tqdm(self.dataset.data_loader, desc='Batch')):
                images = batch[0]
                attributes = batch[1][0]
                # identities = batch[1][1]
                # bboxes = batch[1][2]
                # landmarks = batch[1][3]

                # in fit(), the forward pass, loss calculation and backpropagation is done
                self.model.fit(images, attributes)

            self.model.after_epoch()

            if epoch % self.save_freq == 0:
                self.model.save_ckpt(self.experiment_path, epoch)
                self.model.save_stats(self.experiment_path, epoch)

            if epoch % self.gen_freq == 0:
                self.model.save_fixed_img(self.experiment_path, epoch)
                self.model.save_random_img(self.num_imgs, self.experiment_path, epoch)
                self.model.save_predefined_img(self.num_imgs, self.experiment_path, epoch, self.predefined_images)