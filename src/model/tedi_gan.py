import logging
import misc.config as config
import numpy as np
import os
import torch

from misc.error import ConfigurationError
from model.helper import Tedi_Generator, GradualStyleEncoder, BackboneEncoderUsingLastLayerIntoW, BackboneEncoderUsingLastLayerIntoWPlus
from torchvision.utils import save_image

logger = logging.getLogger('root')

# prerequisits:
# https://stackoverflow.com/questions/40504552/how-to-install-visual-c-build-tools c++ answer by Aaron Belchamber
# Cuda Toolkit 11.6
# pytorch for cuda 11.6


class TediGAN(torch.nn.Module):
    def __init__(self, model_params, optimizer_name, learning_rate, criterion_name, num_classes, device,
                 pretrained_path, current_epoch):
        super(TediGAN, self).__init__()
        if optimizer_name not in config.VALID_OPTIMIZER_NAMES:
            raise ConfigurationError(
                'Specified optimizer is not valid. Valid optimizers: ' + str(config.VALID_OPTIMIZER_NAMES))

        logger.info('Initializing TediGAN...')

        # initialize parameters
        self.dropout = model_params.get('dropout')
        self.alpha = model_params.get('alpha')
        self.beta1 = model_params.get('beta1')
        self.ngf = model_params.get('ngf')
        self.ndf = model_params.get('ndf')
        self.z_channels = model_params.get('z_channels')
        self.num_classes = num_classes
        self.lr = learning_rate
        self.device = device

        # initialize loss function
        if criterion_name not in config.VALID_CRITERION_NAMES:
            raise ConfigurationError(
                'Specified criterion is not valid. Valid loss functions: ' + str(config.VALID_CRITERION_NAMES))
        self.criterion = config.VALID_CRITERION_NAMES[criterion_name]()

        # define architecture
        self.discriminator = GradualStyleEncoder(50, 'ir_se')  # self.set_encoder()
        self.discriminator.to(self.device)

        # initialize generator network
        self.generator = Tedi_Generator(1024, 512, 8)
        self.generator.to(self.device)

        # load pretrained models if given
        if current_epoch is not None and pretrained_path is not None:
            pretrained_generator_path = pretrained_path + '/model/generator/epoch-' + str(current_epoch) + '.pt'
            pretrained_discriminator_path = pretrained_path + '/model/discriminator/epoch-' + str(current_epoch) + '.pt'
            self.generator.load_state_dict(torch.load(pretrained_generator_path, map_location=device), strict=False)
            self.discriminator.load_state_dict(torch.load(pretrained_discriminator_path, map_location=device),
                                               strict=False)
            logger.info(
                'Loaded pretrained models from ' + pretrained_generator_path + ' and ' + pretrained_discriminator_path)

        # initialize optimizers
        self.discriminator_optimizer = config.VALID_OPTIMIZER_NAMES[optimizer_name](self.discriminator.parameters(),
                                                                                    lr=self.lr, betas=(self.beta1, 0.999))
        self.generator_optimizer = config.VALID_OPTIMIZER_NAMES[optimizer_name](self.generator.parameters(),
                                                                                lr=self.lr, betas=(self.beta1, 0.999))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # load weights
        self.load_weights()

        # initialize state attributes
        self.average_losses = {'gan_loss': [], 'd_real_loss': [], 'd_fake_loss': []}
        self.epoch_losses = {'gan_loss': [], 'd_real_loss': [], 'd_fake_loss': []}
        self.average_accuracies = {'acc_real': [], 'acc_fake_1': [], 'acc_fake_2': [], 'acc_real_total': [],
                                   'acc_fake_total': []}
        self.epoch_accuracies = {'acc_real': [], 'acc_fake_1': [], 'acc_fake_2': [], 'total_right_real': 0,
                                 'total_right_fake': 0}
        self.num_total = 0
        self.fix_images = []

        # initialize fixed noise vector
        self.z_fix = torch.randn((1, 128)).to(self.device)
        self.c_fix = (torch.rand((1, self.num_classes), device=self.device) * 2.0).type(torch.long)
        # TODO properly defined attributes
        # self.c_fix = torch.tensor([0, 0, 0, 0, 1, ...])

        logger.info('Successfully initialized CDCGAN')

    def fit(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)

        # 1. generate fake image
        z = torch.randn((images.size(0), self.z_channels)).to(self.device)
        fakes = self.generator(z, targets)

        # 2. train discriminator with all-real batch and then all-fake batch
        self.discriminator.set_grad(True)
        self.discriminator_optimizer.zero_grad()

        pred_real = self.discriminator(images, targets)
        pred_fake = self.discriminator(fakes.detach(), targets)

        d_real = self.criterion(pred_real, torch.ones_like(pred_real))
        d_fake = self.criterion(pred_fake, torch.zeros_like(pred_fake))
        loss_d = (d_real + d_fake) * 0.5
        loss_d.backward()
        self.discriminator_optimizer.step()
        self.discriminator.set_grad(False)

        # 3. train generator
        self.generator_optimizer.zero_grad()
        pred_fake2 = self.discriminator(fakes, targets)
        gan = self.criterion(pred_fake2, torch.ones_like(pred_fake2))
        gan.backward()
        self.generator_optimizer.step()

        # 4. remember values of interest
        self.epoch_losses['gan_loss'].append(gan.item())
        self.epoch_losses['d_real_loss'].append(d_real.item())
        self.epoch_losses['d_fake_loss'].append(d_fake.item())

        self.epoch_accuracies['acc_real'].append(pred_real.squeeze().mean().item())
        self.epoch_accuracies['acc_fake_1'].append(pred_fake.squeeze().mean().item())
        self.epoch_accuracies['acc_fake_2'].append(pred_fake2.squeeze().mean().item())

        self.num_total += images.size(0)
        self.epoch_accuracies['total_right_real'] += torch.sum(
            torch.eq(torch.round(pred_real), torch.ones_like(targets))).cpu().numpy()
        self.epoch_accuracies['total_right_fake'] += torch.sum(
            torch.eq(torch.round(pred_fake), torch.zeros_like(targets))).cpu().numpy()

    def after_epoch(self):
        # remember accumulated values of the epoch and reset accumulators
        for k, v in self.epoch_losses.items():
            self.average_losses[k].append(np.mean(v))
            self.epoch_losses[k] = []
        self.average_accuracies['acc_real'].append(np.mean(self.epoch_accuracies['acc_real']))
        self.average_accuracies['acc_fake_1'].append(np.mean(self.epoch_accuracies['acc_fake_1']))
        self.average_accuracies['acc_fake_2'].append(np.mean(self.epoch_accuracies['acc_fake_2']))
        self.average_accuracies['acc_real_total'].append(self.epoch_accuracies['total_right_real'] / self.num_total)
        self.average_accuracies['acc_fake_total'].append(self.epoch_accuracies['total_right_fake'] / self.num_total)
        self.epoch_accuracies['acc_real'] = []
        self.epoch_accuracies['acc_fake_1'] = []
        self.epoch_accuracies['acc_fake_2'] = []
        self.epoch_accuracies['total_right_real'] = 0
        self.epoch_accuracies['total_right_fake'] = 0

    def save_ckpt(self, experiment_path, epoch):
        paths = [experiment_path + '/model/generator', experiment_path + '/model/discriminator']
        for i, path in enumerate(paths):
            if not os.path.exists(path):
                os.mkdir(path)
                logger.info('Created directory ' + path)
            full_path = path + '/epoch-' + str(epoch) + '.pt'
            torch.save(self.generator.state_dict(), full_path)
            logger.info('Saved CDCGAN model as ' + path)

    def save_stats(self, experiment_path, epoch):
        base_save_path = experiment_path + '/stats/epoch-' + str(epoch)
        if not os.path.exists(base_save_path):
            os.makedirs(base_save_path)
            logger.info('Created directory ' + base_save_path)

        to_save = self.average_losses | self.average_accuracies
        for name, value in to_save.items():
            save_path = base_save_path + '/' + name
            np.save(save_path, value)
        logger.info('Saved stats in ' + base_save_path)

    def save_fixed_img(self, experiment_path, epoch):
        path = self.create_and_get_img_save_path(experiment_path, epoch)
        img = self.generate_image(self.c_fix, self.z_fix)
        save_path = path + '/fixed_img.png'
        save_image(img, save_path)
        logger.info('Saved fixed generated image as ' + save_path)

    def save_random_img(self, n, experiment_path, epoch):
        path = self.create_and_get_img_save_path(experiment_path, epoch)
        for i in range(n):
            img = self.generate_image()
            save_path = path + '/rand_img_' + str(i) + '.png'
            save_image(img, save_path)
        logger.info('Saved randomly generated image in ' + path)

    def generate_image(self, c=None, z=None):
        if c is None:
            c = (torch.rand((1, self.num_classes), device=self.device) * 2.0).type(torch.long)
        if z is None:
            z = torch.randn((1, 128)).to(self.device)
        self.generator.eval()
        img = self.generator.forward(z, c)
        self.generator.train()
        return img

    @staticmethod
    def create_and_get_img_save_path(experiment_path, epoch):
        path = experiment_path + '/results/epoch-' + str(epoch)
        if not os.path.exists(path):
            os.mkdir(path)
            logger.info('Created directory ' + path)
        return path

    def load_weights(self):
        print('Loading encoders weights from irse50!')
        encoder_ckpt = torch.load('model/stylegan2-ffhq-config-f.pt')
        # if input to encoder is not an RGB image, do not load the input layer weights
        # if self.opts.label_nc != 0:
        encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
        self.discriminator.load_state_dict(encoder_ckpt, strict=False)
        print('Loading decoder weights from pretrained!')
        # ckpt = torch.load(self.opts.stylegan_weights)
        ckpt = encoder_ckpt
        self.generator.load_state_dict(ckpt['g_ema'], strict=False)
    # if self.opts.learn_in_w:
    # 	self.__load_latent_avg(ckpt, repeat=1)
    # else:
    # 	self.__load_latent_avg(ckpt, repeat=18)

# def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
#             inject_latent=None, return_latents=False, alpha=None):
# 	if input_code:
# 		codes = x
# 	else:
# 		codes = self.discriminator(x)
# 		# normalize with respect to the center of an average face
# 		if self.opts.start_from_latent_avg:
# 			if self.opts.learn_in_w:
# 				codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
# 			else:
# 				codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

# 	if latent_mask is not None:
# 		for i in latent_mask:
# 			if inject_latent is not None:
# 				if alpha is not None:
# 					codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
# 				else:
# 					codes[:, i] = inject_latent[:, i]
# 			else:
# 				codes[:, i] = 0

# 	input_is_latent = not input_code
# 	images, result_latent = self.generator([codes],
# 	                                     input_is_latent=input_is_latent,
# 	                                     randomize_noise=randomize_noise,
# 	                                     return_latents=return_latents)

# 	if resize:
# 		images = self.face_pool(images)

# 	if return_latents:
# 		return images, result_latent
# 	else:
# 		return images

# def set_encoder(self):
# 	if self.opts.encoder_type == 'GradualStyleEncoder':
# 		encoder = GradualStyleBlock(50, 'ir_se', self.opts)
# 	elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
# 		encoder = BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
# 	elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
# 		encoder = BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
# 	else:
# 		raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
# 	return encoder

# def __load_latent_avg(self, ckpt, repeat=None):
# 	if 'latent_avg' in ckpt:
# 		self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
# 		if repeat is not None:
# 			self.latent_avg = self.latent_avg.repeat(repeat, 1)
# 	else:
# 		self.latent_avg = None

# def get_keys(d, name):
# 	if 'state_dict' in d:
# 		d = d['state_dict']
# 	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
# 	return d_filt
