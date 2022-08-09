import logging
import misc.config as config
import numpy as np
import os
import torch

from misc.error import ConfigurationError
from model.helper import SpectralNormedConv2d
from torchvision.utils import save_image

logger = logging.getLogger('root')


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.2)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


class Discriminator(torch.nn.Module):
    def __init__(self, conv, device, nf=64, num_classes=10, dropout=0.0):
        super().__init__()

        logger.info('Initializing Discriminator...')

        self.device = device
        self.nf = nf
        self.num_classes = num_classes

        # initialize helper functions
        self.sig = torch.nn.Sigmoid()
        self.relu = torch.nn.LeakyReLU(0.2, inplace=True)

        # initialize layers
        self.conv_x = conv(3, self.nf, (4, 4), (2, 2), 1, bias=False)
        self.dropout_x = torch.nn.Dropout(p=dropout)

        self.conv_y = conv(self.num_classes, self.nf, (4, 4), (2, 2), 1, bias=False)
        self.dropout_y = torch.nn.Dropout()

        self.conv1 = conv(nf * 2, self.nf * 4, (4, 4), (2, 2), 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.nf * 4)
        self.dropout1 = torch.nn.Dropout(p=dropout)

        self.conv2 = conv(self.nf * 4, self.nf * 8, (4, 4), (2, 2), 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(self.nf * 8)
        self.dropout2 = torch.nn.Dropout(p=dropout)

        self.conv3 = conv(self.nf * 8, self.nf * 16, (4, 4), (2, 2), 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(self.nf * 16)
        self.dropout3 = torch.nn.Dropout(p=dropout)

        self.conv4 = conv(self.nf * 16, self.nf * 32, (4, 4), (2, 2), 1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(self.nf * 32)
        self.dropout4 = torch.nn.Dropout(p=dropout)

        self.conv5 = conv(self.nf * 32, 1, (2, 2), (1, 1), 0, bias=False)

        logger.info('Successfully initialized Discriminator...')

    def preprocess_c(self, c, img_size):
        c = c.type(torch.LongTensor)
        target = [c.size(0), c.size(1), img_size, img_size]
        c = c[:, :, None, None].expand(target)
        c = c.type(torch.float)
        # torch.set_printoptions(profile="full")
        # print(c)
        return c.to(self.device)

    def forward(self, x, c):
        img_size = x.size(2)

        logger.debug('DISCRIMINATOR')
        logger.debug('0 x ' + str(x.shape))
        logger.debug('0 c ' + str(c.shape))
        x = self.conv_x(x)
        x = self.relu(x)
        x = self.dropout_x(x)
        logger.debug('1 x ' + str(x.shape))

        y = self.preprocess_c(c, img_size)
        logger.debug('1 y ' + str(y.shape))

        y = self.conv_y(y)
        y = self.relu(y)
        y = self.dropout_y(y)
        logger.debug('2 y ' + str(y.shape))

        xy = torch.cat([x, y], dim=1)
        logger.debug(xy.shape)
        xy = self.conv1(xy)
        xy = self.bn1(xy)
        xy = self.relu(xy)
        xy = self.dropout1(xy)
        logger.debug('1 xy ' + str(xy.shape))

        xy = self.conv2(xy)
        xy = self.bn2(xy)
        xy = self.relu(xy)
        xy = self.dropout2(xy)
        logger.debug('2 xy ' + str(xy.shape))

        xy = self.conv3(xy)
        xy = self.bn3(xy)
        xy = self.relu(xy)
        xy = self.dropout3(xy)
        logger.debug('3 xy ' + str(xy.shape))

        xy = self.conv4(xy)
        xy = self.bn4(xy)
        xy = self.relu(xy)
        xy = self.dropout4(xy)
        logger.debug('4 xy ' + str(xy.shape))

        xy = self.conv5(xy)
        logger.debug(xy.shape)
        xy = self.sig(xy)
        logger.debug('5 xy ' + str(xy.shape))

        return xy

    def set_grad(self, status):
        for p in self.parameters():
            p.requires_grad = status


class Generator(torch.nn.Module):
    def __init__(self, device, z_channels=128, nf=64, num_classes=10):
        super().__init__()

        logger.info('Initializing Generator...')

        self.device = device

        # self.initial_z = torch.nn.Sequential(
        #     torch.nn.ConvTranspose2d(z_channels, nf * 8, (4, 4), (1, 1), (0, 0), bias=False),
        #     torch.nn.BatchNorm2d(nf * 8))
        # self.initial_c = torch.nn.Sequential(
        #     torch.nn.ConvTranspose2d(num_classes, nf * 8, (4, 4), (1, 1), (0, 0), bias=False),
        #     torch.nn.BatchNorm2d(nf * 8)
        # )

        self.initial_z = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=4.0),
            torch.nn.Conv2d(z_channels, nf * 8, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(nf * 8)
        )

        self.initial_c = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=4.0),
            torch.nn.Conv2d(num_classes, nf * 8, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(nf * 8)
        )

        self.block = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2.0),
                                         torch.nn.Conv2d(nf * 16, nf * 8, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                         torch.nn.BatchNorm2d(nf * 8),
                                         torch.nn.ReLU(True),
                                         torch.nn.Upsample(scale_factor=2.0),
                                         torch.nn.Conv2d(nf * 8, nf * 4, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                         torch.nn.BatchNorm2d(nf * 4),
                                         torch.nn.ReLU(True),
                                         torch.nn.Upsample(scale_factor=2.0),
                                         torch.nn.Conv2d(nf * 4, nf * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                         torch.nn.BatchNorm2d(nf * 2),
                                         torch.nn.ReLU(True),
                                         torch.nn.Upsample(scale_factor=2.0),
                                         torch.nn.Conv2d(nf * 2, nf, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                         torch.nn.BatchNorm2d(nf),
                                         torch.nn.ReLU(True),
                                         torch.nn.Conv2d(nf, 3, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                         torch.nn.Tanh())

        logger.info('Successfully initialized Generator')

    def forward(self, z, c):
        # z: [batch_size, num_z]
        # c: [batch_size, num_classes]
        # print('GENERATOR')
        # print('z', z.shape)
        # print('c', c.shape)

        z = z[:, :, None, None].to(self.device)
        # print('z', z.shape)
        z = self.initial_z(z)
        # print('z', z.shape)
        c = c[:, :, None, None].float().to(self.device)
        # print('c', c.shape)
        c = self.initial_c(c)
        # print('c', c.shape)
        # z: [batch_size, nf * 8, 4, 4]
        # c: [batch_size, nf * 8, 4, 4]

        x = torch.cat([z, c], dim=1)
        # print('x', x.shape)
        # x: [batch_size, nf * 8 + nf * 8, 4, 4]

        x = self.block(x)
        # print('x', x.shape)
        # x: [batch_size, 3, 64, 64]
        return x


class CDCGAN:
    def __init__(self, model_params, optimizer_name, learning_rate, criterion_name, num_classes, device,
                 pretrained_path, current_epoch):
        if optimizer_name not in config.VALID_OPTIMIZER_NAMES:
            raise ConfigurationError(
                'Specified optimizer is not valid. Valid optimizers: ' + str(config.VALID_OPTIMIZER_NAMES))

        logger.info('Initializing CDCGAN...')

        # initialize parameters
        self.dropout = model_params.get('dropout')
        self.alpha = model_params.get('alpha')
        self.beta1 = model_params.get('beta1')
        self.ngf = model_params.get('ngf')
        self.ndf = model_params.get('ndf')
        self.use_spectral_norm = model_params.get('use_spectral_norm')
        self.z_channels = model_params.get('z_channels')
        self.num_classes = num_classes
        self.lr = learning_rate
        self.device = device

        # initialize loss function
        if criterion_name not in config.VALID_CRITERION_NAMES:
            raise ConfigurationError(
                'Specified criterion is not valid. Valid loss functions: ' + str(config.VALID_CRITERION_NAMES))
        self.criterion = config.VALID_CRITERION_NAMES[criterion_name]()

        # initialize discriminator network
        self.discriminator = Discriminator(conv=(SpectralNormedConv2d if self.use_spectral_norm else torch.nn.Conv2d),
                                           device=self.device, nf=self.ndf, num_classes=self.num_classes,
                                           dropout=self.dropout)
        self.discriminator.apply(weight_init)
        self.discriminator.to(self.device)

        # initialize generator network
        self.generator = Generator(self.device, z_channels=self.z_channels, nf=self.ngf, num_classes=self.num_classes)
        self.generator.apply(weight_init)
        self.generator.to(self.device)

        # load pretrained models if given
        if current_epoch is not None and pretrained_path is not None:
            pretrained_generator_path = pretrained_path + '/model/generator/epoch-' + str(current_epoch) + '.pt'
            pretrained_discriminator_path = pretrained_path + '/model/discriminator/epoch-' + str(current_epoch) + '.pt'
            self.generator.load_state_dict(torch.load(pretrained_generator_path, map_location=device), strict=False)
            self.discriminator.load_state_dict(torch.load(pretrained_discriminator_path, map_location=device), strict=False)
            logger.info('Loaded pretrained models from ' + pretrained_generator_path + ' and ' + pretrained_discriminator_path)

        # initialize optimizers
        self.discriminator_optimizer = config.VALID_OPTIMIZER_NAMES[optimizer_name](self.discriminator.parameters(),
                                                                                    lr=self.lr, betas=(self.beta1, 0.999))
        self.generator_optimizer = config.VALID_OPTIMIZER_NAMES[optimizer_name](self.generator.parameters(),
                                                                                lr=self.lr, betas=(self.beta1, 0.999))
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
        # TODO refactor to trainer
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

    def save_predefined_img(self, n, experiment_path, epoch, c_map):
        path = self.create_and_get_img_save_path(experiment_path, epoch)
        name, c = list(c_map.keys()), list(c_map.values())
        for i in range(n):
            for j in range(len(name)):
                img = self.generate_image(c=torch.FloatTensor(c[j]).unsqueeze(dim=0).to(self.device))
                save_path = path + '/predef_img_' + name[j] + '_' + str(i) + '.png'
                save_image(img, save_path)
        logger.info('Saved predefined image in ' + path)

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
