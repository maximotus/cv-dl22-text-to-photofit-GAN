import logging
import os

import torch

from einops import rearrange
from error import ConfigurationError
from model.helper import SpectralNormedConv2d, SpectralNormedLinear
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam, Adagrad, SGD
from torchvision.utils import save_image

logger = logging.getLogger('root')
VALID_OPTIMIZER_NAMES = {'Adam': Adam, 'Adagrad': Adagrad, 'SGD': SGD}
VALID_CRITERION_NAMES = {'BCELoss': BCELoss, 'CrossEntropyLoss': CrossEntropyLoss}


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
    def __init__(self, conv, linear, nf=64, num_classes=10, dropout=0.0):
        super().__init__()

        logger.info('Initializing Discriminator...')

        self.block_x = torch.nn.Sequential(conv(3, nf, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                           torch.nn.LeakyReLU(0.1, True),
                                           conv(nf, nf * 2, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                           torch.nn.LeakyReLU(0.1, True),
                                           conv(nf * 2, nf * 4, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                           torch.nn.LeakyReLU(0.1, True),
                                           conv(nf * 4, nf * 8, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                           torch.nn.LeakyReLU(0.1, True))
        self.block_c = torch.nn.Sequential(conv(num_classes, nf, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                           torch.nn.LeakyReLU(0.1, True),
                                           conv(nf, nf * 2, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                           torch.nn.LeakyReLU(0.1, True),
                                           conv(nf * 2, nf * 4, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                           torch.nn.LeakyReLU(0.1, True),
                                           conv(nf * 4, nf * 8, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                           torch.nn.LeakyReLU(0.1, True))

        self.fc_x = linear(4 * 4 * nf * 8 * 4, 128)
        self.fc_c = linear(4 * 4 * nf * 8 * 4, 128)

        self.final = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                         linear(4 * nf, 1),
                                         torch.nn.LeakyReLU(0.2, True),
                                         torch.nn.Sigmoid())

        logger.info('Successfully initialized Discriminator...')

    @staticmethod
    def preprocess_c(c, img_size):
        c = c.type(torch.LongTensor)
        target = [c.size(0), c.size(1), img_size, img_size]
        c = c[:, :, None, None].expand(target)
        c = c.type(torch.float)
        # torch.set_printoptions(profile="full")
        # print(c)
        return c

    def forward(self, x, c):
        img_size = x.size(2)

        x = self.block_x(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.fc_x(x)

        c = self.preprocess_c(c, img_size)
        c = self.block_c(c)
        c = rearrange(c, "b c h w -> b (c h w)")
        c = self.fc_c(c)

        xc = torch.cat([x, c], dim=1)
        xc = self.final(xc)

        return xc

    def set_grad(self, status):
        for p in self.parameters():
            p.requires_grad = status


class Generator(torch.nn.Module):
    def __init__(self, z_channels=128, nf=64, num_classes=10):
        super().__init__()

        logger.info('Initializing Generator...')

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

        z = z[:, :, None, None]
        z = self.initial_z(z)
        c = c[:, :, None, None].float()
        c = self.initial_c(c)
        # z: [batch_size, nf * 8, 4, 4]
        # c: [batch_size, nf * 8, 4, 4]

        x = torch.cat([z, c], dim=1)
        # x: [batch_size, nf * 8 + nf * 8, 4, 4]

        x = self.block(x)
        # x: [batch_size, 3, 64, 64]
        return x


class CDCGAN:
    def __init__(self, model_params, optimizer_name, learning_rate, criterion_name, num_classes, device):
        if optimizer_name not in VALID_OPTIMIZER_NAMES:
            raise ConfigurationError(
                'Specified optimizer is not valid. Valid optimizers: ' + str(VALID_OPTIMIZER_NAMES))

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
        if criterion_name not in VALID_CRITERION_NAMES:
            raise ConfigurationError(
                'Specified criterion is not valid. Valid loss functions: ' + str(VALID_CRITERION_NAMES))
        self.criterion = VALID_CRITERION_NAMES[criterion_name]()

        # initialize discriminator network
        self.discriminator = Discriminator(conv=(SpectralNormedConv2d if self.use_spectral_norm else torch.nn.Conv2d),
                                           linear=(SpectralNormedLinear if self.use_spectral_norm else torch.nn.Linear),
                                           nf=self.ndf, num_classes=self.num_classes, dropout=self.dropout)
        self.discriminator.apply(weight_init)
        self.discriminator.to(self.device)
        self.discriminator_optimizer = VALID_OPTIMIZER_NAMES[optimizer_name](self.discriminator.parameters(),
                                                                             lr=self.lr, betas=(self.beta1, 0.999))

        # initialize generator network
        self.generator = Generator(z_channels=self.z_channels, nf=self.ngf, num_classes=self.num_classes)
        self.generator.apply(weight_init)
        self.generator.to(self.device)
        self.generator_optimizer = VALID_OPTIMIZER_NAMES[optimizer_name](self.generator.parameters(),
                                                                         lr=self.lr, betas=(self.beta1, 0.999))

        # initialize state attributes
        self.average_losses = {'gan': [], 'd_real': [], 'd_fake': []}
        self.average_accuracies = {'acc_fake': [], 'acc_real': []}
        self.fix_images = []

        # fixed noise vector
        self.z_fix = torch.randn((1, 128)).to(self.device)
        self.c_fix = (torch.rand((1, self.num_classes), device=self.device) * 2.0).type(torch.long)
        # TODO properly defined attributes
        # self.c_fix = torch.tensor([0, 0, 0, 0, 1, ...])

        logger.info('Successfully initialized CDCGAN')

    def fit(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)

        # 1. generate fake image
        z = torch.randn((images.size(0), 128)).to(self.device)
        fakes = self.generator(z, targets)

        # 2. train discriminator with all-real batch and then all-fake batch
        self.discriminator.set_grad(True)
        self.discriminator_optimizer.zero_grad()

        pred_real = self.discriminator(images, targets)
        pred_fake = self.discriminator(fakes.detach(), targets)
        print(pred_real)
        print(pred_fake)

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

        # TODO save losses and probabilities in class attributes

        # cur_losses["GAN"].append(GAN.item())
        # cur_losses["Dreal"].append(Dreal.item())
        # cur_losses["Dfake"].append(Dfake.item())

        # c_real = torch.round(torch.sigmoid(pred_real)).squeeze()
        # c_fake = torch.round(torch.sigmoid(pred_fake)).squeeze()

        # num_total += images.size(0)
        # num_right_real += torch.sum(torch.eq(c_real, torch.ones_like(targets))).cpu().numpy()
        # num_right_fake += torch.sum(torch.eq(c_fake, torch.zeros_like(targets))).cpu().numpy()

        return gan.item(), d_real.item(), d_fake.item()

    def save_ckpt(self, experiment_path, epoch):
        paths = [experiment_path + '/model/generator', experiment_path + '/model/discriminator']
        for i, path in enumerate(paths):
            if not os.path.exists(path):
                os.mkdir(path)
                logger.info('Created directory ' + path)
            full_path = path + '/' + str(epoch) + '.pt'
            torch.save(self.generator.state_dict(), full_path)
            logger.info('Saved CDCGAN model as ' + path)

    def save_img(self, experiment_path, epoch):
        img = self.generator.forward(self.z_fix, self.c_fix)
        save_path = experiment_path + '/results/' + str(epoch) + '.png'
        save_image(img, save_path)
