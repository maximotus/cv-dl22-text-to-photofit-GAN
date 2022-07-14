import logging
import torch

from einops import rearrange
from error import ConfigurationError
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam, Adagrad, SGD

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


class SpectralNormedConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()

        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                               stride=(stride, stride), padding=padding, bias=bias)
        self.conv = torch.nn.utils.spectral_norm(conv)

    def forward(self, x):
        return self.conv(x)


class SpectralNormedLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()

        linear = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.linear = torch.nn.utils.spectral_norm(linear)

    def forward(self, x):
        return self.linear(x)


class Discriminator(torch.nn.Module):
    def __init__(self, conv, linear, nf=64, embed_dim=32, num_classes=10, dropout=0.0):
        super().__init__()

        logger.info('Initializing Discriminator...')

        self.block1 = torch.nn.Sequential(conv(3, nf, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                          torch.nn.LeakyReLU(0.1, True),
                                          conv(nf, nf * 2, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                          torch.nn.LeakyReLU(0.1, True),
                                          conv(nf * 2, nf * 4, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                          torch.nn.LeakyReLU(0.1, True),
                                          conv(nf * 4, nf * 8, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                          torch.nn.LeakyReLU(0.1, True))

        self.embedding = torch.nn.Embedding(num_classes, embed_dim)

        self.block2 = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                          linear(4 * 4 * nf * 8, 128),
                                          torch.nn.LeakyReLU(0.2, True))

        self.final = torch.nn.Linear(128 + embed_dim, 1)

        logger.info('Successfully initialized Discriminator...')

    def forward(self, x, c=None):
        x = self.block1(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.block2(x)
        c = self.embedding(c)
        x = torch.cat([x, c], dim=1)
        return self.final(x)

    def set_grad(self, status):
        for p in self.parameters():
            p.requires_grad = status


class Generator(torch.nn.Module):
    def __init__(self, z_channels=128, nf=64, embed_dim=32, num_classes=10):
        super().__init__()

        logger.info('Initializing Generator...')

        self.embedding = torch.nn.Embedding(num_classes, embed_dim)

        self.initial = torch.nn.Sequential(torch.nn.Linear(z_channels + embed_dim, 4 * 4 * nf * 8),
                                           torch.nn.ReLU(True))

        self.block = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2.0),
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
        c = self.embedding(c)
        z = torch.cat([z, c], dim=1)
        x = self.initial(z)
        x = rearrange(x, "b (c h w) -> b c h w", h=4, w=4)
        return self.block(x)


class CDCGAN:
    def __init__(self, model_params, optimizer_name, learning_rate, criterion_name, device_name, num_classes):
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
        self.emb_dim = model_params.get('emb_dim')
        self.use_spectral_norm = model_params.get('use_spectral_norm')
        self.z_channels = model_params.get('z_channels')
        self.device = torch.device(device_name)
        self.num_classes = num_classes
        self.lr = learning_rate

        # initialize loss function
        if criterion_name not in VALID_CRITERION_NAMES:
            raise ConfigurationError('Specified criterion is not valid. Valid loss functions: ' + str(VALID_CRITERION_NAMES))
        self.criterion = VALID_CRITERION_NAMES[criterion_name]()

        # initialize discriminator network
        self.discriminator = Discriminator(conv=(SpectralNormedConv2d if self.use_spectral_norm else torch.nn.Conv2d),
                                           linear=(SpectralNormedLinear if self.use_spectral_norm else torch.nn.Linear),
                                           nf=self.ndf, embed_dim=self.emb_dim, num_classes=self.num_classes,
                                           dropout=self.dropout)
        self.discriminator.apply(weight_init)
        self.discriminator.to(self.device)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        # initialize generator network
        self.generator = Generator(z_channels=self.z_channels, nf=self.ngf, embed_dim=self.emb_dim,
                                   num_classes=self.num_classes)
        self.generator.apply(weight_init)
        self.generator.to(self.device)
        self.generator_optimizer = Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

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
