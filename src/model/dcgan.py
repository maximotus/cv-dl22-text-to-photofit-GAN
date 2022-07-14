import torch

from einops import rearrange
from error import ConfigurationError
from torch.optim import Adam, Adagrad, SGD

VALID_OPTIMIZER_NAMES = {'Adam': Adam, 'Adagrad': Adagrad, 'SGD': SGD}


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


class DCGAN:
    def __init__(self, model_params, optimizer_name, image_size, learning_rate, device_name):

        if optimizer_name not in VALID_OPTIMIZER_NAMES:
            raise ConfigurationError('Specified optimizer is not valid. Valid optimizers: ' + str(VALID_OPTIMIZER_NAMES))

        # TODO
        raise NotImplementedError


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # TODO initialize optimizer
        # self.optimizer = VALID_OPTIMIZER_NAMES[optimizer_name](self.parameters(), lr=learning_rate)

        # TODO
        raise NotImplementedError


class Generator(torch.nn.Module):
    def __init__(self, z_channels=128, nf=64, embed_dim=32, num_classes=10):
        super(Generator, self).__init__()

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

    def forward(self, z, c):
        c = self.embedding(c)
        z = torch.cat([z, c], dim=1)
        x = self.initial(z)
        x = rearrange(x, "b (c h w) -> b c h w", h=4, w=4)
        return self.block(x)
