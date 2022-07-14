import linecache
import ssl
import torchvision
import torchvision.transforms as transforms

from itertools import islice
from torch.utils.data import DataLoader
from urllib.error import URLError


class CelebA:
    def __init__(self, image_size, batch_size):

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        try:
            data = torchvision.datasets.CelebA(root='../../data', split='all',
                                               target_type=['attr', 'identity', 'bbox', 'landmarks'],
                                               transform=transform, download=True)
        except URLError:
            ssl._create_default_https_context = ssl._create_unverified_context
            data = torchvision.datasets.CelebA(root='../../data', split='all',
                                               target_type=['attr', 'identity', 'bbox', 'landmarks'],
                                               transform=transform, download=True)

        self.data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.length = len(data)

        attributes = linecache.getline(r'../../data/celeba/list_attr_celeba.txt', 2).strip().split(' ')
        self.attribute_to_idx = dict(zip(attributes, range(len(attributes))))


class CelebAHQ:
    def __init__(self, image_size, batch_size):
        # TODO
        raise NotImplementedError


class LSW:
    def __init__(self, image_size, batch_size):
        # TODO
        raise NotImplementedError
