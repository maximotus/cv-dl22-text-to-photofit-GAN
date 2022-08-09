from dataset import CelebA, CelebAHQ, LSW
from model.cdcgan import CDCGAN
from model.tedi_gan import TediGAN
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam, Adagrad, SGD

# valid names to be specified in the configuration yaml files
VALID_MODES = ['train', 'eval', 'gen']
VALID_MODEL_NAMES = {'CDCGAN': CDCGAN, 'TEDIGAN': TediGAN}
VALID_DATASET_NAMES = {'celebA': CelebA, 'celebA_HQ': CelebAHQ, 'LSW': LSW}
VALID_OPTIMIZER_NAMES = {'Adam': Adam, 'Adagrad': Adagrad, 'SGD': SGD}
VALID_CRITERION_NAMES = {'BCELoss': BCELoss, 'CrossEntropyLoss': CrossEntropyLoss}

# keys corresponding to the overall specification
MODE_KEY = 'mode'
LOG_LEVEL_KEY = 'log_level'
DEVICE_KEY = 'device'
EXPERIMENT_PATH_KEY = 'experiment_path'
EPOCHS_KEY = 'epochs'
NUM_IMGS_KEY = 'num_imgs'
PREDEFINED_IMAGES_KEY = 'predefined_images'

# keys corresponding to the frequencies specification
FREQUENCIES_KEY = 'frequencies'
SAVE_FREQUENCY_KEY = 'save_freq'
GEN_FREQUENCY_KEY = 'gen_freq'

# keys corresponding to the model specification
MODEL_KEY = 'model'
NAME_KEY = 'name'
PRETRAINED_PATH_KEY = 'pretrained_path'
START_EPOCH_KEY = 'start_epoch'
CRITERION_KEY = 'criterion'
OPTIMIZER_KEY = 'optimizer'
LEARNING_RATE_KEY = 'learning_rate'
MODEL_PARAMETERS_KEY = 'parameters'

# keys corresponding to the dataloader specification
DATALOADER_KEY = 'dataloader'
DATASET_KEY = 'dataset'
SIZE_FRACTION_KEY = 'size_fraction'
BATCH_SIZE_KEY = 'batch_size'
IMAGE_SIZE_KEY = 'image_size'
