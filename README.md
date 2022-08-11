# cv-dl22-text-to-photofit-GAN
Highly descriptive text-to-face generation to sythesize authentic faces (photofits for criminology purposes) via a GAN. Main project of the  teaching event "Computer Vision and Deep Learning: Visual Synthesis" in the summer of 2022 at LMU Munich.

# Framework
This Framework currently provides two GAN-Models – cDCGAN and TediGAN – that are easy to train, evaluate and use to generate photofits. It is implemented in PyTorch and highly configurable to accomodate many test cases. Due to its architecture other models, datasets and metrics can easily be added. We are looking forward to your pull requests.
## Models
- cDCGAN: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- TediGAN: [paper](https://arxiv.org/abs/2012.03308) [git](https://github.com/IIGROUP/TediGAN)
## Datasets
- celebA [website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
## Metrics
- FID [paper](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.htm)
- LPIPS [paper](https://arxiv.org/abs/1801.03924)
- BRISQUE [paper](https://ieeexplore.ieee.org/document/6272356)
# How to use
## Requirements
- Python 3.9
- see requirements-pip.txt or requirements-conda.txt
## Configuration
'''yaml
mode: train # (train / eval / gen)
log_level: 20 # CRITICAL = 50, ERROR = 40, WARNING = 30, INFO = 20, DEBUG = 10, NOTSET = 0
device: auto # (cuda / cpu / auto)
experiment_path: ..\experiments
epochs: 500
num_imgs: 20 # how many randomly generated images should be saved
predefined_images: {max: [0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1], daniel: [0,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1]}

frequencies:
  save_freq: 1 # (i.e. how often should the model checkpoint be saved, in epochs)
  gen_freq: 1 # (i.e. how often should test images be generated, in epochs)

dataloader:
  dataset: celebA
  size_fraction: 4
  batch_size: 128
  image_size: 64

model:
  name: CDCGAN
  #pretrained_path: ../experiments/train/template-CDCGAN-train/2022-08-10-10-08-52 # empty if start from scratch
  #start_epoch: 0 # empty if start from scratch
  criterion: BCELoss
  optimizer: Adam
  learning_rate: 0.001
  parameters:
    dropout: 0.2
    alpha: 0.1
    beta1: 0.1
    ngf: 64
    ndf: 64
    z_channels: 128
    use_spectral_norm: False
'''

## Execution
When first executing the main.py the dataset will be downloaded, often the daily quota of available downloads from the google server is reached and an error "zipfile.BadZipFile: File is not a zip file" occures. By trying every few minutes one should be able to completely download the dataset.
```bash
python main.py ../config/<CONFIG-NAME>.yaml
```
