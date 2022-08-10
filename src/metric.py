import os
import lpips
from PIL import Image
import imquality.brisque as brisque

def FID(dataset1_path: str, dataset2_path: str):
    # https://github.com/mseitzer/pytorch-fid
    '''
    calculates frechet distance between two datasets
    '''
    os.system("python -m pytorch_fid {} {}".format(dataset1_path, dataset2_path))

def LPIPS(image1, image2):
    # https://github.com/richzhang/PerceptualSimilarity
    '''
    calculates the perceptual similarity between to images (generated and from dataset)
    '''
    loss_fn = lpips.LPIPS(net='alex')
    d = loss_fn.forward(image1, image2)
    print(d)

def brisque_score(path_to_image: str):
    # https://pypi.org/project/image-quality/
    image = Image.open(path_to_image)

    score = brisque.score(image)
    print(score)

def precision_and_recall_score(path: str):
    # https://arxiv.org/abs/1904.06991
    # https://github.com/kynkaat/improved-precision-and-recall-metric
    # (pip install git+https://github.com/kynkaat/improved-precision-and-recall-metric.git) <- doesnt work you have to clone it yourself
    os.system("python run_metric.py --data_dir {} --realism_score".format(path))