import os
import queue
import subprocess
import lpips
from PIL import Image
import imquality.brisque as brisque
from torchvision import transforms
import brisque

def FID(images1: str, images2: str):
    # https://github.com/mseitzer/pytorch-fid
    '''
    calculates frechet distance between two datasets
    '''
    queue = "python -m pytorch_fid {} {}".format(images1, images2)
    # x = os.system(queue)
    # throws error AttributeError: module 'os' has no attribute 'sched_getaffinity'
    x=subprocess.check_output(queue)#, shell=True
    return x
    

def LPIPS(image1, image2):
    # https://github.com/richzhang/PerceptualSimilarity
    '''
    calculates the perceptual similarity between to images (generated and from dataset)
    '''
    loss_fn = lpips.LPIPS(net='alex')
    d = loss_fn.forward(image1, image2)
    return d.mean().detach().item()

def brisque_score(image):
    # https://pypi.org/project/image-quality/
    # image = transforms.ToPILImage()(image).convert("RGB")
    # score = brisque.score(image)
    # return score
    
    # pip install pybrisque
    brisq = brisque.BRISQUE()
    score = brisq.get_score(image)# ValueError: You can only pass image to the constructor
    return score


def precision_and_recall_score(images, _=None):
    # https://arxiv.org/abs/1904.06991
    # https://github.com/kynkaat/improved-precision-and-recall-metric
    # (pip install git+https://github.com/kynkaat/improved-precision-and-recall-metric.git) <- doesnt work you have to clone it yourself
    q = "python run_metric.py --data_dir {} --realism_score".format(images)
    return subprocess.check_output(q, shell=True)


if __name__ == "__main__":

    im1 = Image.open("../../data/celeba/img_align_celeba/000001.jpg")
    im2 = Image.open("../../data/celeba/img_align_celeba/000002.jpg")
    print(brisque_score(im1))