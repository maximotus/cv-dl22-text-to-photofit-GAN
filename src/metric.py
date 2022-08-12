import subprocess
import lpips
from PIL import Image
from torchvision import transforms
import brisque
import cv2

def FID(image_folder_path1: str, image_folder_path2: str):
    # https://github.com/mseitzer/pytorch-fid
    '''
    calculates frechet distance between two datasets
    '''
    queue = "python -m pytorch_fid {} {}".format(image_folder_path1, image_folder_path2)
    x=subprocess.check_output(queue, shell=True)
    x = x.decode("utf-8").split(" ")[-1] #binary to utf-8 and choose last string 
    x = x[:-2] #cut off \r\n which we don't need
    return float(x)
    

def LPIPS(image_path1, image_path2):
    # https://github.com/richzhang/PerceptualSimilarity
    '''
    calculates the perceptual similarity between to images (generated and from dataset)
    '''
    image1 = transforms.functional.pil_to_tensor(Image.open(image_path1))
    image2 = transforms.functional.pil_to_tensor(Image.open(image_path2))
    image2 = transforms.Resize((64, 64))(image2)

    loss_fn = lpips.LPIPS(net='alex')
    d = loss_fn.forward(image1, image2)
    return d.mean().detach().item()


def brisque_score(image_path):    
    # pip install pybrisque
    '''
    images have to be numpy arrays
    '''
    image = cv2.imread(image_path)
    brisq = brisque.BRISQUE()
    score = brisq.get_score(image)
    return score


if __name__ == "__main__":
    print(LPIPS("../experiments/eval/cdcgan-01/2022-07-22-23-18-31/img/0.jpg","../experiments/eval/cdcgan-01/2022-07-22-23-18-31/img/0.jpg"))