# Project Proposal: Schubert and Erler

# Highly descriptive text-to-face generation to sythezise authentic faces (photofits for criminology purposes) via a GAN

## Team

- Daniel Schubert, 16627792, s.schubert@campus.lmu.de
- Max Erler, 11749383, max.erler@campus.lmu.de

## Problem description

There are already some papers doing text to face generation:

- [Semantic Text-to-Face GAN-ST2FG](https://arxiv.org/pdf/2107.10756.pdf)
- [Text2FaceGAN: Face Generation from Fine Grained Textual Descriptions](https://arxiv.org/pdf/1911.11378.pdf)
- [TediGAN: Text-Guided Diverse Face Image Generation and Manipulation](https://openaccess.thecvf.com/content/CVPR2021/papers/Xia_TediGAN_Text-Guided_Diverse_Face_Image_Generation_and_Manipulation_CVPR_2021_paper.pdf)
- [TediGAN: Text-Guided Diverse Face Image Generation and Manipulation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9163356)
- [Zero-Shot Text-to-Image Generation (DALL-E)](https://arxiv.org/abs/2102.12092)

We for our part want to train and fine tune an existing model architecture from one of the papers above. The generated images should look authentic and match the level of "photofits" (or phantom images, in German "Phantombilder") of real police workers or lawyers. The text should include descriptive criteria, e.g. pointy nose, bald, blue eyes, wide mouth, long eyebrows or brown eyes.

So the first part will be to classifiy / detect important attributes from the text – which is given as an accurate description (i.e. one or more sentences) – and embed them into the latent feature space.

The second part will be the generative part using a VAE or GAN to synthesize images from the textual embeddings.

This is interesting due to building upon existing algorithms and support the work of police workers. Also, pretrained networks are not available.

## Dataset

We want to use :

- [celebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [celebA HQ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html#:~:text=CelebAMask%2DHQ%20is%20a%20large,facial%20attributes%20corresponding%20to%20CelebA.)
- [LSW](http://vis-www.cs.umass.edu/lfw/)

During the training we probably want to combine the datasets or compare each one with our chosen network. Also we want to do some research to find some datasets which maybe fit better to our task due to more descriptive labels.

## Approach

First, we want to use the model from the [Semantic Text-to-Face GAN-ST2FG](https://arxiv.org/pdf/2107.10756.pdf) paper and re-implement it with Pytorch since it out performs all the other networks in their related works comparison and we could not find a better one yet. 

Second, we want to train the unmodified model with the baseline configuration on the different datasets and also on a compostion of them. The goal would be to reproduce the baseline results.

Third, we want to "optimize" or fine tune the hyperparameters to see if we can achieve a better score than the existing model.

Fourth, we want to overthink the model based on our observations and try to optimize it in respect to our task – generating authentic photofits.

Moreover, we want to adapt and optimize the textual embeddings to make them appropriate for face descriptions. If we find a better dataset with more accurate labels we want to adapt the text embeddings to those labels and also optimize the GAN as described above.

## Evaluation and Expected Resutls

We will evaluate the baseline and our adapted networks with exiting metrics like FID, LPIPS, BRISQUE and Manipulative precision metric. Moreover we will try to compose a new metric that can evaluate the usability for criminology purposes.

## Hardware

We have access to a private machine. We are not sure if its enough. We consider the option to use Azure for Students or the cip pool.