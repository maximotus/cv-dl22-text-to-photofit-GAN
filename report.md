# Highly descriptive text-to-face generation to synthesize authentic faces (photofits for criminology purposes) via GANs

TODO maybe change text-to-face to vector-to-face? @SchubertDaniel

## Structure

1. Introduction (DONE)
   1. The Goal of Photofit Creation using GANs (Max) (DONE)
   2. Related Work (Daniel) (DONE)
2. Main
   1. Suitable Datasets (Max) (DONE)
      1. Overview (DONE)
      2. Our Decision (DONE)
   2. Framework
      1. Architecture / Structure (Max) (TODO)
      2. CDCGAN (Max) (TODO)
      3. tediGAN (Daniel) (DONE)
      4. Metrics (Daniel) (DONE)
      5. Experiments
         1. Configuration (Max) (TODO)
         2. Results (Daniel) (ALMOST DONE)
            1. Report (DONE)
            2. Analysis / Discussion (ALMOST DONE)
3. Conclusion (TODO)
   1. Datasets
   2. Mode Collapse
   3. Imagesize
   4. More time + GPU-power
   5. More attributes in dataset, more specialized dataset
4. Future Work (creation of own dataset that is perfectly suitable for our task) (TODO)
5. Collaboration (TODO)
   1. Short paragraph outlining the 50/50-distribution of the workload (pair programming, etc.)
   2. Table annotating the main responsible(s) for each part of the framework

## Introduction

### Photofit Creation using GANs

Our proposed goal for the final project of the "Computer Vision & Deep Learning: Visual Synthesis" lecture was to train 
existing Generative Adversarial Network (GAN) models and fine-tune their architectures in respect to generate authentic
and unambiguous samples of real looking faces. These samples should be of such quality that they could be used as
photofits (also phantom images, i.e. pictures representing a person's memory of a criminal's face, compare 
https://dictionary.cambridge.org/de/worterbuch/englisch/photofit-picture). Even if this common task is already done
by phantom sketch artists working for the police or for lawyers, our assumption is that a GAN creating such photofits
could easily outperform every sketch artist in terms of costs, speed, accuracy and photo-realism. Thus, such a network
for the creation of photofits could be a very useful tool for criminology workers.

### Text-to-Face Synthesis

Inspired by DALL-E and DALL-E 2 (see https://openai.com/dall-e-2/), our first intention was a text-to-image approach
using two separate models – analogous to https://arxiv.org/abs/2012.03308 ??? TODO. On the one hand, we considered to
use a text-encoder model for the embedding of a continuous text describing a criminal's face into the latent space –
such that the semantics of the textual description remain intact. On the other hand, we thought about a GAN or VAE model
creating faces from those latent embeddings.

### Vector-to-Face Synthesis

However, during the execution of the project we changed our plan to only focus on the generation part because of four
crucial arguments. First, the lecture is about visual synthesis and not natural language processing, so our main focus
should be on the creation of images and not on the semantic embedding of continuous text into the latent space.
Second, training a text-encoder and a GAN respectively means twice as much calculation time which is inappropriate for
the relatively short project time. Third, in respect to our described goal (see "The Goal", TODO) we think that possible
downstream applications would benefit more if the photofit creation is conditioned by vectors with values either 0 or 1
representing the truth value (0=False, 1=True) for each descriptive attribute of an image / a face. Fourth, the attempt
of only generating phantom images based on attribute vectors is sufficient to get a proof of concept and to use such 
model for criminology purposes.

Therefore, to keep things simple and appropriate we decided to focus solely on the principle of using a GAN network –
consisting of two separate models, i.e. a discriminator / encoder and a generator / decoder (TODO source GAN Godfellow).

We finalized a more stable adaption of a classical GAN, namely a Deep Convolutional GAN (DCGAN) which explicitly  uses
convolutional and convolutional-transpose layers in the discriminator and generator respectively (TODO source DCGAN).
Since we do not just need to generate random images, but rather images that fit the vectorized description of
a criminals face we conditioned our DCGAN to also use the attribute vector as input – then it is called a Conditional
DCGAN (CDCGAN).

Moreover, we decided to re-implement another CDCGAN architecture: tediGAN (TODO source tediGAN).
TODO @SchubertDaniel short description of tediGAN analogous to the paragraph before describing a CDCGAN
Due to time constraints and some other reasons (see TODO) we were not able to finalize the re-implementation of the
tediGAN.

TODO maybe remove the following
Provided that this project or a similar one is pursued further in the future which we would very
much appreciate, a text-to-face approach could be seen as a possible adaption.

## Main

### Suitable Datasets

During our research in the project planning phase we stumbled across various datasets that could be useful to us in
terms of their properties. Some examples of datasets containing faces and corresponding descriptions or attributes are:

- a [celebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- a [celebA HQ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html#:~:text=CelebAMask%2DHQ%20is%20a%20large,facial%20attributes%20corresponding%20to%20CelebA.)
- a [LFW](http://vis-www.cs.umass.edu/lfw/)
- a [MAAD-Face](https://github.com/pterhoer/MAAD-Face)

Nevertheless, these datasets are not made for criminology purposes, and so they do not perfectly fit our approach of
generating photofits. This has two main reasons.

First, baseline face datasets are mainly constructed for face recognition applications. On the one hand, as the authors
of [MAAD-Face](https://github.com/pterhoer/MAAD-Face) outline, this leads to the consequence that datasets like
[celebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or [LFW](http://vis-www.cs.umass.edu/lfw/) indeed contain a
large amount of face images, but struggle with the overall annotation correctness and the total number of attributes.
On the other hand, [MAAD-Face](https://github.com/pterhoer/MAAD-Face) aims to be better in those terms by merging
face image datasets with their attribute annotations together and check their correctness by a human evaluation.

Second, as we already expected beforehand and was confirmed during the project execution, such relatively low numbers of
distinctive attributes (compare table, TODO) would not fit the demand for accuracy needed for phantom image creation.
Moreover, considering the 40 attributes of [celebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) one can see that
there is some redundancy and incompleteness within – e.g. one extra attribute for each Black_Hair, Blond_Hair,
Brown_Hair, and Gray_Hair, but there is no attribute like  Red_Hair.

TODO table with dataset stats

### Used Dataset

Comparing the statistics of the datasets from above, we concluded to initially use a set that has a good trade-off
between the total number of face images and the total number of distinctive attributes. Even if they are not perfectly
fitted for criminology purposes, our assumption is that if the concept of attribute-conditioned face generation works on
one of these datasets, it will also work on more accurate datasets that could be developed especially for the task of
photofit creation in the future. And especially, it will work better on a better suited dataset.

Even if [MAAD-Face](https://github.com/pterhoer/MAAD-Face) aims to be better than
[celebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [LFW](http://vis-www.cs.umass.edu/lfw/), we decided to
use [celebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). [MAAD-Face](https://github.com/pterhoer/MAAD-Face)
has too many images as if we could manage to train our GAN on it in the give time frame of the project and
[LFW](http://vis-www.cs.umass.edu/lfw/) has too few images.



TODO citations

TODO problem to keep in mind: lack in number of attribute annotations and the overall annotation correctness, see
https://github.com/pterhoer/MAAD-Face

The input text should include descriptive criteria,
e.g. pointy nose, bald, blue eyes, wide mouth, long eyebrows or curly hair.

The first part will be to classifiy / detect important attributes from the text – which is given as an accurate description (i.e. one or more sentences) – and embed them into the latent feature space.

The second part will be the generative part using a VAE or GAN to synthesize images from the textual embeddings.

This task is interesting due to the real use case, that if it works it could support the work of police workers. Also, pretrained networks are not available and training/ building upon existing algorithms and trying to achieve the same or a higher baseline is a challenge we look forward to.


## Related Work
The number of papers concering the same topic as ours, text-to-face generation from attributes, is small. We found 17 papers based on a broad range of keywords. 
The papers can be divided into two groups. One working with photofits/ forensic or composite sketches and the other with attribute guided face generation.
The first group is mainly focused on generating images from those sketches [5,6,7]. This is not what we intended to do. 
The other group employs networks to generate faces from attributes which is a matches our goal [1,8-19].
We selected two papers which give relevant information for our project. First "TediGAN: Text-Guided Diverse Face Image Generation and Manipulation" by Xia et al. published in 2021 [1]. The second paper is "Attribute-Guided Sketch Generation" by Tang et al. published in 2019 [4], which is the only paper bringing both aspects together.

## tediGAN
TediGAN is a GAN and Framework proposed by Xia et al. To generate their images they use a inverted pretrained StyleGAN. The Framework includes multiple options to choose between layers and StyleGANs. Unfortunately neither the git-repository nor their paper provides clear information which was their final and best version. The framework also uses config files in an incomprehensible way. We tried to implement the network into our framework but couldn't get it to start training. Problems we encountered were the configs, which we replaced by one single choice. They also wrote certain layers in C++ and Cuda which we initially struggled with but in the end got to work. But then we encountered a dimension error which was weird due to all shapes matching one another. To resolve the dimension error we logged and followed the flow of the images in the fit() method. To get further information about the configs we contacted the authors but never got an answer.
To check for implementation errors we also cloned their repository and tried executing their proposed way to train with their framework. It failed to start due to missing config options.

## metrics
Regarding the metrics we orientated us among the most frequently used ones from the papers we read and chose the four most relevant. 
In regard of image generation normal metrics, like accuracy, are very relative and should not be used due to their lack of information value. In most ML context this would not apply. When training a discriminator to decide if a picture is from a real dataset or generated from the GAN's generator, the scores most of the time do not result in "good" as in real images. A human could easily differentiate both. 
So for image generation and their realness one should use one of the following metrics for evaluating the new images on the overall similarity: Fréchet Inception Distance (FID), patch similarity (Learned Perceptual Image Patch Similarity, LPIPS), Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) or a metric using a discriminator specifically trained for this task, where you know the results are satisfactory.
FID calculates the Fréchet distance is originally used for the distance between curves of functions but can also be used for the probability distribution, in our case two datasets [3]. $$d^2 = |\mu_X - \mu_Y|^2 + tr(\sum_X + \sum_Y - 2 (\sum_X \sum_Y)^{\frac{1}{2}})$$
One way to check patch similiarity between two images is using LPIPS. This metric is based on comparing similarity of activations in a predefined network. A lower score is better [20]. 
To check overall spatial image quality one could use BRISQUE. It checks for e.g. noise or blurriness. 
Kynkäänniemi et al. propose an improved precision and recall framework, which can additionally to precision and recall scores also calculate a realism score. They use a StyleGAN to evaluate a set of images [2].
Some papers also use humans to give feedback on realness of generated images, which can be unreliable and the number of samples to review is limited [1].

In our framework we implemented FID, LPIPS and BRISQUE. Kynkäänniemi's metric is implemented in an outdated version of TensorFlow. While implementing each metric a few key differences appeared which do not get clarified by any paper: e.g. LPIPS calculate the similarity between two pictures. But are they chosen at random or is an order selected in the beginning and then those images get compared? We choose to generate images based on the same attribute-vectors and compare those to one another.

## Results
We planned the training process to run every model variation at least once for 100 epochs on a quarter of the celeba dataset. Each Training run took between 9 and 11 hours. After this preliminary phase we looked at the generated images, loss, accuracy and metrics of our networks. We then decided that a dropout of 0.3 or 0.5 and spectral convolution layer were beneficial. Thus we ran those on the entire dataset size. Those runs took 12 hours on our system. We also tried out running a model for 200 epochs but noticed mode collapse happened every time. Mode collapse also happend when restarting on an epoch without collapse.

### Report
Percieved realness is an intuitve score between 0 and 5: 0 just noise, 1 shape recognizeable, ge2 = can recognize faces, ge3= face with noise, ge4=face with small artifacts, 5=real faces without errors. DS_size means Dataset size.
| network                                  | accuracy_real accuracy_fake_before_disc accuracy_fake_after_disc | loss_real loss_fake loss_gan | FID     | LPIPS | BRISQUE | percieved realness (between 0 and 5) |
|------------------------------------------|------------------------------------------------------------------|------------------------------|---------|-------|---------|--------------------------------------|
| dropout=0, spectral=False, DS_size=1/4   | 0.994 0.006 0.0007                                               | 0.0072 0.0074 13.868         | 339.957 | 0.379 | 16.946  | 1                                    |
| dropout=0, spectral=True, DS_size=1/4    | 0.968 0.031 0.009                                                | 0.046 0.411 6.815            | 139.744 | 0.161 | 45.917  | 2.5                                  |
| dropout=0.2, spectral=False, DS_size=1/4 | 0.894 0.105 0.062                                                | 0.210 0.211 6.640            | 124.138 | 0.156 | 31.736  | 2.5                                  |
| dropout=0.2, spectral=True, DS_size=1/4  | 0.969 0.028 0.062                                                | 0.0418 0.0356 6.6371         | 168.288 | 0.265 | 44.231  | 2                                    |
| dropout=0.3, spectral=False, DS_size=1/4 | 0.826 0.180 0.129                                                | 0.393 1.003 3.766            | 139.830 | 0.294 | 29.432  | 1                                    |
| dropout=0.3, spectral=True, DS_size=1/4  | 0.952 0.046 0.022                                                | 0.0612 0.0616 5.7318         | 173.991 | 0.205 | 41.228  | 1                                    |
| dropout=0.3, spectral=True, DS_size=1    | 0.999 7.308 7.210                                                | 1.12 0.0 44.171              | 210.512 | 0.271 | 20.791  | 1                                    |
| dropout=0.5, spectral=False, DS_size=1/4 | 0.506 0.501 0.496                                                | 0.750 1.737 0.768            | 145.371 | 0.327 | 73.911  | 1                                    |
| dropout=0.5, spectral=True, DS_size=1/4  | 0.821 0.179 0.131                                                | 0.263 0.262 3.003            | 141.477 | 0.353 | 45.244  | 2                                    |
| dropout=0.5, spectral=True, DS_size=1    | 0.929 0.070 0.048                                                | 0.108 0.107 5.587            | 135.339 | 0.213 | 64.889  | 1.5                                  |

[supposedToBeGoodNetworkGraph](/results/cdcgan04 dropout=0.5 spectral=false datasetsize=quarter-acc-plot.png)
[supposedToBeBadNetwork](/results/cdcgan04 dropout=0.5 spectral=true datasetsize=full-acc-plot.png)


### Analysis / Discussion
When deciding which network was the best you can proceed based on statistics, on the proposed metrics or visually judgeing the generated images per epoch and foremost the last epoch. 
One could also proceed based on theoretically taught metrics, e.g. generator accuracy and discriminator accuracy should meet at 0.5 or at least converge against each other. In this case the network with spectral convolution and dropout value being 0.3 is supposed to be the best one. But even on first glance every human would be able to differentiate between real and fake images. 
Running metrics proposed in the metrics section on trained networks resulted in some mismatching images even compared to the generated ones in the last epoch. When going on the metrics FID is probably the most meaningful. The best network according to this metric would be with dropout 20% and without a spectral convolutional layer. Some generated images are just noise. Every now and then you can recognize parts of a face but the rest is still just random artifacts. So the results are only partly acceptable. The network with 0% dropout without a spectral layer is correctly the worst variant with the highest FID score of 339.957 and images looking like weird color sprinkles [worstImage](/results/29.jpg)
When looking through all images of every last epoch the best network is tied between (dropout=50%, spectral=True, DS_size=1/4) and (dropout=0%, spectral=True, DS_size=1/4). But even these images still have some artifacts in the image or on the face, the images are highly noisy around the face. Sometimes the network tries to generate two faces into one[twoFacesInOne](/results/predef_img_max_7_3specfull_85.png). Also our networks seemed unable to learn the difference between certain attributes. E.g. they are not able to generate blonde people [blonde](/results/predef_img_max_2_spectral2quarter99.png) or they put sunglasses on people [sunglasses](/results/predef_img_max_2_spectral4full.png). The sunglasses phenomenon happend with a constant c-vector which didn't specify it and the glasses would appear and disappear epoch-wise.
A relative good result is [relGood](/results/good-4-spectral-full.png) and [relGood2](/results/predef_img_daniel_1_spectral4quarter.png). But even those fake images are distinguishable compared to the used dataset.


# Sources:
- [1] "TediGAN: Text-Guided Diverse Face Image Generation and Manipulation" by Xia et al., published in 2021 [tediGAN-paper](https://openaccess.thecvf.com/content/CVPR2021/html/Xia_TediGAN_Text-Guided_Diverse_Face_Image_Generation_and_Manipulation_CVPR_2021_paper.html)
- [2] "Improved Precision and Recall Metric for Assessing Generative Models" by Kynkäänniemi et al., published in 2019 [Precision and Recall metric paper](https://proceedings.neurips.cc/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html)
- [3] "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" by Heusel et al., published in 2017 [FID paper](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html)
- [4] "Attribute-Guided Sketch Generation" by Tang et al., published in 2019 [paper](https://ieeexplore.ieee.org/abstract/document/8756586)
- [5] "Matching Composite Sketches to Face Photos: A Component-Based Approach" by Han et al., published in 2012 [paper](https://ieeexplore.ieee.org/abstract/document/6359918)
- [6] "Face sketch-to-photo transformation with multi-scale self-attention GAN" by Lei et al., published in 2020 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231220301995)
- [7] "Evaluating the Performance of Face Sketch Generation using Generative Adversarial Networks" by Sannidhan et al., published in 2019 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865519302831)
- [8] "Attribute-Guided Face Generation Using Conditional CycleGAN" by Lu et al., published in 2018, [paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Yongyi_Lu_Attribute-Guided_Face_Generation_ECCV_2018_paper.html)#
- [9] "Dynamic Facial Expression Generation on Hilbert Hypersphere with Conditional Wasserstein Generative Adversarial Nets" by Otberdout et al., published in 2020 [paper](https://ieeexplore.ieee.org/abstract/document/9117185)
- [10] "How are attributes expressed in face DCNNs?" by Dhar et al., published in 2019 [paper](https://ieeexplore.ieee.org/abstract/document/9320290)
- [11] "Attributes Aware Face Generation with Generative Adversarial Networks" by Yuan et al., published in 2020 [paper](https://ieeexplore.ieee.org/abstract/document/9412022)
- [12] "FRONTAL FACE GENERATION FROM MULTIPLE POSE-VARIANT FACES WITH CGAN IN REAL-WORLD SURVEILLANCE SCENE" by Cheng et al., published in 2018 [paper](https://www.researchgate.net/publication/327805981_Frontal_Face_Generation_from_Multiple_Pose-Variant_Faces_with_CGAN_in_Real-World_Surveillance_Scene)
- [13] "Semi-supervised Adversarial Learning to Generate
Photorealistic Face Images of New Identities from 3D Morphable Model" by Gecer et al., published in 2018 [paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Baris_Gecer_Semi-supervised_Adversarial_Learning_ECCV_2018_paper.html)
- [14] "cGAN Based Facial Expression Recognition for
Human-Robot Interaction" by Deng et al., published in [paper](https://ieeexplore.ieee.org/abstract/document/8606936)
- [15] "Generating Face Images with Attributes for Free" by Liu et al., published in [paper](https://ieeexplore.ieee.org/abstract/document/9146375)
- [16] "Semi-supervised FusedGAN for Conditional Image Generation" by Bodla et al., published in 2018 [paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Navaneeth_Bodla_Semi-supervised_FusedGAN_for_ECCV_2018_paper.html)
- [17] "Conditional Image Generation with PixelCNN Decoders" by Oord et al., published in [paper](https://proceedings.neurips.cc/paper/2016/hash/b1301141feffabac455e1f90a7de2054-Abstract.html)
- [18] "Conditional generative adversarial nets for convolutional face generation" by Gauthieret al., published in 2014 [paper](https://www.foldl.me/uploads/2015/conditional-gans-face-generation/paper.pdf)
- [19] "ELEGANT: Exchanging Latent Encodings with GAN for Transferring Multiple Face Attributes" by Xiao et al., published in 2018 [paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Taihong_Xiao_ELEGANT_Exchanging_Latent_ECCV_2018_paper.html)
- [20] "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" by Zhang et al., published in 2018 [LPIPS-paper](https://arxiv.org/abs/1801.03924)


## 


Gliederung

Einleitung: Was wir erreichen wollte proposal umfangreicher
Related work -Daniel
Datensets die es gibt, welches wir genommen haben+ warum -max
(implementation) Our work: 
    Framework, cdcGAN - Max
    + tediGAN, Metrics -daniel
Experimente: 
        configs, max
            ergebnisse daniel
            analysieren -zusammen

Conclusion:
    Datensets, 
    Mode Collapse, 
    Imagesize, 
    conclusion, 
    more time + GPU-power, 
    more attributes in dataset, more specialized dataset 
Future Works.
    Was ein Datenset bräuchte für diese Aufgabe, wie ausschauen, attribute

Table 
    Capitel Author