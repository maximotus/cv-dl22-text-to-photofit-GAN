# Highly descriptive text-to-face generation to synthesize authentic faces (photofits for criminology purposes) via GANs

TODO maybe change text-to-face to vector-to-face? @SchubertDaniel

## Structure

1. Introduction
   1. The Goal of Photofit Creation using GANs (Max)
   2. Related Work (Daniel)
2. Main
   1. Suitable Datasets (Max)
      1. Overview
      2. Our Decision
   2. Framework
      1. Architecture / Structure (Max)
      2. CDCGAN (Max)
      3. tediGAN (Daniel)
      4. Metrics (Daniel)
      5. Experiments
         1. Configuration (Max)
         2. Results (Daniel)
            1. Report
            2. Analysis / Discussion
3. Conclusion
   1. Datasets
   2. Mode Collapse
   3. Imagesize
   4. More time + GPU-power
   5. More attributes in dataset, more specialized dataset
4. Future Work (creation of own dataset that is perfectly suitable for our task)
5. Collaboration
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

- TODO

The input text should include descriptive criteria,
e.g. pointy nose, bald, blue eyes, wide mouth, long eyebrows or curly hair.

The first part will be to classifiy / detect important attributes from the text – which is given as an accurate description (i.e. one or more sentences) – and embed them into the latent feature space.

The second part will be the generative part using a VAE or GAN to synthesize images from the textual embeddings.

This task is interesting due to the real use case, that if it works it could support the work of police workers. Also, pretrained networks are not available and training/ building upon existing algorithms and trying to achieve the same or a higher baseline is a challenge we look forward to.


## Related Work
The number of papers concering the same topic as ours, text-to-face generation from attributes, is small. We found 17 papers based on a broad range of keywords. 
The papers can be divided into two groups. One working with photofits/ forensic or composite sketches and the other with attribute guided face generation.
The first group is mainly focused on generating images from those sketches. This is not what we intended to do. 
The other group employs networks to generate faces from attributes which is a matches our goal.
We selected two papers which give relevant information for our project. First "TediGAN: Text-Guided Diverse Face Image Generation and Manipulation" by Xia et al. published in 2021. The second paper is "Attribute-Guided Sketch Generation" by Tang et al. published in 2019, which is the only paper bringing both aspects together.


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