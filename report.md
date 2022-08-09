# Highly descriptive text-to-face generation to synthesize authentic faces (photofits for criminology purposes) via GANs

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

## Photofit Creation using GANs

Our proposed goal for the final project of the "Computer Vision & Deep Learning: Visual Synthesis" lecture was to train 
existing Generative Adversarial Network (GAN) models and fine-tune their architectures in respect to generate authentic
and unambiguous samples of real looking faces. These samples should be of such quality that they could be used as
photofits (also phantom images, i.e. pictures representing a person's memory of a criminal's face, compare 
https://dictionary.cambridge.org/de/worterbuch/englisch/photofit-picture). Even if this common task is already done
by phantom sketch artists working for the police or for lawyers, our assumption is that a GAN creating such photofits
could easily outperform every sketch artist in terms of costs, speed, accuracy and photo-realism.

TODO

The input text should include descriptive criteria,
e.g. pointy nose, bald, blue eyes, wide mouth, long eyebrows or curly hair.

The first part will be to classifiy / detect important attributes from the text – which is given as an accurate description (i.e. one or more sentences) – and embed them into the latent feature space.

The second part will be the generative part using a VAE or GAN to synthesize images from the textual embeddings.

This task is interesting due to the real use case, that if it works it could support the work of police workers. Also, pretrained networks are not available and training/ building upon existing algorithms and trying to achieve the same or a higher baseline is a challenge we look forward to.


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