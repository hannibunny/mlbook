# Difussion Models

* Author: Johannes Maucher
* Last update: 21.12.22 !!Draft!!

Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) are two generative Deep Learning models, which can be applied to generate new content. The new content is typically a new instance of a category of which many other instances have been used for training.  

GANs and VAEs suffer from drawbacks, e.g. for GANs it is usually challenging to find a good hyperparameter-configuration such that the training-process is stable. Moreover, the diversity of the generated new content is often quite low. For VAEs it is challenging to find a good loss-function.

Diffusion Models are another type of generative Deep Learning. The concept of diffusion is borrowed from thermodynamics. Gas molecules diffuse from high density to low density areas. During this process entropy increases. In information theory, this entropy-increase corresponds to a loss of information. Such an information loss can be caused by adding noise. 

The idea of Probabilistic Denoising Diffusion models, as introduced in {cite}`Ho20` is to 
1. gradually add noise to a given input in the **forward diffusion process**
2. learn the corresponding inverse process in a **backward diffusion process** and apply this inverse process to generate new instances from noise.

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/denoisingdiffusionprocess.png
---
align: center
width: 600pt
name:  diffusionprocess
---
Source: [https://cvpr2022-tutorial-diffusion-models.github.io](https://cvpr2022-tutorial-diffusion-models.github.io)

```

The range of applications of Diffusion models is similar as for GANs and VAEs, e.g. image generation, image modification, text-driven image generation, image-to-image translation, superresolution, image segmentation or 3D shape generation and completion.

