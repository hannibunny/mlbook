# Difussion Models

* Author: Johannes Maucher
* Last update: 21.12.22 !!Draft!!

## Introduction

Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) are two generative Deep Learning models, which can be applied to generate new content. The new content is typically a new instance of a category of which many other instances have been used for training.  

GANs and VAEs suffer from drawbacks, e.g. for GANs it is usually challenging to find a good hyperparameter-configuration such that the training-process is stable. Moreover, the diversity of the generated new content is often quite low. For VAEs it is challenging to find a good loss-function.

Diffusion Models are another type of generative Deep Learning. The concept of diffusion is borrowed from thermodynamics. Gas molecules diffuse from high density to low density areas. During this process entropy increases. In information theory, this entropy-increase corresponds to a loss of information. Such an information loss can be caused by adding noise. 

The idea of Probabilistic Denoising Diffusion models, as introduced in {cite}`sohl-dickstein15` and significantly improved in {cite}`Ho20`, is to 
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

For the backward diffusion process a conditional probability distributions must be learned. For this a neural network is applied. The forward diffusion process doesn't require a training.

The range of applications of Diffusion models is similar as for GANs and VAEs, e.g. image generation, image modification, text-driven image generation, image-to-image translation, superresolution, image segmentation or 3D shape generation and completion.

## Forward Diffusion Process

Starting form real data $x_0$, which is sampled from a known data distribution $q(x_0)$, the forward diffusion process adds in each step a small amount of Gaussian noise to the current image-version. In this way a sequence of increasingly noisy images $x_0,x_1,x_2,\ldots,x_T$ is generated. For $T \rightarrow \infty$, $x_T$ is equivalent to an isotropic Gaussian distribution.

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/diffusionforward.png
---
align: center
width: 600pt
name:  diffusionforward
---
Forward Process: Gradually add noise. This direction of the process is known. No learning required.

```

The conditional probability distribution $q(x_t|x_{t-1})$ for $x_t$, given $x_{t-1}$ is a Gaussian distribution with 

* mean: $\sqrt{1-\beta_t}x_{t-1}$
* variance: $\beta_t \mathbf{I}$

The set $\lbrace \beta_t \in (0,1)\rbrace_{t=1}^T$ defines a *variance schedule*, i.e. how much noise is added in each step. $\mathbf{I}$ is the identity matrix:

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t \mathbf{I}),
$$   

and

$$
q(x_{1:T}|x_{0}) = \prod_{t=1^T} q(x_t|x_{t-1})
$$


In order to generate the noisy image version $x_t$ in an arbitrary step $t$ it is not necssary to execute $t$ steps in sequence. Instead $x_t$ can be generated as follows:

$$
x_t = \sqrt{\overline{\alpha}_t} x_0 + \sqrt{(1-\overline{\alpha}_t)} \epsilon, \mbox{ where } \epsilon \sim \mathcal{N}(0,\mathbf{I})
$$

## Backward Diffusion Process