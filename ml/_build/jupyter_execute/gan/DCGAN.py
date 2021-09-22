#!/usr/bin/env python
# coding: utf-8

# #  Generative Adversarial Nets (GAN)
# * Author: Johannes Maucher
# * Last Update: 07.01.2019
# 
# References:
# 
# 1. [Ian Goodfellow et al: Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
# 2. [A. Radford and L. Metz: Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/pdf/1511.06434.pdf)

# ## Idea GANs
# 
# In 2014 GANs have been introduced by [Ian Goodfellow et al](https://arxiv.org/pdf/1406.2661.pdf). Since then they are one of the hottest topics in deeplearning research. GANs are able to generate synthetic data that looks similar to data of a given trainingset. In this way artifical images, paintings, texts, audio or handwritten digits can be generated. 
# 
# On an abstract level the idea of GANs can be described as follows: A counterfeiter produces fake banknotes. The police is able to discriminate the fake banknotes from real ones and it provides feedback to the counterfeiter on why the banknotes can be detected as fake. This feedback is used by the counterfeiter in order to produce fake, which is less distinguishable from real bankotes. After some iterations of producing better but not sufficiently good fake the counterfeiter is able to produce fake, which can not be discriminated from real banknotes.
# 
# A GAN consists of two models: 
# 
# 1. The **discriminator** is the police. It learns to discriminate real data from artificially generated fake data.
# 2. The **generator** is the counterfeiter, which learns to generate data, that is indistinguishable from real data.
# 
# 
# ![GAN](./Pics/GAN.png)

# ## Example Applications of GANs
# ### Image-to-Image Translation
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/ganCelebrities.PNG" width="450" class="center">
# 
# 
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/ganTranslation.PNG" width="450" class="center">
# 
# ### Automatic Anime Character Creation
# 
# [Y. Jin et al: Towards the Automatic Anime Characters Creation with Generative Adversarial Networks](https://arxiv.org/pdf/1708.05509.pdf)
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/animeCharacters.png" width="450" class="center">
# 

# In this notebook a GAN is designed, which learns to generate handwritten numbers between 0 and 9, like the ones, given in the MNIST dataset. In this case the real data is the MNIST dataset, which contains 70000 greyscale images of size 28x28, 7000 images for each of the 10 digits.
# 
# The Discriminator model is just a binary classifier, whose task is to distinguish fake, from real images. The fake images are produced by the Generator model. At it's input the Generator model receives 1-dimensional vectors of length 100, whose components are uniformly distributed random float-values between -1 and 1. The discriminator model is learned such that the *cross-entropy-loss* for discrimating real from fake images is minimized by Stochastic Gradient Descent (SGD). The Generator model is learned such that the generated fake data is classified as real-data by the discriminator.

# ## Keras implementation of GAN, which generates MNIST-type handwritten digits 
# 
# Below a Deep Convolutional GAN (DCGAN) as introduced in [A. Radford and L. Metz: Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/pdf/1511.06434.pdf) is implemented. This type of GAN applies a Convolutional Neural Net (CNN) for the Generator- and the Discriminator model.

# In[1]:


#!pip install tqdm


# In[2]:


import tensorflow.compat.v1.keras.backend as K # see https://stackoverflow.com/questions/61056781/typeerror-tensor-is-unhashable-instead-use-tensor-ref-as-the-key-in-keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


# In[19]:


from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm  #this package is used to show progress in training loops
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# The MNIST-handwritten digits dataset is available in the Keras datasets module and can be accessed as follows:

# In[20]:


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# In[21]:


X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# In[22]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(X_train.shape)
print(X_test.shape)

X_train = X_train.astype('float32')
print(X_train[0,:,:,0])


# The values of all input-images range from 0 to 255. Next, a rescaling to the range $[-1,1]$ is performed. This is necesarry, because the output-layer of the Generator model applies a tanh-activation, which has a value rane of $[-1,1]$. 

# In[23]:


X_train = (X_train - 127.5) / 127.5

print(X_train[0,:,:,0])


# The first 9 real images in the training dataset are plotted below: 

# In[24]:


#plt.subplot(3,3,1)
for i in range (9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i+1, :, :, 0], cmap='gray')


# ### Generator Model
# 
# The Generator model receives 1-dimensional vectors of length 100 with uniformly distributed float values from -1 to 1 at it's input. The first layer is a dense layer with 128x7x7 neurons. The output of this dense layer is reshaped into 128 channels, each of size 7x7.
# Using a deconvolution filter of size 5x5 (realized by an UpSampling2D-layer, followed by a Conv2D-layer) 64 channels of size 14x14 are generated in the next layer. Finally, these 64 channels are processed by another deconvolution layer with filters of size 5x5 into a single-channel output of size 28x28. 
# 
# In between layers, batch normalization stabilizes learning. The activation function after the dense- and the first deconvolution-layer is a LeakyReLU. The output deconvolution-layer applies tanh- activation.

# In[25]:


generator = Sequential([
        Dense(128*7*7, input_dim=100),
        LeakyReLU(0.2),
        BatchNormalization(),
        Reshape((7,7,128)),
        UpSampling2D(),
        Conv2D(64, (5, 5), padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(1, (5, 5), padding='same', activation='tanh')
    ])


# In[26]:


generator.summary()


# ### Discriminator Model
# The discriminator is also a CNN. For this experiment on the MNIST dataset, the input is an image (single channel) of size 28x28. The sigmoid output is an indicator for the probability that the input is a real image.
# I.e. if the output value is close to 1, the input is likely a real image, if the output is close to 0, the input is likely a fake. 
# The first convolution layer applies 5x5 filters in order to calculate 64 features. The second convolution layer applies 5x5 filters to the 64 input channels and calculates 128 feature maps. The last dense layer has a single neuron with a sigmoid activation for binary classification.  
# 
# The difference from a typical CNN is the absence of max-pooling in between layers. Instead, a strided convolution is used for downsampling. The activation function used in each CNN layer is a leaky ReLU. A dropout of 0.3 between layers is used to prevent overfitting and memorization. 

# In[27]:


discriminator = Sequential([
        Conv2D(64, (5, 5), strides=(2,2), input_shape=(28,28,1), padding='same'),
        LeakyReLU(0.2), 
        Dropout(0.3),
        Conv2D(128, (5, 5), strides=(2,2), padding='same'),
        LeakyReLU(0.2),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])


# In[28]:


discriminator.summary()


# Next for the Generator- and the Discriminator the training algorithm is defined. Both models are trained by minimizing the *binary cross-entropy* loss. For both models the [*Adam* algorithm](https://arxiv.org/pdf/1412.6980.pdf) is applied. 
# In contrast to standard SGD, Adam applies individual learning-rates for each learnable parameter and adpats these learning-rates individually during training.

# In[29]:


generator.compile(loss='binary_crossentropy', optimizer=Adam())


# In[30]:


discriminator.compile(loss='binary_crossentropy', optimizer=Adam())


# ### Build GAN by combining Generator and Discriminator
# Now, since Generator- and Discriminator models are defined, the overall adversarial model can be build, by simply stacking these models together. The output of the generator is passed to the input of the discriminator:

# In[31]:


discriminator.trainable = False
ganInput = Input(shape=(100,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=Adam())


# In[32]:


gan.summary()


# ### Train GAN
# Finally, a function `train()` is implemented, which defines the overall training process. The flow-chart of this function is depicted below: 
# 
# ![GANtrainingProcess](./Pics/GANtrainingProcess.png)

# In[33]:


def train(epoch=10, batch_size=128):
    batch_count = X_train.shape[0] // batch_size
    
    for i in range(epoch):
        for j in tqdm(range(batch_count)):
            # Random input for the generator
            noise_input = np.random.rand(batch_size, 100)
            
            # select batchsize random images from X_train
            # these are the real images that will be passed to the discriminator
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
            
            # Predictions from the generator:
            predictions = generator.predict(noise_input, batch_size=batch_size)
            
            # the discriminator takes in the real images and the generated images
            X = np.concatenate([predictions, image_batch])
            
            # labels for the discriminator
            y_discriminator = [0]*batch_size + [1]*batch_size
            
            # train the discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_discriminator)
            
            # train the generator
            noise_input = np.random.rand(batch_size, 100)
            y_generator = [1]*batch_size
            discriminator.trainable = False
            gan.train_on_batch(noise_input, y_generator)


# Train for 30 epochs. Depending on your hardware this process may take a long time. In my experiments on CPU about 20min per epoch and on GPU 12-15sec per epoch.

# In[34]:


train(30, 128)


# Save the weights, learned after these 30 epochs.

# In[35]:


generator.save_weights('gen_30_scaled_images.h5')
discriminator.save_weights('dis_30_scaled_images.h5')


# Start from the weight learned in 30 epochs and continue training for another 20 epochs:

# In[42]:


train(30, 128)


# Save the weights, learned after 60 epochs:

# In[43]:


generator.save_weights('gen_60_scaled_images.h5')
discriminator.save_weights('dis_60_scaled_images.h5')


# ### Generate new numbers from GAN learned over 30 epochs
# Load the learned weights and apply them for generating fake images:

# In[44]:


generator.load_weights('gen_30_scaled_images.h5')
discriminator.load_weights('dis_30_scaled_images.h5')


# In[45]:


def plot_output():
    try_input = np.random.rand(100, 100)
    preds = generator.predict(try_input)

    plt.figure(figsize=(10,10))
    for i in range(preds.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(preds[i, :, :, 0], cmap='gray')
        plt.axis('off')
    
    # tight_layout minimizes the overlap between 2 sub-plots
    plt.tight_layout()


# In[46]:


plot_output()


# ### Generate new numbers from GAN learned over 60 epochs
# Load the learned weights and apply them for generating fake images:

# In[47]:


generator.load_weights('gen_60_scaled_images.h5')
discriminator.load_weights('dis_60_scaled_images.h5')


# In[48]:


plot_output()


# In[ ]:




