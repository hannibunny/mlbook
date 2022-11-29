#!/usr/bin/env python
# coding: utf-8

# # Implementing Neural Networks with Keras
# * Author: Johannes Maucher
# * Last Update: 29.11.2022

# ## What you will learn:
# * Define, train and evaluate MLP in Keras
# * Define, train and evaluate CNN in Keras
# * Visualization of learning-curves
# * Implement cross-validation in Keras
# * Image classification based on the CIFAR-10 dataset, which is included in [Keras datasets](https://keras.io/datasets/).

# ## Imports and Configurations

# In[26]:


#!pip install visualkeras
#!pip install Pillow
#!pip install tensorflow-cpu


# In[27]:


#import tensorflow
#from tensorflow import keras


# In[28]:


from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Input,Dropout,Flatten,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import set_image_data_format
import os


# In[29]:


set_image_data_format("channels_last")


# In[30]:


import warnings
warnings.filterwarnings("ignore") 


# The following code-cell is just relevant if notebook is executed on a computer with multiple GPUs. It allows to select the GPU. 

# In[31]:


#from os import environ
#environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#environ["CUDA_VISIBLE_DEVICES"]="1"


# In this notebook the neural network shall not learn models, which already exists. This is implemented as follows. The three models (MLP and two different CNNs) are saved to the files, whose name is assigned to the variables `mlpmodelname`, `cnnsimplemodelname` and `cnnadvancedmodelname`, respectively. 
# If these files exist (checked by `os.path.isfile(filename)`) a corresponding AVAILABLE-Flag is set. If this flag is `False`, the corresponding model will be learned and saved, otherwise the existing model will be loaded from disc.

# In[32]:


modeldirectory="models/"
mlpmodelname=modeldirectory+"dense512"
cnnsimplemodelname=modeldirectory+"2conv32-dense512"
cnnadvancedmodelname=modeldirectory+"2conv32-4conv64-dense512"


# In[33]:


import os.path

if os.path.isdir(mlpmodelname):
    MLP_AVAILABLE=True
else:
    MLP_AVAILABLE=False
    
if os.path.isdir(cnnsimplemodelname):
    CNN1_AVAILABLE=True
else:
    CNN1_AVAILABLE=False
    
if os.path.isdir(cnnadvancedmodelname):
    CNN2_AVAILABLE=True
else:
    CNN2_AVAILABLE=False


# In[34]:


CNN1_AVAILABLE


# ## Access Data 
# 
# Load the Cifar10 image dataset from `keras.datasets`. Determine the shape of the training- and the test-partition.

# In[35]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[36]:


print(np.shape(X_train))
print(np.shape(X_test))


# ## Visualize Data
# 
# Viusalize the first 9 images of the training-partition, using function `imshow()` from `matplotlib.pyplot`.

# In[37]:


# create a grid of 3x3 images
plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    B=X_train[i].copy()
    #B=B.swapaxes(0,2)
    #B=B.swapaxes(0,1)
    plt.imshow(B)
# show the plot
plt.show()


# ## Preprocessing 
# Scale all images such that all their values are in the range $[0,1]$.

# In[38]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0


# Labels of the first 9 training images:

# In[39]:


print(y_train[:9])


# **Label-Encoding:** Transform the labels of the train- and test-partition into a one-hot-encoded representation. 

# In[40]:


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
num_classes=len(y_train[0,:])
print(y_train[:9,:])


# ## MLP
# ### Architecture
# In Keras the architecture of neural networks can be defined in two different ways:
# 
# * Using the `Sequential` model
# * Using the functional API
# 
# Below the two approaches are demonstrated. The first approach is simpler, but restricted to neural networks which consist of a linear stack of layers. The second approach is more flexible and allows to define quit complex network architectures, e.g. with more than one input, more than one output or with parallel branches.

# #### Network definition option1: Using the sequential model

# In[41]:


if MLP_AVAILABLE:
    model=load_model(mlpmodelname)
    print("MLP MODEL ALREADY AVAILABLE \nLOAD EXISTING MODEL")
else:
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32,3)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[42]:


import visualkeras


# In[43]:


visualkeras.layered_view(model,legend=True)


# #### Network definition option 2: Using the functional API

# In[44]:


# This returns a tensor
inputs = Input(shape=(32, 32,3))
x=Flatten()(inputs)
x=Dense(512, activation='relu')(x)
x=Dense(num_classes, activation='softmax')(x)
model2 = Model(inputs=inputs, outputs=x)
model2.summary()


# ### Define Training Parameters 
# Apply Stochastic Gradient Descent (SGD) learning, for minimizing the `categorical_crossentropy`. The performance metric shall be `accuracy`. Train the network.

# In[45]:


if not MLP_AVAILABLE:
    # Compile model
    epochs = 8
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# ### Perform Training

# In[46]:


if not MLP_AVAILABLE:
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32,verbose=False)
    model.save(mlpmodelname)
    MLP_AVAILABLE=True
else:
    print("TRAINED MODEL ALREADY AVAILABLE")


# ### Evaluation 
# Visualize the learning-curve on training- and test-data. 

# In[47]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb
sb.set_style("whitegrid")
sb.set_context("notebook")


# In[48]:


try:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    max_val_acc=np.max(val_acc)

    epochs = range(1, len(acc) + 1)

    plt.figure()

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()
except:
    print("LEARNING CURVE ONLY AVAILABLE IF TRAINING HAS BEEN PERFORMED IN THIS RUN")


# In[49]:


loss,acc = model.evaluate(X_train,y_train, verbose=0)
print("Accuracy on Training Data : %.2f%%" % (acc*100))


# In[50]:


loss,acc = model.evaluate(X_test,y_test, verbose=0)
print("Accuracy on Test Data: %.2f%%" % (acc*100))


# ## CNN 
# ### Define Architecture

# In[51]:


if CNN1_AVAILABLE:
    model=load_model(cnnsimplemodelname)
    print("CNN SIMPLE MODEL ALREADY AVAILABLE \nLOAD EXISTING MODEL")
else:    
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32,3), padding='same',activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[52]:


visualkeras.layered_view(model,legend=True)


# ### Define Training Parameters

# In[53]:


if not CNN1_AVAILABLE:
    # Compile model
    epochs = 10
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# ### Perform Training

# In[54]:


if not CNN1_AVAILABLE:
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
    model.save(cnnsimplemodelname)
    CNN1_AVAILABLE=True
else:
    print("TRAINED MODEL ALREADY AVAILABLE")


# ### Evaluation

# In[55]:


try:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    max_val_acc=np.max(val_acc)

    epochs = range(1, len(acc) + 1)

    plt.figure()

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()
except:
    print("LEARNING CURVE ONLY AVAILABLE IF TRAINING HAS BEEN PERFORMED IN THIS RUN")


# In[56]:


loss,acc = model.evaluate(X_train,y_train, verbose=0)
print("Accuracy on Training Data : %.2f%%" % (acc*100))


# In[57]:


loss,acc = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy on Test Data: %.2f%%" % (acc*100))


# ## A more complex CNN
# 
# ### Architecture

# In[58]:


def createModel():
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32,3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


# In[59]:


if CNN2_AVAILABLE:
    model=load_model(cnnadvancedmodelname)
    print("CNN ADVANCED MODEL ALREADY AVAILABLE \nLOAD EXISTING MODEL")
else:  
    model = createModel()
model.summary()


# In[60]:


visualkeras.layered_view(model,legend=True)


# ### Define Training Parameters

# In[61]:


if not CNN2_AVAILABLE:
    batch_size = 256
    epochs = 50
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# ### Perform Training

# In[62]:


if not CNN2_AVAILABLE:
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_test, y_test))
    model.save(cnnadvancedmodelname)
    CNN2_AVAILABLE=True
else:
    print("TRAINED MODEL ALREADY AVAILABLE")


# ### Evaluate

# In[63]:


try:
    plt.figure(figsize=[8,6])
    plt.plot(history.history['accuracy'],'r',linewidth=3.0)
    plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.show()
except:
    print("LEARNING CURVE ONLY AVAILABLE IF TRAINING HAS BEEN PERFORMED IN THIS RUN")


# In[64]:


loss,acc = model.evaluate(X_train,y_train, verbose=0)
print("Accuracy on Training Data : %.2f%%" % (acc*100))


# In[65]:


loss,acc = model.evaluate(X_test,y_test, verbose=0)
print("Accuracy on Test Data : %.2f%%" % (acc*100))


# ## Visualize Feature Maps in 2nd Conv-Layer
# 
# The output of an arbitrary layer, for a given input image can be visualized as demonstrated below.
# 
# First we select and display an image, for which the featuremaps in the 2nd Convolution Layer of the previously defined and trained network shall be generated:

# In[142]:


img=X_train[7:8,:,:,:]


# In[143]:


img.shape


# In[144]:


plt.figure(figsize=(3,3))
plt.imshow(img[0])


# Next, we define a network, which contains the first 2 convolution layers of the previously trained network:

# In[145]:


FirstLayer=Model(inputs=model.inputs, outputs=model.layers[1].output)


# In[146]:


FirstLayer.summary()


# Then we pass the selected image to the extracted subnetwork. The output are the feature-maps of the second convolutional layer:

# In[147]:


feature_maps = FirstLayer.predict(img)


# There are 32 feature-maps, each of size $(32 \times 32)$:

# In[148]:


feature_maps.shape


# Finally we visualize these 32 feature-maps:

# In[149]:


# alle Feature Maps plotten
square = 8
ix = 1
plt.figure(figsize=(20,20))
for _ in range(8):
    for _ in range(4):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature_maps[0, :, :, ix-1])
        ix += 1
plt.show()


# In[ ]:




