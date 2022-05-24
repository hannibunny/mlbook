#!/usr/bin/env python
# coding: utf-8

# # Implementing Neural Networks with Keras
# * Author: Johannes Maucher
# * Last Update: 02.11.2020

# ## What you will learn:
# * Define, train and evaluate MLP in Keras
# * Define, train and evaluate CNN in Keras
# * Visualization of learning-curves
# * Implement cross-validation in Keras
# * Image classification based on the CIFAR-10 dataset, which is included in [Keras datasets](https://keras.io/datasets/).

# ## Imports and Configurations

# In[40]:


#!pip install Pillow
#!pip install tensorflow-cpu


# In[41]:


#import tensorflow
#from tensorflow import keras


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
#from keras.datasets import cifar10
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Input,Dropout,Flatten,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import set_image_data_format
import os


# In[43]:


set_image_data_format("channels_last")


# In[44]:


import warnings
warnings.filterwarnings("ignore") 


# The following code-cell is just relevant if notebook is executed on a computer with multiple GPUs. It allows to select the GPU. 

# In[45]:


#from os import environ
#environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#environ["CUDA_VISIBLE_DEVICES"]="1"


# In this notebook the neural network shall not learn models, which already exists. This is implemented as follows. The three models (MLP and two different CNNs) are saved to the files, whose name is assigned to the variables `mlpmodelname`, `cnnsimplemodelname` and `cnnadvancedmodelname`, respectively. 
# If these files exist (checked by `os.path.isfile(filename)`) a corresponding AVAILABLE-Flag is set. If this flag is `False`, the corresponding model will be learned and saved, otherwise the existing model will be loaded from disc.

# In[67]:


modeldirectory="models/"
mlpmodelname=modeldirectory+"dense512"
cnnsimplemodelname=modeldirectory+"2conv32-dense512"
cnnadvancedmodelname=modeldirectory+"2conv32-4conv64-dense512"


# In[68]:


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


# In[69]:


CNN1_AVAILABLE


# ## Access Data 
# 
# Load the Cifar10 image dataset from `keras.datasets`. Determine the shape of the training- and the test-partition.

# In[49]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[50]:


print(np.shape(X_train))
print(np.shape(X_test))


# ## Visualize Data
# 
# Viusalize the first 9 images of the training-partition, using function `imshow()` from `matplotlib.pyplot`.

# In[51]:


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

# In[52]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0


# Labels of the first 9 training images:

# In[53]:


print(y_train[:9])


# **Label-Encoding:** Transform the labels of the train- and test-partition into a one-hot-encoded representation. 

# In[54]:


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

# In[60]:


if MLP_AVAILABLE:
    model=load_model(mlpmodelname)
    print("MLP MODEL ALREADY AVAILABLE \nLOAD EXISTING MODEL")
else:
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32,3)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
model.summary()


# #### Network definition option 2: Using the functional API

# In[16]:


# This returns a tensor
inputs = Input(shape=(32, 32,3))
x=Flatten()(inputs)
x=Dense(512, activation='relu')(x)
x=Dense(num_classes, activation='softmax')(x)
model2 = Model(inputs=inputs, outputs=x)
model2.summary()


# ### Define Training Parameters 
# Apply Stochastic Gradient Descent (SGD) learning, for minimizing the `categorical_crossentropy`. The performance metric shall be `accuracy`. Train the network.

# In[61]:


if not MLP_AVAILABLE:
    # Compile model
    epochs = 8
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# ### Perform Training

# In[62]:


if not MLP_AVAILABLE:
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32,verbose=False)
    model.save(mlpmodelname)
    MLP_AVAILABLE=True
else:
    print("TRAINED MODEL ALREADY AVAILABLE")


# ### Evaluation 
# Visualize the learning-curve on training- and test-data. 

# In[63]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb
sb.set_style("whitegrid")
sb.set_context("notebook")


# In[64]:


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


# In[65]:


loss,acc = model.evaluate(X_train,y_train, verbose=0)
print("Accuracy on Training Data : %.2f%%" % (acc*100))


# In[66]:


loss,acc = model.evaluate(X_test,y_test, verbose=0)
print("Accuracy on Test Data: %.2f%%" % (acc*100))


# ## CNN 
# ### Define Architecture

# In[71]:


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


# ### Define Training Parameters

# In[72]:


if not CNN1_AVAILABLE:
    # Compile model
    epochs = 10
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# ### Perform Training

# In[73]:


if not CNN1_AVAILABLE:
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
    model.save(cnnsimplemodelname)
    CNN1_AVAILABLE=True
else:
    print("TRAINED MODEL ALREADY AVAILABLE")


# ### Evaluation

# In[26]:


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


# In[27]:


loss,acc = model.evaluate(X_train,y_train, verbose=0)
print("Accuracy on Training Data : %.2f%%" % (acc*100))


# In[28]:


loss,acc = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy on Test Data: %.2f%%" % (acc*100))


# ## A more complex CNN
# 
# ### Architecture

# In[29]:


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


# In[30]:


if CNN2_AVAILABLE:
    model=load_model(cnnadvancedmodelname)
    print("CNN ADVANCED MODEL ALREADY AVAILABLE \nLOAD EXISTING MODEL")
else:  
    model = createModel()
model.summary()


# ### Define Training Parameters

# In[31]:


if not CNN2_AVAILABLE:
    batch_size = 256
    epochs = 50
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# ### Perform Training

# In[32]:


if not CNN2_AVAILABLE:
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_test, y_test))
    model.save(cnnadvancedmodelname)
    CNN2_AVAILABLE=True
else:
    print("TRAINED MODEL ALREADY AVAILABLE")


# ### Evaluate

# In[33]:


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


# In[34]:


loss,acc = model.evaluate(X_train,y_train, verbose=0)
print("Accuracy on Training Data : %.2f%%" % (acc*100))


# In[35]:


loss,acc = model.evaluate(X_test,y_test, verbose=0)
print("Accuracy on Test Data : %.2f%%" % (acc*100))


# ## Implementation of Cross Validation in Keras
# 
# Here: Cross-Validation of MLP

# In[36]:


CROSS_VAL=True


# In[37]:


def build_model_mlp():
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32,3)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    ###################################################
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model


# In[38]:


def cross_validation(build_model,train_data,train_targets,folds=3,num_epochs=10):
    num_val_samples = int(len(train_data) / folds)
    all_scores = []
    for i in range(folds):
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate(                                     
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)

        model = build_model()                                                    
        model.fit(partial_train_data, partial_train_targets,                     
                  epochs=num_epochs, batch_size=32, verbose=0)
        val_score = model.evaluate(val_data, val_targets, verbose=0)      
        all_scores.append(val_score)
        print(" Loss on test data: %2.4f \n Accuracy on test data: %2.4f \n"% (val_score[0],val_score[1]))
    return all_scores


# In[39]:


if CROSS_VAL:
    results=cross_validation(build_model_mlp,X_train,y_train)
    print(results)


# In[ ]:




