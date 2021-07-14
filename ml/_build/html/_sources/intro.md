# Intro and Overview Machine Learning Lecture

* Author: Prof. Dr. Johannes Maucher
* Email: maucher@hdm-stuttgart.de
* Last Update: June, 25th 2021

## Goals of this Lecture

**Goals of this lecture are:**


* Understand Neural Networks, in particular Deep Neural Networks
* Learn how to implement neural network- and deep neural network applications with [Keras](https://keras.io/).

<a id='data_mining'></a>
## Contents

### Intro

1. [Basic Concepts of Data Mining and Machine Learning](00BasicConcepts.ipynb)
    * Definition
    * Categories
    * Validation
	
2. [Basics of Probability Theory (external Link)](https://hannibunny.github.io/probability/intro.html)

### Conventional ML Algorithms
    
2. [Support Vector Machines (SVM)](./machinelearning/svm.md)
    * SVM
    
2. [Gaussian Process](./machinelearning/gp.md)
    * GP
    
    
### Neural Networks and Deep Neural Networks

11. [Conventional Neural Networks](neuralnetworks/01NeuralNets.ipynb) 
    * Natural Neuron
    * General Notions for Artificial Neural Networks
    * Single Layer Perceptron (SLP)
        * Architectures for Regression and Classification
        * Gradient Descent- and Stochastic Gradient Descent Learning
    * Gradient Descent- and Stochastic Gradient Descent Learning
    * Multilayer Perceptron (MLP) Architectures for Regression and Classification
    * Backpropagation-Algorithm for Learning
    

12. [Recurrent Neural Networks (RNN)](neuralnetworks/02RecurrentNeuralNetworks.ipynb) 
    * Simple Recurrent Neural Networks (RNNs)
    * Long short-term Memory Networks (LSTMs)
    * Gated Recurrent Units (GRUs)
    * Application Categories of Recurrent Networks


13. [Deep Neural Networks: Convolutional Neural Networks (CNN)](neuralnetworks/03ConvolutionNeuralNetworks.ipynb) 
    * Overall Architecture of CNNs
    * General concept of convolution filtering
    * Layer-types of CNNs: 
        * Convolution, 
        * Pooling, 
        * Fully-Connected 

14. [MLP and CNN for Object Classification](neuralnetworks/03KerasMLPandCNNcifar.ipynb)
    * Example Data: Cifar-10 Image Dataset
    * Image Representation in numpy
    * Define, train and evaluate MLP in Keras
    * Define, train and evaluate CNN in Keras 
    
    
19. [Apply pretrained CNNs for object classification - original task](neuralnetworks/04KerasPretrainedClassifiers.ipynb)
    * Access image from local file system
    * Download and apply pretrained CNNs for object recognition in arbitrary images
    

    
20. [Use of pretrained CNNs for object classification - new task: Classify x-ray images of lungs into healthy and covid-19](neuralnetworks/05KerasPretrainedCovid.ipynb)
    * Download pretrained feature-extractor (CNN without the classifier part)
    * Define new classifier architecture and concatenate it with pretrained classifier
    * Fine-tune network with task-specific data
    * Apply the fine-tuned network for object-recognition
    
    

### Autoencoder

14. [Autoencoder](neuralnetworks/04VariationalAutoencoder.ipynb)
    * AE    

### GAN

15. [Generative Adversarial Networks](gan/DCGAN.ipynb)

### Modelling of Text
    
    
    
16. [Modelling of Words and Texts / Word Embeddings](text/01ModellingWordsAndTexts.ipynb) 
    * Concept of Word-Embeddings
    * Skip-Gram and CBOW
    * Working with pretrained word-embeddings
    
    
    
14. [Text Classification with CNNs and LSTMs](text/02TextClassification.ipynb)
    * Example Data: IMDB-Movie Reviews for Sentiment Classification
    * Text preprocessing and representation with Keras
    * Load and apply pretrained word-embedding
    * News classification with CNN
    * News classification with LSTM
    


