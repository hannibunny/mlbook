# Intro and Overview Machine Learning Lecture

* Author: Prof. Dr. Johannes Maucher
* Email: maucher@hdm-stuttgart.de
* Last Update: October, 4th 2021

## Goals of this Lecture

**Goals of this lecture are:**


* Understand the basic concepts, procedures and categories of Machine Learning
* For any task or problem: Know, which algorithm-type is suitable for this task
* **Understand** conventional ML-algorithms
* **Understand** Deep Learning
* **Understand** the currently most important types of Deep Neural Networks   
* **Implementations** are provided in order to support understanding

<a id='data_mining'></a>
## Contents

### Intro

1. [Basic Concepts of Data Mining and Machine Learning](00BasicConcepts.ipynb)
    * Definition
    * Categories
    * Validation
	
2. [Basics of Probability Theory (external Link)](https://hannibunny.github.io/probability/intro.html)

### Conventional ML Algorithms
    
1. [K-Nearest Neighbours](./machinelearning/knn.ipynb)
2. [Bayes Classification](./machinelearning/parametricClassification1D.ipynb)
3. [Linear Regression](./machinelearning/LinReg.md)
4. [Linear Classification](./machinelearning/LinearClassification.ipynb)
5. [Support Vector Machines (SVM)](./machinelearning/svm.md)
6. [Gaussian Process](./machinelearning/gp.md)


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

### GAN

15. [Generative Adversarial Networks](gan/DCGAN.ipynb)


### Reinforcement Learning

1. [Reinforcement Learning](rl/reinforcement.md)

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
    

### Graph Neural Networks

17. [Graph Neural Networks](neuralnetworks/GraphNeuralNetworks.ipynb)
    * Concepts of GNNs
    * Implementation of GNN with Keras
    * Document Classification with GNNs

### Attention, Transformer, BERT

1. [Attention, Transformer, BERT](transformer/attention.md)

