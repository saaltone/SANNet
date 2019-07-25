# SANNet - Neural Network Framework

## Background
SANNet is an artificial neural network framework that provides functionalities to build multi-layer neural networks. It has been born from long term interest towards artificial neural networks and curiosity to understand inner workings of neural networks.

Framework's primary components are neural network instance, input layer, variable number of hidden layers and output layer. Neural network instance and layers run in their dedicated threads to enable concurrency between multiple neural network instances.

SANNet is written on Java and has been built from group up starting from implementation of matrix library and functions. 

## Layers, activation and loss functions.
Framework provides vanilla feedforward layer, recurrent layers (simple recurrent layer, LSTM layer, Graves LSTM layer, GRU layer) and convolutional layers (convolutional layer, max / average pooling layer). It supports multiple layer activation functions and loss functions for output layer. Additionally there are multiple initialization methods for weight parameters like Xavier/He/Lecun uniform and normal initializations.

## Optimization
Framework implements most typically used optimization methods starting from basic vanilla gradient descent up to more sofisticated parameter optimization methods such as Adam and AMSGrad.

## Regularization
Framework provides few regularization methods: drop-out, gradient clipping, early stopping and L1/L2/Lp regularization. Lp regularization is an experimental method and mathematically direct extension of L1/L2 methods.

## Normalization
Framework support following normalization methods: batch normalization, layer normalization ad weight normalization.

## Metrics
Framework provides accuracy metrics for **regression**. Additionally it provides basic metrics for **classification** such as accuracy, precision, recall, specificity and F1 score and calculates confusion matrix as needed.

## Utilities
Framework provides libraries to read inputs from CVS and text files, normalize, split and encode data. It also has persistence support to serialize trained neural network instances into file and restore trained neural network instances from file for later use.

All feedback is welcome.
