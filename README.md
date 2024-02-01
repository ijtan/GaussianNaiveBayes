# Gaussian Naive Bayes From Scratch

This repository provides a Python implementation of the Gaussian Naive Bayes (GNB) classifier from scratch, without using any external libraries. The GNB model assumes that the distribution of the feature variables is normal, making it suitable for various classification tasks.

## Overview
The `CustomNaiveBayes` class contains methods for calculating the classes, prior probabilities, mean, and variance of the input data. The `fit` method is used to train the model on the input data, while the `predict` method is used to predict the class labels for new data. The `accuracy` method is used to calculate the accuracy of the model.

## Features
- Implementation of GNB classifier from scratch
- Explanation of Bayesian inference and GNB logic
- Utilizes Gaussian probability density function for classification

## Usage
To use the GNB model, follow the instructions provided in the code. The code uses the Iris dataset from the scikit-learn library to train and test the GNB model. The `load_iris` function is used to load the dataset, and the `X` and `y` variables are used as input data for the `fit` method. The `predict` method is used to predict the class labels for the input data, and the `accuracy` method is used to calculate the accuracy of the model.

## Conclusion
The provided code offers a comprehensive resource for learning about the GNB classifier and its underlying principles. It provides a useful starting point for implementing the GNB model without relying on external libraries, providing a valuable learning resource for machine learning enthusiasts.
