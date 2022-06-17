# Second-order Optimization Methods in Deep Neural Networks Training
This repository contains code and report for the final project of *Optimization for Machine learning (CS-439)* course.

## Abstract
Optimization in machine learning, both theoretical and applied, is dominated by first-order gradient methods such as stochastic gradient descent (SGD). In this article, we explore the applicability of second-order methods to the image classification problem. We analyze the performance of the Convolutional Neural Network trained with AdaHesian, LBFGS and BB methods in comparison to the first-order optimization methods such as SGD, SGD-momentum and Adam.

## Repository structure
This repository contains the report, code and results of our experiments.

`lib` contains:

- `models.py` - CNN model
- `train.py` - train pipeline for CIFAR classification

Also there is `Experiments.ipynb` notebook with experiments and `train.py` script.

Folder `results` contains the dictionaries with results (train and valid loss, scores and running time) for `Adam`, `SGD`, `SGD-momentum`, `AdaHessian`, `BB` optimizers.


The project has been developed and test with `Python 3.7.11`

Required libraries:

- `torchvision == 0.12.0`
- `matplotlib.pyplot`
- `sklearn`
- `numpy`
- `time`
- `torch == 1.11.0`

## Team

This project is accomplished by:  
- Anastasia Filippova anastasiia.filippova@epfl.ch
- Sofia Blinova sofia.blinova@epfl.ch
- Tikhon Parshikov tikhon.parshikov@epfl.ch
