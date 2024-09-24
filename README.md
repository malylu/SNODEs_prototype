# Prototype of Standalone Neural ODEs and Adversarial Attacks with MATLAB Source Code

## Training sNODEs using non-linear conjugate gradient method

This folder contains two runnable scripts for training up an sNODE network that classifies downsampled MNIST data.


### runTrainingODE45.m
This script loads *MNIST data downsampled to 14x14* and then trains an sNODE network by NCG-method, using *Matlab's ODE-solvers*. It sets up parameters as needed and calls **ncgTrainingODE45**. The trained weights and biases for the sNODE will be saved in a file according to the settings.

### runTrainingEuler.m
This script loads *MNIST data downsampled to 14x14* and then trains an sNODE network by NCG-method, using exclusively *Euler's method* to solve the ODEs. It sets up parameters as needed and calls **ncgTrainingEuler**. The trained weights and biases for the sNODE will be saved in a file according to the settings.


## Adversarial attacks
This folder contains one runnable script for attacking classifiers done by an sNODE network.

### adversarialAttack.m
The script loads an sNODE network trained by NCG using Euler's method (alternatively a ResNet analog, possibly trained by SGD). Then, it loads test dataset from MNIST 14x14 and performs adversarial attack as described in our paper


## Auxilliary files
* __mnist14x14_uint16.mat__ -- Downsampled MNIST data

* __ncg_tanh_l2_run1.mat__ -- An sNODE network trained by NCG with Euler's method

* __initWB.m__ and __initWBcell.m__ -- Functions to initialize weights and biases at the beginning of a training session

* __evalAccuracyEuler.m__ and __evalAccuracyODE45.m__ -- Functions that run inference on given data and return the achieved accuracy

* __L2toW12.m__ and __L2toW12cell.m__ -- Functions that transform L2-gradient descent to Sobolev-gradient descent

* __ncgGamma.m__ -- Function that computes a conjugate gradient coefficient

* __ncgTrainingEuler.m__ -- Function that performs training using Euler's method

* __ncgTrainingODE45.m__ -- Function that performs training using MATLAB's ode-solvers

* __setActFun.m__ -- Function that loads activation function and its derivative

* __splineInt.m__ -- A simple cubic spline interpolation for functions discretized with equidistant nodes


## Additional files available
Additional readily trained sNODE network files have been posted at [snodes.bitbucket.io](https://snodes.bitbucket.io/)
