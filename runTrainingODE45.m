% runTrainingODE45
%
% This script loads MNIST data downsampled to 14x14 and then trains
% an sNODE network by NCG-method, using Matlab's ODE-solvers
%
% It sets up parameters as needed and calls ncgTrainingODE45
%
% The trained weights and biases for the sNODE will be saved in a file
% according to the settings.

clear;
verbose = 1;  % messages printed to stdout

mu1 = 1;    % coefficient at l2-loss
mu2 = 0;    % coefficient at CE-loss
mu3 = 0;    % coefficient at output magnitude
mu4 = 0;    % weight decay (L2 norm of W and b)

noEpochs = 10;      % number of epochs
noIterations = 6;   % number of iterations in each epoch
actFun = 'tanh';    % activation function (see help of setActFun)
tResolution = 50;   % number of subintervals of a unit time interval for Euler discretization
T = 3;              % final time of the network. Output is obtained at t=T
bSobolev = true;    % use Sobolev gradient descent
batchSize = 100;    % number of images in each minibatch
measFrequency = 20; % how often accuracy is measured (every 20th batch)

outputFileNm = 'ncg_ode1.mat';  % file where the trained weights and biases will be saved

% load downsampled 14x14 MNIST data
load('mnist14x14_uint16.mat');
trainInputData = double(trnInput); trainClass = double(trnClass);
testInputData = double(tstInput);  testClass = double(tstClass);

% desired renormalization of the input
inputBgEnc = 0;
inputFgEnc = -1;

% values for the modified one-hot encoding used for l2-loss
l2enc0 = 0;
l2enc1 = 1;

% NB: if activation function is non-negative (e.g. ReLU) and if l2-loss is
%     used to match the network output with the ground truth, then
%     l2enc0 and l2enc1 MUST BE GREATER THAN or equal to inputBgEnc and
%     inputFgEnc
%     This is a limitation due to the structure of an sNODE.
%     This requirement can be relaxed when CE-loss is used

ncgTrainingODE45(verbose, trainInputData, trainClass, testInputData, testClass, ...
                 inputBgEnc, inputFgEnc, l2enc0, l2enc1, ...
                 actFun, mu1, mu2, mu3, mu4, ...
                 noEpochs, noIterations, batchSize, ...
                 tResolution, T, bSobolev, ...
                 measFrequency, outputFileNm);
