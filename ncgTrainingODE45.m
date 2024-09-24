function ncgTrainingODE45(verbose, trainInputData, trainClass, testInputData, testClass,  inputBgEnc, inputFgEnc, l2enc0, l2enc1, actFun, mu1, mu2, mu3, mu4, noEpochs, noIterations, batchSize, tResolution, T, bSobolev, measFrequency, outputFileNm)
% ncgTrainingODE45(verbose, trainInputData, trainClass, testInputData,
%                  testClass, inputBgEnc, inputFgEnc, l2enc0, l2enc1,
%                  actFun, mu1, mu2, mu3, mu4, noEpochs, noIterations,
%                  batchSize, tResolution, T, bSobolev, measFrequency,
%                  outputFileNm)
%
% This function trains weights and biases in an sNODE using the non-linear
% conjugate gradient method, where differential equations are solved by
% ODE45 or other solvers. It saves the trained weights in outputFileNm.
% This output file contains also some of the settings of the architecture
% of the particular sNODE. 
% 
% Parameters:
% verbose:
% 0 = quiet mode, (only errors are printed out)
% positive value = file identifier to output messages (e.g., 1 = stdout)
%
% trainInputData, testInputData:
% Input data for the training and test sets, respectively. These are to be
% stored in a matrix, where each column corresponds to one data-point.
% Values of the data will be linearly rescaled to fit between inputBgEnc
% and inputFgEnc
%
% trainClass, testClass:
% Ground truth for classification of the given data (zero-based)
%
% inputBgEnc and inputFgEnc:  
% Values of input data will be (linearly) rescaled, so that the least value
% is mapped onto inputBgEnc and the greatest value onto inputFgEnc.
%
% l2enc0 and l2enc1:
% When using l2-loss, it might be desirable to not use exactly one-hot
% encoding but rather its modification, where 0 is represented by l2enc0
% and 1 is represented by l2enc1.
% These parameters do NOT have any effect for CE-loss
%
% actFun:
% Activation function in the sNODE, e.g. 'tanh' or 'relu'. See the help of
% setActFun for a full list of possible activation functions.
%
% mu1, mu2, mu3, mu4:
% Parameters for the cost functional. mu1 for l2-loss, mu2 for CE-loss,
% mu3 for l2-magnitude of output, mu4 for weight decay (l2-norm of W,b)
%
% noEpochs, noIterations:
% Number of epochs and iterations in each epoch
%
% batchSize:
% Number of datapoints included in each minibatch
%
% tResolution:
% Number subintervals in a unit time interval used in the Euler
% discretization and in the stored weights and matrices.
%
% T:
% Final time in the sNODE, i.e., time of the output layer.
%
% bSobolev:
% Flag true/false to use Sobolev gradient descent direction
%
% measFrequency:
% How often accuracy on the whole training set and the whole set is to be
% measured, namely every measFrequency'th batch. Set to negative value to
% measure it exactly once, at the end of each epoch
%
% outputFileNm:
% Name of the file where the trained weights and biases (including
% measurements of accuracy during the training) will be saved
%

inputShift = min(trainInputData(:));
inputScaleFactor = (inputFgEnc-inputBgEnc) / (max(trainInputData(:)) - inputShift);

% Rescale input data
xTrain = inputBgEnc + inputScaleFactor * (double(trainInputData) - inputShift);
xTest  = inputBgEnc + inputScaleFactor * (double(testInputData) - inputShift);

nTrain = size(xTrain, 2);

% Ground truth classes, make sure to have compatible data type (double)
cTrain = double(trainClass);
cTest = double(testClass);
noClasses = max(cTrain(:))+1;

% One-hot encoding for the ground truth for CE-loss
y01Train = zeros(size(xTrain));
y01Train( sub2ind(size(y01Train), cTrain+1, 1:nTrain) ) = 1; 

% Modified one-hot encoding for the ground truth for l2-loss
yTrain = l2enc0 + (l2enc1 - l2enc0) * y01Train;

% load activation functions
[sigma, dsigma] = setActFun(actFun, 1);

% determine number of batches
noBatches = floor(nTrain / batchSize);
if (measFrequency<=0) || (measFrequency > noBatches)
    measFrequency = noBatches;
end

% normalization of mu due to extra factors in the cost functional
muNorm = [batchSize; batchSize; batchSize; tResolution];
m1 = mu1 / muNorm(1);  % for l2-distance between output and target
m2 = mu2 / muNorm(2);  % for cross-entropy between output and target
m3 = mu3 / muNorm(3);  % for l2-norm of output
m4 = mu4 / muNorm(4);  % for weight decay

% Epoch start
currentEpoch = 0;

% ODE solver settings (abs. tolerance is set more generously)
odeSolverOpt = odeset('RelTol', 1e-3, 'AbsTol', 1e-3);

% Initialize weights, biases, and the direct problem
% Discretization parameters
%lastIdx = T * tResolution;
layerWidth = size(xTrain, 1);
tStep = 1/tResolution;

[Wdisc, Bdisc, tdisc] = initWB('normal', 1/layerWidth, layerWidth, T, tResolution);
tLastIdx = length(tdisc);

W = @(t) reshape(splineInt(t, tStep, Wdisc, tLastIdx), layerWidth, layerWidth);
B = @(t) splineInt(t, tStep, Bdisc, tLastIdx);

% Compute accuracy at the start
trainAcc = [0, 0, evalAccuracyODE45(xTrain, cTrain, sigma, Wdisc, Bdisc, tStep, odeSolverOpt)];
testAcc = [0, 0, evalAccuracyODE45(xTest, cTest, sigma, Wdisc, Bdisc, tStep, odeSolverOpt)];

while currentEpoch < noEpochs
    currentEpoch = currentEpoch+1;

    % randomize batches within this epoch
    dataBatches = reshape(randperm(nTrain, noBatches * batchSize), batchSize, noBatches);

    % start running through batches
    currentBatch = 0;
    while currentBatch < noBatches
        currentBatch = currentBatch + 1;
        if verbose > 0
            fprintf(verbose, 'Running batch #%d of epoch #%d\n', currentBatch, currentEpoch);
            batchTimer = tic;
        end
        
        % Initialize batch
        x0 = xTrain(:, dataBatches(:,currentBatch));       %  load input data x_k(0) from the current batch. Each datapoint is a column in the matrix x0
        y  = yTrain(:, dataBatches(:,currentBatch));       %  load l2-encoded output data y_k from the current batch. Each datapoint is a column in the matrix y
        y01= y01Train(:, dataBatches(:,currentBatch));     %  load one hot encoded output data y_k from the current batch. Each datapoint is a column in the matrix y01
        yc = cTrain(:, dataBatches(:,currentBatch));       %  load class data from the current batch.

        j=0;            % Iteration counter

        UWdisc = zeros(size(Wdisc)); UBdisc = zeros(size(Bdisc)); % allocate memory
        dWdisc = zeros(size(Wdisc)); dBdisc = zeros(size(Bdisc)); % the first iteration in each batch uses the steepest descent, so there is no "previous step"
        dW = @(t) reshape(splineInt(t, tStep, dWdisc, tLastIdx), layerWidth, layerWidth);
        dB = @(t) splineInt(t, tStep, dBdisc, tLastIdx);

        L2dE = 1;                                                 % L2 norm of E': dummy in the first iteration

        while j < noIterations
            j = j+1;
            
            % Solve the direct problem on the batch data
            xODE    = @(t, x) sigma(W(t) * x + B(t));
            xODEvec = @(t, x) reshape(xODE(t, reshape(x, layerWidth, batchSize)), [], 1);

            xSol = ode45(xODEvec, [0, T], x0(:), odeSolverOpt);
            xVec = @(t) deval(xSol,t);
            xVecT = xVec(T);
            x = @(t) reshape(xVec(t), layerWidth, batchSize);
            xT = reshape(xVecT, layerWidth, batchSize);

            % Solve the adjoint problem (final value problem, steps back in time)
            lODE    = @(t, l) W(t).' * (dsigma(W(t) * x(t) + B(t)) .* l) ;
            lODEvec = @(t, l) reshape(lODE(T-t, reshape(l, layerWidth, batchSize)), [],1);
            lT = m1 * (xVecT - y(:)) ...
                + m2 * (reshape([softmax(xT(1:noClasses,:)); zeros(size(xT)-[noClasses,0])], [], 1) - y01(:)) ...
                + m3 * (xVecT - T*sigma(0));
            lSol = ode23(lODEvec, [0, T], lT(:), odeSolverOpt);
            lVec = @(t) deval(lSol,T-t);
            l = @(t) reshape(lVec(t), layerWidth, batchSize);

            % Compute E'
            % UW and UB contain the direction of Steepest Descent for E'
            Utmp = @(t) l(t) .* dsigma(W(t) * x(t) + B(t));
            UB = @(t) m4 * B(t) + sum(Utmp(t), 2);
            UW = @(t) m4 * W(t) + Utmp(t) * (x(t)).';

            % Discretize UB and UW
            for i = 1:length(tdisc)
                UWdisc(:,i) = reshape(UW(tdisc(i)), [], 1);
                UBdisc(:,i) = UB(tdisc(i));
            end
            
            % If Sobolev gradient descent is desired, then apply
            % transformation
            if bSobolev
                UWdisc = L2toW12(UWdisc, tStep);
                UBdisc = L2toW12(UBdisc, tStep);
            end
            
            % Compute gamma (Iteration 2+). Set gamma = 0 in Iteration 1.
            % Fletcher--Reeves conjugate gradient coefficient
            lastL2dE = L2dE;
            L2dE = tStep * ( sum(UBdisc(:).^2) + sum(UWdisc(:).^2) );
            gamma = +(j>1) * L2dE / lastL2dE;
            assert(isfinite(gamma), 'Gamma undefined');

            % Find the discretized update direction (conjugate gradient)
            % The direction is scaled down for the sake of sensitivity
            % problem that is solved by Eulers method
            dWdisc = (-UWdisc + gamma * dWdisc) * 1e-3;
            dBdisc = (-UBdisc + gamma * dBdisc) * 1e-3;

            dW = @(t) reshape(splineInt(t, tStep, dWdisc, tLastIdx), layerWidth, layerWidth);
            dB = @(t) splineInt(t, tStep, dBdisc, tLastIdx);

            % Solve the sensitivity problem
            xiODE = @(t, xi) dsigma(W(t)*x(t)+B(t)) .* (W(t) * xi + dB(t) + dW(t) * x(t));
            xiT = zeros(size(x0));

            for t = 0:0.05:T-0.05
                xiT = xiT + 0.05*xiODE(t, xiT);
            end
            xiVecT=xiT(:);
            xiTCE = xiT; xiTCE(11:end,:) = 0;

            m4coeff1 = (sum(sum(dWdisc .* Wdisc)) + sum(sum(dBdisc .* Bdisc))) * tStep;
            m4coeff2 = (sum(sum(dWdisc.^2)) + sum(sum(dBdisc.^2))) * tStep;

            % Compute steplength eta
            etaeqnkonst = m1 * sum(xiVecT .* (xVecT-y(:)) ) - ...
                m2 * sum(xiVecT .* y01(:)) + ...
                m3 * sum(xiVecT .* (xVecT - T*sigma(0)) ) + ...
                m4 * m4coeff1;

            etaeqnlin = (m1 + m3) * sum(xiVecT.^2) + ...
                m4 * m4coeff2;

            etaeqn = @(b) etaeqnkonst + etaeqnlin * b + ...
                m2 * sum(sum(softmax(xT(1:noClasses,:) + b * xiT(1:noClasses,:)) .* xiT(1:noClasses,:)));
            
            eta = fzero(etaeqn, -etaeqn(0)/(etaeqnlin + (etaeqnlin==0)));
            assert(isfinite(eta), 'Eta undefined');


            % Update W and B:  Wnew = Wprev + eta * newdirW and analogously for B
            Wdisc = Wdisc + eta * dWdisc;
            Bdisc = Bdisc + eta * dBdisc;

            W = @(t) reshape(splineInt(t, tStep, Wdisc, tLastIdx), layerWidth, layerWidth);
            B = @(t) splineInt(t, tStep, Bdisc, tLastIdx);
        end
        if verbose > 0
           fprintf(verbose, 'Elapsed time is %0.2f seconds\n', toc(batchTimer));
        end
        
        if mod(currentBatch, measFrequency) == 0
            trainAcc(end+1, :) = [currentEpoch, currentBatch, evalAccuracyODE45(xTrain, cTrain, sigma, Wdisc, Bdisc, tStep, odeSolverOpt)];
            testAcc(end+1, :)  = [currentEpoch, currentBatch, evalAccuracyODE45(xTest, cTest, sigma, Wdisc, Bdisc, tStep, odeSolverOpt)];
            save(['tmp_', outputFileNm], 'Bdisc', 'Wdisc', 'actFun', 'trainAcc', 'testAcc', 'tResolution', 'T', 'inputBgEnc', 'inputScaleFactor', 'inputShift', 'odeSolverOpt');
            if verbose>0
                fprintf(verbose, 'Train Acc: %0.1f%%,    Test Acc: %0.1f%%\n', trainAcc(end, 3), testAcc(end, 3));
            end
        end

    end
end

save(outputFileNm, 'Bdisc', 'Wdisc', 'actFun', 'trainAcc', 'testAcc', 'tResolution', 'T', 'inputBgEnc', 'inputScaleFactor', 'inputShift', 'odeSolverOpt');
delete(['tmp_', outputFileNm]);
