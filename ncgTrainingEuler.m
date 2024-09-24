function ncgTrainingEuler(verbose, trainInputData, trainClass, testInputData, testClass,  inputBgEnc, inputFgEnc, l2enc0, l2enc1, actFun, mu1, mu2, mu3, mu4, noEpochs, noIterations, batchSize, tResolution, T, bSobolev, ncgMethod, measFrequency, outputFileNm)
% ncgTrainingEuler(verbose, trainInputData, trainClass, testInputData,
%                  testClass, inputBgEnc, inputFgEnc, l2enc0, l2enc1,
%                  actFun, mu1, mu2, mu3, mu4, noEpochs, noIterations,
%                  batchSize, tResolution, T, bSobolev, ncgMethod,
%                  measFrequency, outputFileNm)
%
% This function trains weights and biases in an sNODE using the non-linear
% conjugate gradient method, where differential equations are solved
% exclusively by Euler's method. It saves the trained weights in
% outputFileNm. This output file contains also some of the settings of the
% architecture of the particular sNODE. 
% 
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
% ncgMethod:
% Conjugate gradient method to be used, e.g., 'FR' for Fletcher--Reeves.
% See help of ncgGamma to see a full list of possible methods
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

% last index for Euler discretization
lastIdx = T * tResolution;
layerWidth = size(xTrain, 1);
tStep = 1/tResolution;

% load activation functions
[sigma, dsigma] = setActFun(actFun, tResolution);

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

% Initialize weights, biases, and the direct problem
[Wcell, Bcell, ~] = initWBcell('normal', 1/layerWidth, layerWidth, T, tResolution);
W = @(t) Wcell{t+1};
B = @(t) Bcell{t+1};

% Compute accuracy at the start
trainAcc = [0, 0, evalAccuracyEuler(xTrain, cTrain, sigma, Wcell, Bcell)];
testAcc = [0, 0, evalAccuracyEuler(xTest, cTest, sigma, Wcell, Bcell)];

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

        dWcell = cellfun(@(M) zeros(size(M)), Wcell, 'un', 0);     % the first iteration in each batch uses the steepest descent, so there is no "previous step"
        dBcell = cellfun(@(M) zeros(size(M)), Bcell, 'un', 0);     % the first iteration in each batch uses the steepest descent, so there is no "previous step"

        % UW and UB contain the direction of Steepest Descent for Ecal'
        UWcell = cellfun(@(M) eye(size(M)), Wcell, 'un', 0);       % the first iteration in each batch uses the steepest descent, so there is no "previous step"
        UBcell = cellfun(@(M) eye(size(M)), Bcell, 'un', 0);       % the first iteration in each batch uses the steepest descent, so there is no "previous step"

        while j < noIterations
            j = j+1;
            
            % Solve the direct problem on the batch data
            xDE = @(t, x) sigma(W(t) * x + B(t));
            xtmp = x0;
            xsol = {xtmp};
            for i = 0:lastIdx-1
                xtmp = xtmp + xDE(i, xtmp);
                xsol{i+2} = xtmp;             % cell-arrays use 1-based indexing, so time (i+1) is at (i+2)nd cell
            end
            x = @(t) xsol{t+1};
            xT = x(lastIdx);

            % Solve the adjoint problem (final value problem, steps back in time)
            LDE  = @(t, LL) W(t).' * (dsigma(W(t) * x(t) + B(t)) .* LL);
            Ltmp = m1 * (xT - y) + m2 * ([softmax(xT(1:noClasses,:)); zeros(size(xT)-[noClasses,0])] - y01) + m3 * (xT - T*sigma(0));
            Lsol{lastIdx} = Ltmp;
            for i = lastIdx-1:-1:1
                Ltmp = Ltmp + LDE(i, Ltmp);
                Lsol{i} = Ltmp;
            end
            
            % Save Ecal' from previous iteration (for gamma)
            UWcellprev = UWcell;
            UBcellprev = UBcell;

            % Compute E'
            % UW and UB contain the direction of Steepest Descent for E'
            Utcell  = cellfun(@(L, W, x, B) L .* dsigma(W * x + B), Lsol, Wcell, xsol(1:end-1), Bcell, 'UniformOutput', false);
            UBcell = cellfun(@(B, U) m4*B + sum(U,2), Bcell, Utcell, 'UniformOutput', false);
            UWcell = cellfun(@(W, U, x) m4*W + U * (x.'), Wcell, Utcell, xsol(1:end-1), 'UniformOutput', false);

            if bSobolev
                UWcell = L2toW12cell(UWcell, tStep);
                UBcell = L2toW12cell(UBcell, tStep);
            end

            % Compute gamma (Iteration 2+). Set gamma = 0 in Iteration 1.
            gamma = ncgGamma(ncgMethod, UWcell, UBcell, UWcellprev, UBcellprev, j);            
            assert(isfinite(gamma), 'Gamma undefined');
            
            % Set d = -Eprim (+ gamma * dprevious)
            dBcell = cellfun(@(PREV, NEW) -NEW + gamma*PREV, dBcell, UBcell, 'UniformOutput', false);
            dWcell = cellfun(@(PREV, NEW) -NEW + gamma*PREV, dWcell, UWcell, 'UniformOutput', false);
            dcurrB = @(t) dBcell{t+1};
            dcurrW = @(t) dWcell{t+1};
            
            Linftydcurrent = max(cellfun(@(M) max(max(abs(M))), dWcell)) + max(cellfun(@(M) max(max(abs(M))), dBcell));
            assert(Linftydcurrent>0, 'New direction is zero. Gradient method yields no good info with this batch/data');

            % Solve the sensitivity problem
            xiDE = @(t, xi) dsigma(W(t)*x(t)+B(t)) .* (W(t) * xi + dcurrB(t) + dcurrW(t) * x(t));
            xiT = zeros(size(x0));
            for i = 0:lastIdx-1
                xiT = xiT + xiDE(i, xiT);
            end
            xiTCE = xiT;
            xiTCE(11:end,:) = 0;

            % Compute steplength eta
            etaeqnkonst = m1 * sum(sum(xiT .* (xT-y))) - ...
                m2 * sum(sum(xiTCE .* y01)) + ...
                m3 * sum(sum(xiT .* (xT - T*sigma(0)) )) + ...
                m4 * sum(cellfun(@(M,N) sum(sum(M.*N)), Wcell, dWcell) + cellfun(@(M,N) sum(sum(M.*N)), Bcell, dBcell));
            
            etaeqnlin = (m1 + m3) * sum(sum(xiT.^2)) + ...
                m4 * sum(cellfun(@(M) sum(sum(M.^2)), dWcell) + cellfun(@(M) sum(sum(M.^2)), dBcell));
            
            etaeqn = @(b) etaeqnkonst + etaeqnlin * b + ...
                m2 * sum(sum(softmax(xT(1:noClasses,:) + b * xiT(1:noClasses,:)) .* xiT(1:noClasses,:)));
            
            eta = fzero(etaeqn, -etaeqn(0)/(etaeqnlin + (etaeqnlin==0)));
            assert(isfinite(eta), 'Eta undefined');

            % Update W and B:  Wnew = Wprev + eta * newdirW and analogously for B
            Wcell = cellfun(@(PREVW, NEWDIRW) PREVW + eta * NEWDIRW, Wcell, dWcell, 'UniformOutput', false);
            Bcell = cellfun(@(PREVB, NEWDIRB) PREVB + eta * NEWDIRB, Bcell, dBcell, 'UniformOutput', false);
            W = @(t) Wcell{t+1};
            B = @(t) Bcell{t+1};
        end
        if verbose > 0
           fprintf(verbose, 'Elapsed time is %0.2f seconds\n', toc(batchTimer));
        end
        if mod(currentBatch, measFrequency) == 0
            trainAcc(end+1, :) = [currentEpoch, currentBatch, evalAccuracyEuler(xTrain, cTrain, sigma, Wcell, Bcell)];
            testAcc(end+1, :) = [currentEpoch, currentBatch, evalAccuracyEuler(xTest, cTest, sigma, Wcell, Bcell)];
            save(['tmp_', outputFileNm], 'Bcell', 'Wcell', 'actFun', 'trainAcc', 'testAcc', 'tResolution', 'T', 'inputBgEnc', 'inputScaleFactor', 'inputShift');
            if verbose>0
                fprintf(verbose, 'Train Acc: %0.1f%%,    Test Acc: %0.1f%%\n', trainAcc(end, 3), testAcc(end, 3));
            end
        end

    end
end

save(outputFileNm, 'Bcell', 'Wcell', 'actFun', 'trainAcc', 'testAcc', 'tResolution', 'T', 'inputBgEnc', 'inputScaleFactor', 'inputShift');
delete(['tmp_', outputFileNm]);
