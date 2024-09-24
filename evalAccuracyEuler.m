function acc = evalAccuracyEuler(inputData, classData, sigma, Wcell, Bcell)
% acc = evalAccuracyEuler(inputData, classData, sigma, Wcell, Bcell)
%
% evalAccuracyEuler runs inference on inputData by determining argmax of
% the ResNet/sNODE network output, where the neural network is modelled by
% the direct problem xDE.
% Note: The inference is run in batches of 100 datapoints.
%
% Input parameters:
% inputData: a matrix of datapoints to be run through network. Each column
%            is one datapoint
% classData: a row-vector containing zero-based ground truth for given data
% sigma, Wcell and Bcell: determine the right-hand side of the sNODE,
%                         i.e., xDE = @(t, x) sigma(Wcell{t+1} * x + B{t+1})
% Output:
% Percentage of correctly classified datapoints (already multiplied by 100)

    L = max(classData(:))+1;  % number of classes to consider in the networks output
    K = size(inputData,2);    % number of datapoints to be tested

    lastIdx = length(Wcell);
    xDE = @(t, x) sigma(Wcell{t+1} * x + Bcell{t+1});

    batchRemainder = mod(K, 100);

    errCount = 0;

    for currStart = 1:100:K-99
        xt = inputData(:, currStart:currStart+99);     %  load input data, 100 images at once
        trueClass = classData(currStart:currStart+99)+1;  %  load class data and compensate for being 0-based

        for i = 0:lastIdx-1       % solve the direct problem using Euler method
            xt = xt + xDE(i, xt);  
        end

        [~, inferredClass] = max( xt(1:L, :) ); % find argmax of the network output
        errCount = errCount + sum((abs(trueClass-inferredClass)>0.5));
    end

    if batchRemainder>0
        xt = inputData(:, end-batchRemainder+1 : end);       %  load input data, remainder
        trueClass = classData(end-batchRemainder+1 : end)+1; %  load class data and compensate for being 0-based

        for i = 0:lastIdx-1       % solve the direct problem using Euler method
            xt = xt + xDE(i, xt);  
        end

        [~, inferredClass] = max( xt(1:L, :) ); % find argmax of the network output
        errCount = errCount + sum((abs(trueClass-inferredClass)>0.5));
    end

    acc = 100 * (1 - errCount / K);

end