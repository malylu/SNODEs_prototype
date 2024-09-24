function acc = evalAccuracyODE45(inputData, classData, sigma, Wdisc, Bdisc, tStep, odeSolverOpt)
% acc = evalAccuracyODE45(inputData, classData, sigma, Wdisc, Bdisc, tStep, odeSolverOpt)
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
% sigma, Wdisc, Bdisc, tStep: determine the right-hand side of the sNODE,
%                             i.e., xDE = @(t, x) sigma(W(t)*x + B(t)),
%                             where W and B are obtained by interpolating
%                             Wdisc and Bdisc
% Output:
% Percentage of correctly classified datapoints (already multiplied by 100)

    L = max(classData(:))+1;  % number of classes to consider in the networks output

    K = size(inputData,2);    % number of datapoints to be tested
    layerWidth = size(inputData,1);

    tLastIdx = size(Wdisc,2);
    T = (tLastIdx-1) * tStep;

    W = @(t) reshape(splineInt(t, tStep, Wdisc, tLastIdx), layerWidth, layerWidth);
    B = @(t) splineInt(t, tStep, Bdisc, tLastIdx);

    xODE    = @(t, x) sigma(W(t) * x + B(t));
    xODEvec = @(t, x) reshape(xODE(t, reshape(x, layerWidth, [])), [], 1);
    
    batchRemainder = mod(K, 100);

    errCount = 0;

    for currStart = 1:100:K-99
        x0 = inputData(:, currStart:currStart+99);     %  load input data, 100 images at once
        trueClass = classData(currStart:currStart+99)+1;  %  load class data and compensate for being 0-based

        xSol = ode45(xODEvec, [0, T], x0(:), odeSolverOpt);
        xT = reshape(deval(xSol,T), layerWidth, 100);

        [~, inferredClass] = max( xT(1:L, :) ); % find argmax of the network output
        errCount = errCount + sum((abs(trueClass-inferredClass)>0.5));
    end

    if batchRemainder>0
        x0 = inputData(:, end-batchRemainder+1 : end);       %  load input data, remainder
        trueClass = classData(end-batchRemainder+1 : end)+1; %  load class data and compensate for being 0-based

        xSol = ode45(xODEvec, [0, T], x0(:), odeSolverOpt);
        xT = reshape(deval(xSol,T), layerWidth, batchRemainder);

        [~, inferredClass] = max( xT(1:L, :) ); % find argmax of the network output
        errCount = errCount + sum((abs(trueClass-inferredClass)>0.5));
    end

    acc = 100 * (1 - errCount / K);

end