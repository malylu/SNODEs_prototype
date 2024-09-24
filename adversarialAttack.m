% adversarialAttack
%
% The script loads an sNODE network trained by NCG using Euler's method
% (alternatively a ResNet analog, possibly trained by SGD)
%
% It loads test dataset from MNIST 14x14 and performs adversarial attack as
% described in our paper
%
% The script saves results of its attacks in a file (see variable
% outputFileNm). This output file contains following:
% advNorms:  l2-norms of adversarial perturbation
% advTest:   image data after the attack that caused misclassification
% advClass:  classification of the perturbed images
% inputTest: unperturbed image data
% classTest: classification of unperturbed images
%
% If an image is incorrectly classified before any attack, then attack is
% skipped and advNorms will contain the value 100 for that particular image
%
% If the adversarial attack is unsuccessful, then advNorms contains the
% l2-norm of the original image, while advTest will contain only zeros.

clear;

% Following parameters of the attack are readily available to be adjusted
maxIter = 30;
stepLen = 0.03;
kappa = 0.975;
verbose = 1;  % fileID for message output. Quiet mode if verbose <= 0
outputFileNm = 'attack_ncg_tanh_l2_run1.mat';
targeted = false; % true means that each original is subjected to 9 targeted attacks
                  % false means that an untargeted attack is done

% load mnist data
load('mnist14x14_uint16.mat', 'tstInput', 'tstClass');
inputTest = double(tstInput(:, 1:1000));
classTest = double(tstClass(:, 1:1000)) + 1;  % compensate for 1-based indexing in Matlab
noClasses = max(classTest(:));

imgCnt = min(length(classTest), size(inputTest,2));
imgSize = size(inputTest,1);

% load an sNODE-network
load('ncg_tanh_l2_run1.mat');
[sigma, dsigma] = setActFun(actFun, tResolution);
lastIdx = T * tResolution;

% transform input data according to the training settings
inputTest  = inputBgEnc + inputScaleFactor * (inputTest - inputShift);

% set up sNODE as well as the corresponding sensitivity ODE
xDE  = @(x,t)    sigma(Wcell{t+1}*x + Bcell{t+1});
xiDE = @(xi,x,t)dsigma(Wcell{t+1}*x + Bcell{t+1}) .* (Wcell{t+1} * xi);
    
% allocate memory for the attack results
advTest = zeros(196, length(classTest));
advNorms = vecnorm(inputTest);
advClass = classTest;

attackTimer = tic;
for imgNo=1:imgCnt
    minNorm = 100;
    x0 = inputTest(:, imgNo);     % load an image
    xt = x0;                      % initial condition for xDE
    xit = eye(196);               % initial condition for xiDE
    for j=0:149                   % solve the direct and sensitivity problem at once
        xt  = xt + xDE(xt, j);
        xit = xit + xiDE(xit, xt, j);
    end

    [~, xCls] = max(xt(1:noClasses));    % read the inferred class
    
    if (xCls~=classTest(imgNo))   % Skip images that are incorrectly classified
      advNorms(imgNo)=100;
      continue;
    end
    
    xit0 = xit;  % save xi(T) and x(T) from the original image
    xt0 = xt;    % they are reused in case of repeated targeted attack
        
    x0prev = x0;

    for currTgt = 1:1+9*targeted          % if targeted, the several distinct attacks are performed

        if targeted
            if (currTgt == xCls)  % skip the target if it is the current classification
                continue;
            else
                xit = xit0; 
                xt  = xt0;
            end
        end
        
        
        for iterNo=1:maxIter
            P10 = xit(1:noClasses,:).';
            z10 = xt(1:noClasses);
            z10diff = (z10(xCls) - kappa * z10).';  % can become negative if attack already succeeded
            P10diff  = kappa * P10 - P10(:, xCls);
            P10norms = vecnorm(P10diff);
            xi0norms = z10diff ./ P10norms;  % can become negative if attack already succeeded
            xi0norms(xCls) = Inf;
            %
            if targeted
                i = currTgt;
            else
                [~, i] = min(xi0norms);  % will skip Inf and NaN
            end
            %
            if isfinite(xi0norms(i))
                v = P10diff(:,i);
            else
                v = randn(size(xt));
            end

            xt = x0prev + (stepLen/norm(v))*v;
            xt = max(-1,min(0,xt));
            nxi = norm(xt - x0);

            x0prev = xt;
            xit = eye(196);

            for j=0:149
                xit = xit + xiDE(xit, xt, j);
                xt  = xt + xDE(xt, j);
            end
            [~, i] = max(xt(1:noClasses));
            if (i ~= xCls) && (minNorm > nxi)
                minNorm = nxi;
                advTest(:, imgNo) = x0prev;
                advNorms(imgNo) = nxi;
                advClass(imgNo) = i;
            end
        end
    end

    if mod(imgNo, 100 - 90*targeted) == 0
        save(['tmp_', outputFileNm], 'advNorms', 'advTest', 'advClass', 'classTest', 'inputTest');
        if verbose > 0
            fprintf(verbose, 'Image #%d has been processed. ', imgNo);
            fprintf(verbose, 'Elapsed time is %0.2f seconds\n', toc(attackTimer));
            attackTimer = tic;
        end
    end
end

save(outputFileNm, 'advNorms', 'advTest', 'advClass', 'classTest', 'inputTest');
delete(['tmp_', outputFileNm]);