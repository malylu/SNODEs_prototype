function [Wout, Bout, nW] = initWBcell (initType, initFactor, matrixSize, T, tResolution)
% [Wout, Bout, nW] = initWBcell(initType, initFactor, matrixSize, T, tResolution);
%
% initWBcell returns sampled initial weights and biases based on the
% following paraameters:
%  initType:   'constant', 'uniform', 'normal' (otherwise zero)
%  initFactor: factor in the initialization (Note: one should take care of
%              normalization by setting initFactor to an appropriate value,
%              either 1/sqrt(matrixSize) or 1/matrixSize (depends on type)
%  matrixSize: W is matrixSize-by-matrixSize, B is matrixSize-by-1
%  T:          final time (i.e., time of the output)
%  tResolution: number of intervals in the subdivision of a unit time
%               interval. The sampling takes this into account
% Output:
%  Wout: Weight matrices saved in cells. Each cell Wout{k} contains the
%        matrix W(t) where t = (k-1)/tResolution. Last cell gives
%        W(T - 1/tResolution)
%  Bout: Bias vectors saved in cells. Each cell Bout{k} contains the
%        column vector B(t) where t = (k-1)/tResolution. Last cell gives
%        B(T - 1/tResolution)
%  nW: Total number of parameters, in all W(t) and all B(t)

    switch true
        case (strcmpi(initType, 'constant')==1)
            Wtmp = initFactor*ones(T+1, matrixSize^2);
            Btmp = initFactor*ones(T+1, matrixSize);

        case (strcmpi(initType, 'uniform')==1)
            Wtmp = initFactor*(rand(T+1, matrixSize^2)-0.5);
            Btmp = initFactor*(rand(T+1, matrixSize)  -0.5);

        case (strcmpi(initType, 'normal')==1)
            Wtmp = initFactor*randn(T+1, matrixSize^2);
            Btmp = initFactor*randn(T+1, matrixSize);

        otherwise
            Wtmp = zeros(T+1, matrixSize^2);
            Btmp = zeros(T+1, matrixSize);

    end

    tout = linspace(0,T-1/tResolution, T*tResolution).';
    Wout2 = interp1((0:T).', Wtmp, tout, 'spline').';
    Bout2 = interp1((0:T).', Btmp, tout, 'spline').';
    nW = size(Wout2,1)*size(Wout2,2) + size(Bout2,1)*size(Bout2,2);

    Wout = cellfun(@(M) reshape(M, matrixSize, matrixSize), num2cell(Wout2,1), 'UniformOutput', false);
    Bout = num2cell(Bout2,1);
end