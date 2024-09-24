function [Wout, Bout, tout]=initWB(initType, initFactor, matrixSize, T, tResolution);
% [Wout, Bout, tout] = initWB(initType, initFactor, matrixSize, T, tResolution);
%
% initWB returns sampled initial weights and biases based on the following
% parameters:
%  initType:   'constant', 'uniform', 'normal' (otherwise zero)
%  initFactor: factor in the initialization (Note: one should take care of
%              normalization by setting initFactor to an appropriate value,
%              either 1/sqrt(matrixSize) or 1/matrixSize (depends on type)
%  matrixSize: W is matrixSize-by-matrixSize, B is matrixSize-by-1
%  T:          final time (i.e., time of the output)
%  tResolution: number of intervals in the subdivision of a unit time
%               interval. The sampling takes this into account
% Output:
%  Wout: Weight matrices vectorized and saved in columns of a matrix
%        Each column Wout(:,k) contains the vectorization of matrix W(t),
%        where t = (k-1)/tResolution. Last column is W(T)
%
%  Bout: Bias vectors saved in  saved in columns of a matrix
%        Each column Bout(:,k) contains the vectorization of matrix B(t),
%        where t = (k-1)/tResolution. Last column is W(T)
%
%  tout: Vector (column) of timepoints used in the discretization

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

tout = linspace(0,T, T*tResolution + 1).';
Wout = interp1((0:T).', Wtmp, tout, 'spline').';
Bout = interp1((0:T).', Btmp, tout, 'spline').';
end