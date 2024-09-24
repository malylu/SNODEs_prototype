function [sigma, dsigma] = setActFun(funType, tResolution)
% [sigma, dsigma] = setActFun(funType, tResolution)
%
% setActFun returns an activation function and its derivative based on the
% parameter funType:
%  'sigmoid'  Sigmoid (logistic / soft step)
%  'tanh'     Hyperbolic tangent
%  'relu'     Rectified Linear Unit
%  'lrelu'    Leaky ReLU (factor 0.01 for negative values)
%  'srelu'    C^1-smooth modification of ReLU (non-standard)
%  'slrelu'   C^1-smooth modification of Leaky ReLU (non-standard)
%  'elu'      Exponential Linear Unit (alpha = 0.2, hence not smooth)
%  'selu'     C^1-smooth version of ELU (alpha = 1)
%  'softplus' C^Inf-smooth approximation of ReLU
%  otherwise  Identity
%
% The activation is scaled by the factor 1/tResolution to take into account
% the steplength in the Euler discretization of an sNODE.

if ~exist('tResolution', 'var')
    tResolution = 1;
end;

switch true
    case (strcmpi(funType, 'sigmoid')==1) % sigmoid (logistic/soft step)
        sigma  = @(t) 1./(1+exp(-t)) / tResolution;
        dsigma = @(t) 1./(2*cosh(t)+2) / tResolution;
    case (strcmpi(funType, 'tanh')==1) % tanh
        sigma  = @(t) tanh(t) / tResolution;
        dsigma = @(t) 2./(1+cosh(2*t)) / tResolution;
    case (strcmpi(funType, 'relu')==1) % ReLU
        sigma  = @(t) max(0, t) / tResolution;
        dsigma = @(t) +(t>0) / tResolution;
    case (strcmpi(funType, 'srelu')==1) % Smoothened ReLU (custom)
        smoothing = 0.1;
        sigma  = @(t) ((((t+smoothing) .* ((-smoothing<t) & (t<smoothing))).^2)/(4*smoothing) + t .* (t>=smoothing)) / tResolution;
        dsigma = @(t) (( (t+smoothing) .* ((-smoothing<t) & (t<smoothing))    )/(2*smoothing) + (t>=smoothing)) / tResolution;
    case (strcmpi(funType, 'lrelu')==1) % Leaky ReLU
        sigma  = @(t) max(0.01*t, t) / tResolution;
        dsigma = @(t) (0.01+0.99*(t>0)) / tResolution;
    case (strcmpi(funType, 'slrelu')==1) % Smoothened Leaky ReLU (custom)
        smoothing = 0.1;
        sigma  = @(t) (0.01*t + 0.99*((((t+smoothing) .* ((-smoothing<t) & (t<smoothing))).^2)/(4*smoothing) + t .* (t>=smoothing))) / tResolution;
        dsigma = @(t) (0.01   + 0.99*(( (t+smoothing) .* ((-smoothing<t) & (t<smoothing))    )/(2*smoothing) +      (t>=smoothing))) / tResolution;
    case (strcmpi(funType, 'elu')==1) % ELU
        alphaelu=0.2;  alphaelukomplement=1-alphaelu;
        sigma  = @(t) max(alphaelu*(exp(min(t,0))-1), t) / tResolution;
        dsigma = @(t) (alphaelu*exp(min(t,0))+alphaelukomplement*(t>0)) / tResolution;
    case (strcmpi(funType, 'selu')==1) % C^1 ELU, i.e., ELU with alpha=1
        sigma  = @(t) max(exp(min(t,0))-1, t) / tResolution;
        dsigma = @(t) exp(min(t,0)) / tResolution;
    case (strcmpi(funType, 'softplus')==1) % SoftPlus
        sigma  = @(t) (max(t, 0) + log1p(exp(-abs(t)))) / tResolution;
        dsigma = @(t) (1./(1+exp(-t))) / tResolution;
    otherwise % Identity
        sigma  = @(r,t) t / tResolution;
        dsigma = @(r,t) ones(size(t)) / tResolution;
end

end