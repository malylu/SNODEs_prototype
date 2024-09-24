function vDisc = L2toW12(uDisc, tStep)
% vDisc = L2toW12(uDisc, tStep)
%
% L2toW12 takes in a discretization of a function in L2-dual pairing,
% where jth column in uDisc corresponds to the values at t=(j-1)*tStep,
% and outputs the corresponding function in W12-dual pairing
%
% Transformation from L2-descent direction to Sobolev-descent direction
% applied to UB and UW in the training script and UB and UW are to be 
% replaced by the output of this transformation.
% Note: Integration done using the midpoint rule.
%
% L2toW12 does NOT take in cells, but the weight matrix (or bias vector)
% needs to be vectorized at each timepoint and these vectorizations are
% stored in a large matrix where each column corresponds to the vectorized
% UW(t) (or UB(t)) at a single timepoint t.
%
% L2toW12 outputs the transformed UW (or UB) saved in a large matrix
% (again one column = one timepoint)
%
% Parameter tStep is supposed to be 1/tResolution
%
% If UB and UW are stored in cells, then use L2toW12cell instead.


    tLastIdx = size(uDisc, 2);
    T = tStep * (tLastIdx-1);
    Mdim = size(uDisc, 1);

    cosht = cosh(0 : tStep : T);

    coshmidp = cosh(tStep/2 : tStep : T);
    umidp = splineInt(tStep/2 : tStep : T, tStep, uDisc, tLastIdx);
    u1int = [zeros(Mdim, 1), cumsum(coshmidp .* umidp, 2)];
    u2int = [zeros(Mdim, 1), cumsum(coshmidp .* umidp(:,end:-1:1), 2)];

    vDisc = (tStep / sinh(T)) * (cosht(:,end:-1:1) .* u1int + cosht .* u2int(:,end:-1:1));
end
