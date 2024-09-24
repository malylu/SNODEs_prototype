function vDisc = L2toW12cell(uDisc, tStep)
% vDisc = L2toW12cell(uDisc, tStep)
%
% L2toW12 takes in a discretization of a function in L2-dual pairing,
% where jth cell in uDisc corresponds to the values at t=(j-1)*tStep,
% and outputs the corresponding function in W12-dual pairing
%
% Transformation from L2-descent direction to Sobolev-descent direction
% applied to UB and UW in the training script and UB and UW are to be 
% replaced by the output of this transformation.
% Note: Integration done using the midpoint rule.
%
% L2toW12 outputs the transformed UW (or UB) saved in a large matrix
% (again one cell = one timepoint)
%
% Parameter tStep is supposed to be 1/tResolution
%
% Both input and output of L2toW12cell is stored in cells. If UB and UW are
% stored as a matrix where 1 column = 1 timepoint, then use L2toW12 instead

% This function is really a mere wrapper for L2toW12

  uSize    = size(uDisc{1});
  uDiscVec = cell2mat(cellfun(@(M) M(:), uDisc, 'UniformOutput', false));
  vDiscVec = L2toW12(uDiscVec, tStep);
  vDisc    = cellfun(@(M) reshape(M, uSize), num2cell(vDiscVec, 1), 'UniformOutput', false);
end
