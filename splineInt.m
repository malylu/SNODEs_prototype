function yValues = splineInt(tQ, tStep, discData, maxIdx)
% yValues = splineInt(tQ, tStep, discData, maxIdx)
%
% splineInt is a simple implementation of cubic spline interpolation where
% the discretized data are given in equidistant points in the interval
% [0, maxIdx*tStep].
% Note: splineInt does NOT extrapolate outside of this interval.
% 
% tQ = query points in the interval [0, maxIdx*tStep], where the function
%      values are to be determined. tQ may be a row vector.
% tStep = distance between neighboring nodes in the given discretization
% discData = function data that are to be interpolated; each column in the
%            matrix discData contains function values at a single timepoint
%            discData(:, k) is the function value at time (k-1)*tStep
% maxIdx (optional) = index of the last column of the known discretized
%                     data

  if ~exist('maxIdx', 'var')
      maxIdx = size(discData,2);
  end
  
  tdiv = tQ/tStep;
  tidx1 = max(1, min(1+floor(tdiv), maxIdx-1));
  tidx2 = 1+tidx1;  
  tidx0 = max(1,tidx1-1);
  tidx3 = min(maxIdx, tidx2+1);
 
  tw0 = 1 + tdiv - tidx1;  % t
  tw1 = 1 - tw0;           % 1 - t
  tw00 = 1 + 2*tw0;        % 1 + 2t
  tw01 = 1 + 2*tw1;        % 3 - 2t
  
  p0 = discData(:,tidx0);
  p1 = discData(:,tidx1);
  p2 = discData(:,tidx2);
  p3 = discData(:,tidx3);
 
  m1 = (p2 - p0) ./ (tidx2 - tidx0); % approximate derivative (left)
  m2 = (p3 - p1) ./ (tidx3 - tidx1); % approximate derivative (right)

  yValues = tw00 .* tw1.^2 .* p1 + ...
            tw0 .* tw1.^2 .* m1 + ...
            tw0.^2 .* tw01 .* p2 - ...
            tw0.^2 .* tw1 .* m2;
end
