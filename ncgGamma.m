function gamma = ncgGamma(cgmMethod, UWcell, UBcell, UWcellprev, UBcellprev, j)
% gamma = ncgGamma(cgmMethod, UWcell, UBcell, UWcellprev, UBcellprev, j)
%
% ncgGamma computes the conjugate gradient coefficient for the NCG-method
% when the current and the last steepest descent directions are known
%
% cgmMethod selects which conjugate gradient method is used:
%   'FR':     Fletcher--Reeves
%   'PR':     Polak--Ribiere (positive values only)
%   'FR-PR':  Combination of Fletcher--Reeves and Polak--Ribiere
%   'HS':     Hestenes--Stiefel
%   'DY':     Dai--Yuan
%  otherwise: only the current steepest descent is used, i.e., gamma = 0
%  
% UWcell, UBcell:  the current steepest descent direction for update of
%                  weights and biases. These are given in cells, where
%                  each cell contains E'_W(t) and E'_b(t), respectively,
%                  for one timepoint (timepoints are equidistantly
%                  distributed).
%               
% UWcellprev, UBcellprev:  the steepest descent direction of previous
%                          iteration. These have the same structure as
%                          UWcell and UBcell
%
% j:  counter of iterations of NCG-method. In the first iteration, one
%     always uses steepest descent as there is no previous E'.

if j<=1
    gamma = 0;
else
    switch true
        case (strcmpi(cgmMethod, 'FR')==1)
            % Fletcher-Reeves
            gammaT = sum(cellfun(@(UWcurr) sum(sum(UWcurr.^2)), UWcell)) + sum(cellfun(@(UBcurr) sum(UBcurr.^2), UBcell));
            gammaN = sum(cellfun(@(M) sum(sum(M.^2)), UWcellprev)) + sum(cellfun(@(M) sum(M.^2), UBcellprev));
            gamma = gammaT / gammaN;

        case (strcmpi(cgmMethod, 'PR')==1)
            % Polak-Ribiere, positive only
            gammaT = sum(cellfun(@(UWprev,UWcurr) sum(sum(UWcurr .* (UWcurr - UWprev))), UWcellprev, UWcell)) + sum(cellfun(@(UBprev,UBcurr) sum(UBcurr .* (UBcurr - UBprev)), UBcellprev, UBcell));
            gammaN = sum(cellfun(@(M) sum(sum(M.^2)), UWcellprev)) + sum(cellfun(@(M) sum(M.^2), UBcellprev));
            gamma = max(0, gammaT) / gammaN;

        case (strcmpi(cgmMethod, 'FR-PR')==1)
            % FR-PR combination as in Nocedal--Wright (5.48)
            gammaTFR = sum(cellfun(@(UWcurr) sum(sum(UWcurr.^2)), UWcell)) + sum(cellfun(@(UBcurr) sum(UBcurr.^2), UBcell));
            gammaTPR = gammaTFR - ( sum(cellfun(@(UWprev,UWcurr) sum(sum(UWcurr .* UWprev)), UWcellprev, UWcell)) + sum(cellfun(@(UBprev,UBcurr) sum(UBcurr .* UBprev), UBcellprev, UBcell)) );
            gammaN = sum(cellfun(@(M) sum(sum(M.^2)), UWcellprev)) + sum(cellfun(@(M) sum(M.^2), UBcellprev));
            gamma = max(-gammaTFR, min(gammaTFR, gammaTPR)) / gammaN;

        case (strcmpi(cgmMethod, 'HS')==1)
            % Hestenes-Stiefel
            gammaT = sum(cellfun(@(UWprev,UWcurr) sum(sum(UWcurr .* (UWcurr - UWprev))), UWcellprev, UWcell)) + sum(cellfun(@(UBprev,UBcurr) sum(UBcurr .* (UBcurr - UBprev)), UBcellprev, UBcell));
            gammaN = sum(cellfun(@(UWprev,UWcurr, dWprev) sum(sum(dWprev .* (UWcurr - UWprev))), UWcellprev, UWcell, dWcell)) + sum(cellfun(@(UBprev,UBcurr, dBprev) sum(dBprev .* (UBcurr - UBprev)), UBcellprev, UBcell, dBcell));
            gamma = gammaT / gammaN;

        case (strcmpi(cgmMethod, 'DY')==1)
            % Dai-Yuan
            gammaT = sum(cellfun(@(UWcurr) sum(sum(UWcurr.^2)), UWcell)) + sum(cellfun(@(UBcurr) sum(UBcurr.^2), UBcell));
            gammaN = sum(cellfun(@(UWprev,UWcurr, dWprev) sum(sum(dWprev .* (UWcurr - UWprev))), UWcellprev, UWcell, dWcell)) + sum(cellfun(@(UBprev,UBcurr, dBprev) sum(dBprev .* (UBcurr - UBprev)), UBcellprev, UBcell, dBcell));
            gamma = gammaT / gammaN;

        otherwise
            % Steepest Descent (no CGM)
            gamma = 0;
    end
end