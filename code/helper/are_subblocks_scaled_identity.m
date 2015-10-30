function [ is_scaled_identity ] = are_subblocks_scaled_identity(C)
% Return true if each element in the n x n cell array C is (approximately) a
% scaled identity matrix.  This function only *approximately* (controlled by
% a few variables at the beginning of the code) checks for this.  Expect each
% element in the cell array C to be a square matrix of the same size. 
%

    % A threshold on the mean absolute value of the off-diagonal entries. 
    % If it is higher than the threshold, return false. 
    mean_abs_offdiag_thresh = 1e-4;
    % A threshold on the standard deviation of the diagonal entries. 
    % It if is higher than the threshold, return false.
    std_diag_thresh = 1e-4;
    % check 50 blocks
    ncheck = min(51, size(C, 1)) - 1;


    dx = size(C{1,1}, 1);
    linInd = sub2ind(size(C), 1:ncheck, 2:(ncheck+1));
    blocks = cat(3, C{linInd});
    nblocks = size(blocks, 3);
    dx2 = dx*dx;
    vblocks = reshape(blocks, [dx2, nblocks]);

    diagInd = false(1, dx2);
    diagInd(1:(dx+1):end) = true;
    offDiagInd = ~diagInd;

    offDiagVec = vblocks(offDiagInd, :);
    offDiagVec = offDiagVec(:);
    diagMat = vblocks(diagInd, :);
    if mean(abs(offDiagVec)) > mean_abs_offdiag_thresh
        % mean. abs. of off-diagonal entries is too large.
        % Say no.
        is_scaled_identity = false;
    elseif mean(std(diagMat, 0, 1)) > std_diag_thresh 
        % std of the diagonal entries of each block is too large. Say no.
        is_scaled_identity = false;
    else 
        is_scaled_identity = true;
    end

end

