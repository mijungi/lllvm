function [ M ] = cell2diagmeans( C )
%CELL2DIAGMEANS Return the mean element of the diagonal entries of each square 
%dxd matrix in C. Assume C is a m x n cell array where C{i, j} is a d x d matrix.
% - Return a m x n matrix.
%

    assert(iscell(C), 'C is not a cell array');
    [d1, d2] = size(C{1, 1});
    assert(d1==d2, 'Elements in C are not square');
    d = d1;
    [m, n] = size(C);
    blocks3d = cat(3, C{:});
    nblocks = size(blocks3d, 3);
    d2 = d*d;
    vblocks = reshape(blocks3d, [d2, nblocks]);

    diagInd = false(1, d2);
    diagInd(1:(d+1):end) = true;
    diagMat = vblocks(diagInd, :);
    M = reshape(mean(diagMat, 1), [m, n]);
end

