function [ C, LU, LV] = mnormrnd( M, U, V, n, LU, LV )
%MNORMRND Generate random variates from a matrix normal distribution with 
%the mean M and parameters U, V.
%   - See https://en.wikipedia.org/wiki/Matrix_normal_distribution
%
% Inputs:
%   - LU, LV are optional. U = LU*LU'.
%   - U, V are mandatory and can be scalar which will be expanded to scaled
%   identity matrices. If U, V are originally scalar, supply the scalar values 
%   instead of forming scaled identities. 
%
% Return a 3d array of size size(M, 1) x size(M, 2) x n
%
% Common usage: 
%   - X = mnormrnd(M, U, V); % draw one sample
%   - X = mnormrnd(zeros(4, 5), 2, 3); 
%
% @author Wittawat Jitkrittum
% created: 8 May 2015
%

if nargin < 3
    error('M, U, V are mandatory');
end

[a, b] = size(M);
if nargin < 6
    if isscalar(V)
        LV = eye(b)*sqrt(V);
    else
        assert(all(size(V) == [b b]), 'size of V is not compatible with M');
        [RV, numv] = cholcov(V);
        LV = RV';
        if isnan(numv)
            error('V is not square symmetric.');
        elseif numv > 0
            error('V is not positive semi-definite. %d negative eigenvalues', numv);
        end
            
    end
end

if nargin < 5
    if isscalar(U)
        LU = eye(a)*sqrt(U);
    else
        assert(all(size(U) == [a a]), 'size of U is not compatible with M');
        [RU, numu] = cholcov(U);
        LU = RU';
        if isnan(numu)
            error('U is not square symmetric.');
        elseif numu > 0
            error('U is not positive semi-definite. %d negative eigenvalues', numu);
        end
    end
end

assert(all(size(LU) == [a, a]), 'incompatible size of LU');
assert(all(size(LV) == [b, b]), 'incompatible size of LV');

if nargin < 4 
    n = 1;
end

%   - Implementation relies on the result described in
%   http://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf
%   (see KRON 13) relating Cholesky factorization and Kronecker product.
%K = kron(LV, LU);
%Z = randn(numel(M), n);
%vecDraws = bsxfun(@plus, M(:),  K*Z);
%C = reshape(vecDraws, [a, b, n]);


% from https://en.wikipedia.org/wiki/Matrix_normal_distribution
C = zeros(a, b, n);
for i=1:n
    z = randn(a, b);
    C(:, :, i) = M + LU*z*LV';
end


end

