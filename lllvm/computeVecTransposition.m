function [A_n] = computeVecTransposition(A, n)


% A = [[1:6]' 10.*[1:6]' 100.*[1:6]']; 
% 
% n = 3;

[size1, size2] = size(A);
Arshp = zeros(n, size2, size1/n);

for i=1:size1/n
    Arshp(:,:,i) = A((i-1)*n+1:i*n, :);
end

A_n = reshape(Arshp(:), n*size2, []);

%% Ahmad's comment: This is much faster for small matrix A
%%                  but unfortunately slower for bigger A
%% A_n = reshape(permute(reshape(A',size2,n,[]),[2 1 3]),size2*n,[]);
