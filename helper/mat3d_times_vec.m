function mul = mat3d_times_vec(M, v)
% M: a x b x c matrix 
% v: c vector. Can be a row or column vector.
% Output: a x b matrix = sum_i(M(:, :, i)*v(i))
%
[a, b, c] = size(M);
assert(c==length(v));
mul2d = reshape(M, [a*b, c])*v(:);
mul = reshape(mul2d, [a, b]);

end
