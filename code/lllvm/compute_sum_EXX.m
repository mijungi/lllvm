function sum_EXX = compute_sum_EXX(nonzero_p, nonzero_q, i, j, Ltilde, EXX)

n = size(Ltilde,1);
dx = size(EXX,1)/n;

idx_mat = zeros(n, n);
idx_mat(nonzero_p, nonzero_q) = 1;
idx_mat_pq = kron(idx_mat, ones(dx, dx));
EXX_pq = EXX.*idx_mat_pq;
EXX_pq = reshape(EXX_pq(EXX_pq~=0), length(nonzero_p)*dx, []);

idx_mat = zeros(n, n);
idx_mat(nonzero_p, j) = 1;
idx_mat_pj = kron(idx_mat, ones(dx, dx));
EXX_pj = EXX.*idx_mat_pj;
EXX_pj = reshape(EXX_pj(EXX_pj~=0), length(nonzero_p)*dx, []);
EXX_pj = repmat(EXX_pj , 1, length(nonzero_q));

idx_mat = zeros(n, n);
idx_mat(i, nonzero_q) = 1;
idx_mat_iq = kron(idx_mat, ones(dx, dx));
EXX_iq = EXX.*idx_mat_iq;
EXX_iq = reshape(EXX_iq(EXX_iq~=0), [], length(nonzero_q)*dx);
EXX_iq = repmat(EXX_iq, length(nonzero_p), 1);

idx_mat = zeros(n, n);
idx_mat(i, j) = 1;
idx_mat_ij = kron(idx_mat, ones(dx, dx));
EXX_ij = EXX.*idx_mat_ij;
EXX_ij = reshape(EXX_ij(EXX_ij~=0), dx, dx);
EXX_ij = repmat(EXX_ij, length(nonzero_p), length(nonzero_q));

sum_EXX = EXX_pq - EXX_pj - EXX_iq + EXX_ij;