function sum_ECC = compute_sum_ECC(nonzero_p, nonzero_q, i, j, Ltilde, ECC)

n = size(Ltilde,1);
dx = size(ECC,1)/n;

idx_mat = zeros(n, n);
idx_mat(nonzero_p, nonzero_q) = 1;
idx_mat_pq = kron(idx_mat, ones(dx, dx));
ECC_pq = ECC.*idx_mat_pq;
ECC_pq = reshape(ECC_pq(ECC_pq~=0), length(nonzero_p)*dx, []);

idx_mat = zeros(n, n);
idx_mat(nonzero_p, j) = 1;
idx_mat_pj = kron(idx_mat, ones(dx, dx));
ECC_pj = ECC.*idx_mat_pj;
ECC_pj = reshape(ECC_pj(ECC_pj~=0), length(nonzero_p)*dx, []);
ECC_pj = repmat(ECC_pj , 1, length(nonzero_q));

idx_mat = zeros(n, n);
idx_mat(i, nonzero_q) = 1;
idx_mat_iq = kron(idx_mat, ones(dx, dx));
ECC_iq = ECC.*idx_mat_iq;
ECC_iq = reshape(ECC_iq(ECC_iq~=0), [], length(nonzero_q)*dx);
ECC_iq = repmat(ECC_iq, length(nonzero_p), 1);

idx_mat = zeros(n, n);
idx_mat(i, j) = 1;
idx_mat_ij = kron(idx_mat, ones(dx, dx));
ECC_ij = ECC.*idx_mat_ij;
ECC_ij = reshape(ECC_ij(ECC_ij~=0), dx, dx);
ECC_ij = repmat(ECC_ij, length(nonzero_p), length(nonzero_q));

sum_ECC = ECC_pq + ECC_pj + ECC_iq + ECC_ij;