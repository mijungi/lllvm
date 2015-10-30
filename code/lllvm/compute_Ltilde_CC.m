function Ltilde_pq_tiled = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, ECC, str)

n = size(Ltilde,1);
dx = size(ECC,1)/n;

if strcmp(str, 'pq')
    % computing:  sum_p sum_q Ltilde(p,q) eta_pi eta_qj <CpCq + CpCj + CiCq + CiCj>
    
    Ltilde_pq = Ltilde(nonzero_p, nonzero_q);
    Ltilde_pq_tiled = kron(Ltilde_pq, ones(dx, dx));
    
elseif strcmp(str, 'pj')
    % computing:  sum_p sum_q Ltilde(p,j) eta_pi eta_qj <CpCq + CpCj + CiCq + CiCj>
    
    Ltilde_pq = Ltilde(nonzero_p, j);
    Ltilde_pq_tiled = kron(repmat(Ltilde_pq, 1, length(nonzero_q)), ones(dx, dx));
    
elseif strcmp(str, 'iq')
    % computing:  sum_p sum_q Ltilde(i,q) eta_pi eta_qj <CpCq + CpCj + CiCq + CiCj>
    
    Ltilde_pq = Ltilde(i, nonzero_q);
    Ltilde_pq_tiled = kron(repmat(Ltilde_pq, length(nonzero_p), 1), ones(dx, dx));
    
elseif strcmp(str, 'ij')
    % computing:  sum_p sum_q Ltilde(i,j) eta_pi eta_qj <CpCq + CpCj + CiCq + CiCj>
    
    Ltilde_pq = Ltilde(i, j);
    Ltilde_pq_tiled = kron(repmat(Ltilde_pq, length(nonzero_p), length(nonzero_q)), ones(dx, dx));
    
else
    display('sorry, we do not understand what you mean');
end


% then compute the ECC part

% idx_mat = zeros(n, n);
% idx_mat(nonzero_p, nonzero_q) = 1;
% idx_mat_pq = kron(idx_mat, ones(dx, dx));
% ECC_pq = ECC.*idx_mat_pq;
% ECC_pq = reshape(ECC_pq(ECC_pq~=0), length(nonzero_p)*dx, []);
% 
% idx_mat = zeros(n, n);
% idx_mat(nonzero_p, j) = 1;
% idx_mat_pj = kron(idx_mat, ones(dx, dx));
% ECC_pj = ECC.*idx_mat_pj;
% ECC_pj = reshape(ECC_pj(ECC_pj~=0), length(nonzero_p)*dx, []);
% ECC_pj = repmat(ECC_pj , 1, length(nonzero_q));
% 
% idx_mat = zeros(n, n);
% idx_mat(i, nonzero_q) = 1;
% idx_mat_iq = kron(idx_mat, ones(dx, dx));
% ECC_iq = ECC.*idx_mat_iq;
% ECC_iq = reshape(ECC_iq(ECC_iq~=0), [], length(nonzero_q)*dx);
% ECC_iq = repmat(ECC_iq, length(nonzero_p), 1);
% 
% idx_mat = zeros(n, n);
% idx_mat(i, j) = 1;
% idx_mat_ij = kron(idx_mat, ones(dx, dx));
% ECC_ij = ECC.*idx_mat_ij;
% ECC_ij = reshape(ECC_ij(ECC_ij~=0), dx, dx);
% ECC_ij = repmat(ECC_ij, length(nonzero_p), length(nonzero_q));
% 
% sum_ECC = ECC_pq + ECC_pj + ECC_iq + ECC_ij;

% ECC_cell_for_sum = mat2cell(Ltilde_pq_tiled.*sum_ECC, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
% ECC_mat_for_sum = reshape([ECC_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
% Ltilde_ECC = sum(ECC_mat_for_sum, 3);