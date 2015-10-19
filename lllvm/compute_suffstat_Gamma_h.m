%% to compute the sufficient statistic of Gamma and h
% that appear in eq (54-55) for cov_c and mean_c
% Mijung wrote
% Oct 14, 2015

function [Gamma, H] = compute_suffstat_Gamma_h(G, mean_x, cov_x, Y, gamma, epsilon)

% inputs
% (1) G: adjacency matrix
% (2) mean_x : size of (dx*n, 1) 
% (3) cov_x : size of (dx*n, dx*n)
% (4) Y : observations, size of (dy, n)
% (5) gamma: noise precision in likelihood
% (6) epsilon: a constant term added to prior precision for C

% outputs
% (1) Gamma: eq(64)
% (2) H : eq(66)

% unpack essential quantities
[dy, n] = size(Y);
dx = size(cov_x,2)/n;

h = sum(G,2);


% (1) computing H

kronGones = kron(G,ones(1,dx));
kronINones = kron(eye(n),ones(1,dx)); 

Yd = reshape(Y,dy,n);
Xd = reshape(mean_x,dx,n);
e1 = repmat(Xd, n,1)' .* kronGones - ...
    repmat(reshape(Xd*G,n*dx,1)',n,1) .* kronINones;
e2 = (Yd*G)*( repmat(reshape(Xd,n*dx,1)',n,1) .* kronINones ) ;
e3 = ( repmat(h',dy,1).*Yd  )*( kronINones .* repmat(mean_x',n,1) );

H = Yd*e1 -  e2 + e3;

% (2) computing Gamma
%Gamma_mi  = compute_Gamma_mijung(G, mean_x, cov_x, Y, gamma, epsilon);
Gamma_wi  = compute_Gamma_wittawat(G, mean_x, cov_x, Y, gamma, epsilon );
%display(sprintf('norm(Gamma_mi - Gamma_wi, fro) = %.3g', norm(Gamma_mi-Gamma_wi, 'fro')));
%Gamma = Gamma_mi;
Gamma = Gamma_wi;
end

function [Gamma ] = compute_Gamma_wittawat(G, mean_x, cov_x, Y, gamma, epsilon )

% inputs
% (1) G: adjacency matrix
% (2) mean_x : size of (dx*n, 1) 
% (3) cov_x : size of (dx*n, dx*n)
% (4) Y : observations, size of (dy, n)
% (5) gamma: noise precision in likelihood
% (6) epsilon: a constant term added to prior precision for C

% outputs
% (1) Gamma: eq(64)

[dy, n] = size(Y);
dx = size(cov_x,2)/n;

EXX = cov_x + mean_x * mean_x';
% an nxn cell array. Each element is a dx x dx matrix.
EXX_cell = mat2cell(EXX, dx*ones(1, n), dx*ones(1, n));
% compute Ltilde : inv(epsilon*ones_n*ones_n' + 2*gamma*L)
h = sum(G,2);
L = diag(h) - G;
Ltilde = inv(epsilon*ones(n, n) + 2*gamma*L); 

Gamma = zeros(n*dx, n*dx); 

% computing the upper off-diagonal first 
for i=1:n
    for j=i+1:n
        Gamij = compute_Gamij_wittawat(Ltilde, G, EXX_cell, gamma, i, j);
        Gamma(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gamij;
    end
end
Gamma = Gamma + Gamma'; 
clear j
% compute the diagonal
for i=1:n
    Gamii = compute_Gamij_wittawat(Ltilde, G, EXX_cell, gamma, i, i);
    Gamma(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx) = Gamii;
end

end% end compute_Gamma_wittawat

function Gamij = compute_Gamij_wittawat(Ltilde, G, EXX_cell, gamma, i, j)
    % lambda in the note. depend on i,j. n x n
    Lamb_ij = Ltilde - bsxfun(@plus, Ltilde(:, j), Ltilde(i, :)) + Ltilde(i, j);
    % mu. depend on i,j. n x n 0-1 sparse matrix.
    % All mu_xxx are logical.
    Mu_ij = logical(sparse(G(:, i))*sparse(G(j, :)));

    % K1 
    % dx x dx x #1's in Mu_ij
    K1_blocks = cat(3, EXX_cell{Mu_ij}); 
    K1_ij = mat3d_times_vec(K1_blocks, Lamb_ij(Mu_ij));

    % K2
    % sparse nxn
    W_ij = Lamb_ij.*Mu_ij;
    Mu_p = logical(sum(Mu_ij, 2));
    % dx x dx x #1's in Mu_p
    K2_blocks = cat(3, EXX_cell{Mu_p, j});
    W_p = sum(W_ij, 2);
    K2_ij = -mat3d_times_vec(K2_blocks, W_p(Mu_p));

    % K3. This has a similar structure as K2.
    Mu_q = logical(sum(Mu_ij, 1));
    % dx x dx x #1's in Mu_q
    K3_blocks = cat(3, EXX_cell{i, Mu_q});
    W_q = sum(W_ij, 1);
    K3_ij = -mat3d_times_vec(K3_blocks, W_q(Mu_q));

    % T4
    EXX_ij = EXX_cell{i, j};
    K4_ij = sum(W_p)*EXX_ij;

    Gamij = gamma^2*(K1_ij+K2_ij+K3_ij+K4_ij);
end

function [Gamma ] = compute_Gamma_mijung(G, mean_x, cov_x, Y, gamma, epsilon )
% inputs
% (1) G: adjacency matrix
% (2) mean_x : size of (dx*n, 1) 
% (3) cov_x : size of (dx*n, dx*n)
% (4) Y : observations, size of (dy, n)
% (5) gamma: noise precision in likelihood
% (6) epsilon: a constant term added to prior precision for C

% outputs
% (1) Gamma: eq(64)

[dy, n] = size(Y);
dx = size(cov_x,2)/n;

EXX = cov_x + mean_x * mean_x';
% compute Ltilde : inv(epsilon*ones_n*ones_n' + 2*gamma*L)
h = sum(G,2);
L = diag(h) - G;
Ltilde = inv(epsilon*ones(n, n) + 2*gamma*L); 

Gamma = zeros(n*dx, n*dx); 

% computing the upper off-diagonal first 
for i=1:n
    for j=i+1:n
        nonzero_p = find(G(i,:));
        nonzero_q = find(G(j,:));
        
        sum_EXX = compute_sum_EXX(nonzero_p, nonzero_q, i, j, Ltilde, EXX); 

        Ltilde_pq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'pq');
        Ltilde_pj = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'pj');
        Ltilde_iq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'iq');
        Ltilde_ij = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'ij');
        
        EXX_cell_for_sum = mat2cell(Ltilde_pq.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_EXX_pq = sum(EXX_mat_for_sum, 3);
        
        EXX_cell_for_sum = mat2cell(Ltilde_pj.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_EXX_pj = sum(EXX_mat_for_sum, 3);
        
        EXX_cell_for_sum = mat2cell(Ltilde_iq.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_EXX_iq = sum(EXX_mat_for_sum, 3);
        
        EXX_cell_for_sum = mat2cell(Ltilde_ij.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_EXX_ij = sum(EXX_mat_for_sum, 3);

        Gamma(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = gamma^2*(Ltilde_EXX_pq - Ltilde_EXX_pj - Ltilde_EXX_iq + Ltilde_EXX_ij);
    end
end

Gamma = Gamma + Gamma'; 

% now computing the diagonal term

for i=1:n
    j=i;
    nonzero_p = find(G(i,:));
    nonzero_q = find(G(j,:));
    
    sum_EXX = compute_sum_EXX(nonzero_p, nonzero_q, i, j, Ltilde, EXX);
    
    Ltilde_pq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'pq');
    Ltilde_pj = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'pj');
    Ltilde_iq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'iq');
    Ltilde_ij = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'ij');
    
    EXX_cell_for_sum = mat2cell(Ltilde_pq.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_EXX_pq = sum(EXX_mat_for_sum, 3);
    
    EXX_cell_for_sum = mat2cell(Ltilde_pj.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_EXX_pj = sum(EXX_mat_for_sum, 3);
    
    EXX_cell_for_sum = mat2cell(Ltilde_iq.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_EXX_iq = sum(EXX_mat_for_sum, 3);
    
    EXX_cell_for_sum = mat2cell(Ltilde_ij.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_EXX_ij = sum(EXX_mat_for_sum, 3);
    
    Gamma(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = gamma^2*(Ltilde_EXX_pq - Ltilde_EXX_pj - Ltilde_EXX_iq + Ltilde_EXX_ij);
    
end

end
