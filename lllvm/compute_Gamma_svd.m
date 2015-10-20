function [Gamma, Gamma_epsilon, Gamma_L ] = compute_Gamma_svd(G, EXX, Y, gamma, Ltilde_epsilon, Ltilde_L )
% Gamma_svd  = compute_Gamma_svd(G, mean_x, cov_x, Y, gamma, Ltilde_epsilon, Ltilde_L );

% inputs
% (1) G: adjacency matrix
% (2) mean_x : size of (dx*n, 1) 
% (3) cov_x : size of (dx*n, dx*n)
% (4) Y : observations, size of (dy, n)
% (5) gamma: noise precision in likelihood
% (6) epsilon: a constant term added to prior precision for C
% (7) Ltilde_epsilon
% (8) Ltilde_L

% outputs
% (1) Gamma: eq(64)
% (2) Gamma_epsilon: first term in Gamma without gamma
% (3) Gamma_L : second term in Gamma without gamma

[dy, n] = size(Y);
dx = size(EXX,1)/n;

% EXX = cov_x + mean_x * mean_x';
% an nxn cell array. Each element is a dx x dx matrix.
EXX_cell = mat2cell(EXX, dx*ones(1, n), dx*ones(1, n));
% compute Ltilde : inv(epsilon*ones_n*ones_n' + 2*gamma*L)
% h = sum(G,2);
% L = diag(h) - G;

% Ltilde = inv(epsilon*ones(n, n) + 2*gamma*L); 

% computing the upper off-diagonal first 
% tic;
% Gamma = zeros(n*dx, n*dx); 
Gamma_epsilon = zeros(n*dx, n*dx); 
Gamma_L = zeros(n*dx, n*dx); 

% remember: Gamma = gamma^2*Gamma_epsilon + gamma/2*Gamma_epsilon

for i=1:n
    for j=i+1:n
        Gamij_epsilon = compute_Gamij_svd(Ltilde_epsilon, G, EXX_cell, i, j);
        Gamij_L = compute_Gamij_svd(Ltilde_L, G, EXX_cell, i, j);
        Gamma_epsilon(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gamij_epsilon;
        Gamma_L(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gamij_L;
    end
end
Gamma_epsilon = Gamma_epsilon + Gamma_epsilon';
Gamma_L = Gamma_L + Gamma_L';

% Gamma = Gamma + Gamma'; 

% toc;

% tic;
% Gamma = zeros(n*dx, n*dx); 
% for i=1:n
% 
%     j_nonzero_idx = find(G(i,:));
%     j_nonzero_idx =  j_nonzero_idx(logical(j_nonzero_idx>i));
%     for jj=1:length(j_nonzero_idx)
%         j = j_nonzero_idx(jj); 
%         Gamij = compute_Gamij_wittawat(Ltilde, G, EXX_cell, gamma, i, j);
%         Gamma(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gamij;
%     end
% end
% Gamma = Gamma + Gamma'; 

% toc;

%%
% clear j

% compute the diagonal
for i=1:n
    Gamii_epsilon = compute_Gamij_svd(Ltilde_epsilon, G, EXX_cell, i, i);
    Gamii_L = compute_Gamij_svd(Ltilde_L, G, EXX_cell, i, i);
    Gamma_epsilon(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx)  = Gamii_epsilon;
    Gamma_L(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx) = Gamii_L;
%     Gamma(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx) =  gamma^2*Gamii_epsilon + gamma/2*Gamii_L;
end

Gamma = gamma^2*Gamma_epsilon + gamma/2*Gamma_L;


end% end compute_Gamma_wittawat

function Gamij = compute_Gamij_svd(Ltilde, G, EXX_cell, i, j)
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

    Gamij = K1_ij+K2_ij+K3_ij+K4_ij;
end