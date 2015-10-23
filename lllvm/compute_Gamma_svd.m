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

% toc; 

%%
% tic; 
% for i=1:n
%     j_nonzero_idx = find(G(i,:));
%     j_nonzero_idx =  j_nonzero_idx(logical(j_nonzero_idx>i));
%     for jj=1:length(j_nonzero_idx)
%         j = j_nonzero_idx(jj);
%         Gamij_epsilon = compute_Gamij_svd(Ltilde_epsilon, G, EXX_cell, i, j);
%         Gamij_L = compute_Gamij_svd(Ltilde_L, G, EXX_cell, i, j);
%         Gamma_epsilon(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gamij_epsilon;
%         Gamma_L(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gamij_L;
%     end
% end
% Gamma_epsilon = Gamma_epsilon + Gamma_epsilon';
% Gamma_L = Gamma_L + Gamma_L';
% toc; 
% remember: Gamma = gamma^2*Gamma_epsilon + gamma/2*Gamma_epsilon
%%
%
% tic; 
use_scaled_identity_check = true;
%use_scaled_identity_check = false;
if use_scaled_identity_check && are_subblocks_scaled_identity(EXX_cell)
    % This subblocks of EXX_cell will become scaled identity after a few 
    % steps of EM. This if part is to make the computation faster.
    % One can always use the else part for everything (just slow).
    M = cell2diagmeans(EXX_cell);
    for i=1:n
        for j=i+1:n 
            Gamij_epsilon = compute_Gamij_scaled_iden(Ltilde_epsilon, G, M, dx, i, j);
            Gamij_L = compute_Gamij_scaled_iden(Ltilde_L, G, M, dx, i, j);
            Gamma_epsilon(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gamij_epsilon;
            Gamma_L(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gamij_L;

            % check with the full version
            %if true 
            %    Gamij_epsilon_full = compute_Gamij_svd(Ltilde_epsilon, G, EXX_cell, i, j);
            %    Gamij_L_full = compute_Gamij_svd(Ltilde_L, G, EXX_cell, i, j);
            %    display(sprintf('|Gamij_ep - Gamij_ep_full|_fro = %.5g', ...
            %        norm(Gamij_epsilon-Gamij_epsilon_full) ));
            %    display(sprintf('|Gamij_L - Gamij_L_full|_fro = %.5g', ...
            %        norm(Gamij_L-Gamij_L_full) ));
            %end
        end
    end
    Gamma_epsilon = Gamma_epsilon + Gamma_epsilon';
    Gamma_L = Gamma_L + Gamma_L';
    clear j
    % compute the diagonal
    for i=1:n
        Gamii_epsilon = compute_Gamij_scaled_iden(Ltilde_epsilon, G, M, dx, i, i);
        Gamii_L = compute_Gamij_scaled_iden(Ltilde_L, G, M, dx, i, i);
        Gamma_epsilon(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx)  = Gamii_epsilon;
        Gamma_L(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx) = Gamii_L;
    end

else
    % Subblocks of EXX_cell are not scaled identity.

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
    clear j
    % compute the diagonal
    for i=1:n
        Gamii_epsilon = compute_Gamij_svd(Ltilde_epsilon, G, EXX_cell, i, i);
        Gamii_L = compute_Gamij_svd(Ltilde_L, G, EXX_cell, i, i);
        Gamma_epsilon(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx)  = Gamii_epsilon;
        Gamma_L(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx) = Gamii_L;
    end
end

Gamma = gamma^2*Gamma_epsilon + gamma/2*Gamma_L;
end% end compute_Gamma_wittawat



function Gamij = compute_Gamij_scaled_iden(Ltilde, G, M, dx, i, j)
% This function assumes that each block in EXX_cell is a scaled identity matrix.
% This will happen at the latter stage of EM. The scales multipled to the identity 
% matrices are collected in M.
%
% - Ltilde: n x n 
% - G: graph n x n
% - M: n x n matrix. Assume that EXX_cell{i, j} = a_ij*I (scaled identity). 
%    Then EXX_scale(i, j) = a_ij. M is such that M_ij = a_ij.
%
%TODO: WJ: Parallelize over i, j. Should be possible.
%

    % mu. depend on i,j. n x n 0-1 sparse matrix.
    % All mu_xxx are logical.
    sgi = logical(sparse(G(:, i)));
    sgj = logical(sparse(G(j, :)));
    Mu_ij = logical(sparse(G(:, i))*sparse(G(j, :)));
    % lambda in the note. depend on i,j. #neighbours of i x #neighbours of j
    Lamb_ij = Ltilde(sgi, sgj) - bsxfun(@plus, Ltilde(sgi, j), Ltilde(i, sgj)) + Ltilde(i, j);

    % K1 
    K1_ij = M(Mu_ij)'*Lamb_ij(:);

    % K2
    W_p = sum(Lamb_ij, 2);
    K2_ij = -M(sgi, j)'*W_p;

    % K3. This has a similar structure as K2.
    W_q = sum(Lamb_ij, 1);
    K3_ij = -M(i, sgj)*W_q';

    % T4
    K4_ij = sum(W_p)*M(i, j);

    Gamij = (K1_ij+K2_ij+K3_ij+K4_ij)*eye(dx);

end

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
