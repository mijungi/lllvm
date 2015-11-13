function [Gamma,  Gamma_L ] = compute_Gamma_svd(G, EXX, Y, gamma, Ltilde_epsilon, Ltilde_L )
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
% (2) Gamma_L : second term in Gamma without gamma

[dy, n] = size(Y);
dx = size(EXX,1)/n;

% EXX = cov_x + mean_x * mean_x';
% an nxn cell array. Each element is a dx x dx matrix.
%EXX_cell = mat2cell(EXX, dx*ones(1, n), dx*ones(1, n));
% compute Ltilde : inv(epsilon*ones_n*ones_n' + 2*gamma*L)
%h = sum(G,2);
%L = diag(h) - G;

%Ltilde = inv(epsilon*ones(n, n) + 2*gamma*L); 

% computing the upper off-diagonal first 
% tic;
% Gamma = zeros(n*dx, n*dx); 
if nnz(G)/numel(G) <= 0.01
    % sparse logical G
    spLogG = logical(sparse(G));
else
    spLogG = logical(G);
end

%Gam_Ln3 = GammaUtils.compute_Gamma_n3(Ltilde_L, spLogG, EXX, dx);
Gam_ultimate = GammaUtils.compute_Gamma_ultimate(Ltilde_L, spLogG, EXX, dx);
% Wittawat: The two versions are not exactly the same. I do not know why. 
% I guess this is a numerical issue; although the Frobenius norm error is quite 
% large (<1.0). If I pass in Ltilde instead of Ltilde_L to both versions, 
% I get Fro. norm = 0. The error is probably due to the fact that Ltilde_L has 
% one eigenvalue = 0, and that the eigendecomposition used in 
% GammaUtils.compute_Gamma_n3 does not give exactly 0 for the eigenvalue.
%
%display(sprintf('|Gam_Ln3 - Gam_ultimate| = %.5f', norm(Gam_Ln3-Gam_ultimate, 'fro') ));
Gamma_L = Gam_ultimate;
%Gamma_L = Gam_Ln3;

Gamma = gamma/2*Gamma_L;
end% end compute_Gamma_wittawat


function Gam = compute_Gam_scaled_iden(Ltilde, spLogG, M, dx)
% This function assumes that each block in EXX_cell is a scaled identity matrix.
%
% - Ltilde: n x n 
% - spLogG: sparse logical graph n x n
% - M: n x n matrix. Assume that EXX_cell{i, j} = a_ij*I (scaled identity). 
%    Then M(i, j) = a_ij. 

    % decompose Ltilde into La'*La
    [U, V] = eig(Ltilde);
    La = diag(sqrt(diag(V)))*U';
    % decompose M into T'*T
    [UC, VC] = eig(M);
    T = diag(sqrt(diag(VC)))*UC';

    Fac = GammaUtils.Gamma_factor(La, T, spLogG);
    Coeff = real(Fac'*Fac);
    %clear Fac;
    Gam = kron(Coeff, eye(dx));
end

function Gamij = compute_Gamij_scaled_iden(Ltilde, spLogG, M, dx, i, j)
% This function assumes that each block in EXX_cell is a scaled identity matrix.
% This will happen at the latter stage of EM. The scales multipled to the identity 
% matrices are collected in M.
%
% - Ltilde: n x n 
% - spLogG: sparse logical graph n x n
% - M: n x n matrix. Assume that EXX_cell{i, j} = a_ij*I (scaled identity). 
%    Then EXX_scale(i, j) = a_ij. M is such that M_ij = a_ij.
%
%TODO: WJ: Parallelize over i, j. Should be possible.
%

    % mu. depend on i,j. n x n 0-1 sparse matrix.
    % All mu_xxx are logical.
    sgi = spLogG(:, i);
    sgj = spLogG(j, :);
    % lambda in the note. depend on i,j. #neighbours of i x #neighbours of j
    Lamb_ij = Ltilde(sgi, sgj) - bsxfun(@plus, Ltilde(sgi, j), Ltilde(i, sgj)) + Ltilde(i, j);
    %Mu_ij = logical(sparse(G(:, i))*sparse(G(j, :)));
    Mu_ij = bsxfun(@and, sgi, sgj);

    % K1 
    K1_ij = M(Mu_ij)'*Lamb_ij(:);

    % K2
    W_p = sum(Lamb_ij, 2);
    %if all(abs(W_p) < 1e-10)
    %    K2_ij = zeros(dx, dx);
    %else
        K2_ij = -M(sgi, j)'*W_p;
    %end

    % K3. This has a similar structure as K2.
    W_q = sum(Lamb_ij, 1);
    %if all(abs(W_q) < 1e-10)
    %    K3_ij = zeros(dx, dx);
    %else
        K3_ij = -M(i, sgj)*W_q';
    %end

    % T4
    K4_ij = sum(W_p)*M(i, j);

    Gamij = (K1_ij+K2_ij+K3_ij+K4_ij)*eye(dx);

end

function Gamij = compute_Gamij_svd2(Ltilde, spLogG, EXX_cell, i, j)
    sgi = spLogG(:, i);
    sgj = spLogG(j, :);
    dx = size(EXX_cell{1, 1}, 1);
    % All mu_xxx are logical.
    % mu. depend on i,j. n x n 0-1 sparse matrix.
    % lambda in the note. depend on i,j. n x n
    Lamb_ij = Ltilde(sgi, sgj) - bsxfun(@plus, Ltilde(sgi, j), Ltilde(i, sgj))+ Ltilde(i, j);
    %Mu_ij = bsxfun(@and, sgi, sgj);

    % K1 
    % dx x dx x nnz(sgi*sgj)
    K1_blocks = cat(3, EXX_cell{sgi, sgj}); 
    K1_ij = mat3d_times_vec(K1_blocks, Lamb_ij(:));

    % K2
    % sparse nxn
    W_ij = Lamb_ij;
    W_p = sum(W_ij, 2);
    if all(abs(W_p) < 1e-10)
        K2_ij = zeros(dx, dx);
    else
        %Mu_p = logical(sum(Mu_ij, 2));
        % dx x dx x length of sgi
        K2_blocks = cat(3, EXX_cell{sgi, j});
        K2_ij = -mat3d_times_vec(K2_blocks, W_p);
    end

    W_q = sum(W_ij, 1);
    if all(abs(W_q) < 1e-10)
        K3_ij = zeros(dx, dx);
    else
        % K3. This has a similar structure as K2.
        %Mu_q = logical(sum(Mu_ij, 1));
        % dx x dx x #1's in Mu_q
        K3_blocks = cat(3, EXX_cell{i, sgj});
        K3_ij = -mat3d_times_vec(K3_blocks, W_q);
    end

    % T4
    EXX_ij = EXX_cell{i, j};
    K4_ij = sum(W_p)*EXX_ij;

    Gamij = K1_ij+K2_ij+K3_ij+K4_ij;
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
