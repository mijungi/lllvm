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
EXX_cell = mat2cell(EXX, dx*ones(1, n), dx*ones(1, n));
% compute Ltilde : inv(epsilon*ones_n*ones_n' + 2*gamma*L)
% h = sum(G,2);
% L = diag(h) - G;

% Ltilde = inv(epsilon*ones(n, n) + 2*gamma*L); 

% computing the upper off-diagonal first 
% tic;
% Gamma = zeros(n*dx, n*dx); 
if nnz(G)/numel(G) <= 0.01
    % sparse logical G
    spLogG = logical(sparse(G));
else
    spLogG = logical(G);
end
Gamma_L = zeros(n*dx, n*dx); 

% toc; 

%%
% tic; 
% for i=1:n
%     j_nonzero_idx = find(G(i,:));
%     j_nonzero_idx =  j_nonzero_idx(logical(j_nonzero_idx>i));
%     for jj=1:length(j_nonzero_idx)
%         j = j_nonzero_idx(jj);
%         Gamij_L = compute_Gamij_svd(Ltilde_L, G, EXX_cell, i, j);
%         Gamma_L(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gamij_L;
%     end
% end
% Gamma_L = Gamma_L + Gamma_L';
% toc; 
%%
%
% tic; 
use_scaled_identity_check = true;
n_thresh = 700;
%use_scaled_identity_check = false;
if use_scaled_identity_check && are_subblocks_scaled_identity(EXX_cell)
    % This subblocks of EXX_cell will become scaled identity after a few 
    % steps of EM. This if part is to make the computation faster.
    % One can always use the else part for everything (just slow).
    M = cell2diagmeans(EXX_cell);
    if n <= n_thresh
        Gamma_L = compute_Gam_scaled_iden(Ltilde_L, spLogG, M, dx);
        %display(sprintf('|Gam_L - GamL_cubic| = %.6f', norm(Gamma_L-GamL_cubic, 'fro') ));
    else
        for i=1:n
            for j=i+1:n 
                Gamij_L = compute_Gamij_scaled_iden(Ltilde_L, spLogG, M, dx, i, j);
                Gamma_L(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gamij_L;

                % check with the full version
                %if true 
                %    Gamij_L_full = compute_Gamij_svd2(Ltilde_L, spLogG, EXX_cell, i, j);
                %    display(sprintf('|Gamij_L - Gamij_L_full|_fro = %.5g', ...
                %        norm(Gamij_L-Gamij_L_full) ));
                %end
            end
        end
        Gamma_L = Gamma_L + Gamma_L';
        clear j
        % compute the diagonal
        for i=1:n
            Gamii_L = compute_Gamij_scaled_iden(Ltilde_L, spLogG, M, dx, i, i);
            Gamma_L(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx) = Gamii_L;
        end
    end

else
    % Subblocks of EXX_cell are not scaled identity.
    if n <= n_thresh

        Gam_Ln3 = compute_Gamma_n3(Ltilde_L, spLogG, EXX, dx);
        Gamma_L = Gam_Ln3;
    else

        for i=1:n
            for j=i+1:n
                Gamij_L = compute_Gamij_svd2(Ltilde_L, spLogG, EXX_cell, i, j);
                Gamma_L(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gamij_L;

                %Gamij_L2 = compute_Gamij_svd2(Ltilde_L, spLogG, EXX_cell, i, j);
                %display(sprintf('|Gamij_L - Gamij_L2| = %.6f', norm(Gamij_L-Gamij_L2)));
            end
        end
        Gamma_L = Gamma_L + Gamma_L';
        clear j
        % compute the diagonal
        for i=1:n
            Gamii_L = compute_Gamij_svd2(Ltilde_L, spLogG, EXX_cell, i, i);
            Gamma_L(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx) = Gamii_L;
        end
    end
    %display(sprintf('|Gamma_L - Gam_Ln3|_fro = %.6f', norm(Gamma_L-Gam_Ln3) ));

end

Gamma = gamma/2*Gamma_L;
end% end compute_Gamma_wittawat

function Gam = compute_Gamma_n3(Ltilde, spLogG, EXX, dx)
% Compute <Gamma> using O(n^3) memory. This is the fastest function but unfortunately 
% requires large memory. 
%
    % decompose Ltilde into La'*La
    [U, V] = eig(Ltilde);
    La = diag(sqrt(diag(V)))*U';
    n = size(La, 1);
    AllCoeff = zeros(n, n, dx, dx);
    for r=1:dx 
        for s=1:dx
            % n x n covariance matrix. Each element i,j is E[(x_i^\top x_j)_{r,s}]
            % EXX_rs may not be positive definite if r != s.
            EXX_rs = EXX(r:dx:end, s:dx:end);
            % n x n
            Coeff_rs = compute_coeff_Gamma(La, spLogG, EXX_rs, dx);
            AllCoeff(:, :, r, s) = Coeff_rs;
        end
    end
    AllCoeff = permute(AllCoeff, [3 4 1 2]);
    % See http://www.ee.columbia.edu/~marios/matlab/Matlab%20array%20manipulation%20tips%20and%20tricks.pdf
    % section 6.1.1 to understand what I am doing.
    AllCoeff = permute(AllCoeff, [1 3 2 4]);
    Gam = real(reshape(AllCoeff, [dx*n, dx*n]));
end

function Coeff = compute_coeff_Gamma(La, spLogG, C, dx)
% Much like compute_Gam_scaled_iden. However return the actual coefficients 
% before taking a Kronecker product with I_dx.
%  - La: A factor such that La'*La = Ltilde.
%  - C is square but not necessarily symmetric. 
%

    % decompose CCscale = U'*V 
    [u,s,v] = svd(C);
    U = diag(sqrt(diag(s)))*u';
    V = diag(sqrt(diag(s)))*v';

    Ubar = Gamma_factor(La, U, spLogG);
    Vbar = Gamma_factor(La, V, spLogG);
    Coeff = (Ubar'*Vbar);
end

function M = Gamma_factor(La, U, spLogG )
% Return a matrix M : n*size(U, 1) x n

    % TODO: The following code requires O(n^3) storage requirement. 
    n = size(La, 1);
    M = zeros(n*n, size(U, 1));
    for i=1:n
        Gi = spLogG(i, :);
        Lai = La(:, i);
        Ui = U(:, i);

        s1 = La(:, Gi)*U(:, Gi)';
        s2 = sum(Gi)*Lai*Ui';
        s3 = -Lai*sum(U(:, Gi), 2)';
        s4 = -sum(La(:, Gi), 2)*Ui';

        M(:, i) = reshape(s1+s2+s3+s4, [n*size(U, 1), 1]);
    end
end


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

    Fac = Gamma_factor(La, T, spLogG);
    Coeff = (Fac'*Fac);
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
