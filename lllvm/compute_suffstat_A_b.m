%% to compute the sufficient statistic of A and b
% that appear in eq (47-48) for cov_x and mean_x
% Mijung wrote
% Oct 13, 2015

function [A, b] = compute_suffstat_A_b(G, mean_c, cov_c,  Y, gamma, epsilon)

% inputs
% (1) G: adjacency matrix
% (2) mean_c : size of (dy, dx*n) 
% (3) cov_c : size of (dx*n, dx*n)
% (4) Y : observations, size of (dy, n)
% (5) gamma: noise precision in likelihood
% (6) epsilon: a constant term added to prior precision for C

% outputs
% (1) A: eq(57)
% (2) b: eq(59)

% unpack essential quantities
[dy, n] = size(Y);
dx = size(mean_c,2)/n;

h = sum(G,2);

%(1) computing b
prodm = mean_c'*Y;
rowblocks = mat2cell( kron(G,ones(dx,1)) .* prodm , dx*ones(1,n),n);
b_without_gamma = ( reshape(sum(cat(3,rowblocks{:}),3),dx*n,1) - ...
    sum(prodm' .* kron(eye(n),ones(1,dx)) * kron(G,eye(dx)))' )...
    - ( sum( kron(G,ones(dx,1)) .* prodm ,2) - ...
    kron(h,ones(dx,1)) .* sum( kron(eye(n),ones(dx,1)) .* prodm ,2) );
b = gamma*b_without_gamma;


%(2) computing A
%A_mi = compute_A_mijung(G, mean_c, cov_c, Y, gamma, epsilon);
A_wi = compute_A_wittawat(G, mean_c, cov_c, Y, gamma, epsilon);
%display(sprintf('|A_mi - A_wi|_fro = %.3g', norm(A_mi - A_wi, 'fro')) );

%A = A_mi;
A = A_wi;
end

function A = compute_A_wittawat(G, mean_c, cov_c,  Y, gamma, epsilon)
% inputs
% (1) G: adjacency matrix
% (2) mean_c : size of (dy, dx*n) 
% (3) cov_c : size of (dx*n, dx*n)
% (4) Y : observations, size of (dy, n)
% (5) gamma: noise precision in likelihood
% (6) epsilon: a constant term added to prior precision for C

% output
% A: eq(57)
%
[dy, n] = size(Y);
dx = size(mean_c,2)/n;
% compute Ltilde : inv(epsilon*ones_n*ones_n' + 2*gamma*L)
h = sum(G,2);
L = diag(h) - G;
% TODO Wittawat: Eventually we will replace the computation of Ltilde by inv with 
% eigendecomposition.
Ltilde = inv(epsilon*ones(n, n) + 2*gamma*L); 

% E[C^T C]: a n*dx x n*dx matrix
ECTC = dy*cov_c + mean_c' * mean_c;
% an nxn cell array. Each element is a dx x dx matrix.
ECTC_cell = mat2cell(ECTC, dx*ones(1, n), dx*ones(1, n));

% sparse logical G
spLogG = logical(sparse(G));

%%
% A = zeros(n*dx, n*dx);
% % tic;
% for i=1:n
% %     compute this only for neighbouring j's of i
%     j_nonzero_idx = find(G(i,:));
%     j_nonzero_idx =  j_nonzero_idx(logical(j_nonzero_idx>i));
%     for jj=1:length(j_nonzero_idx)
%         j = j_nonzero_idx(jj); 
%         Aij = compute_Aij_wittawat2(Ltilde, G, ECTC_cell, gamma, i, j);
%         A(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Aij;
%     end
% end
% A = A + A'; 

%%
% toc;
% compute the upper part. 
use_scaled_identity_check = true;
%use_scaled_identity_check = false;
A = zeros(n*dx, n*dx);
if use_scaled_identity_check && are_subblocks_scaled_identity(ECTC_cell)
    % This subblocks of ECTC_cell will become scaled identity after a few 
    % steps of EM. This if part is to make the computation faster.
    % One can always use the else part for everything (just slow).
    CCscale = cell2diagmeans(ECTC_cell);
    for i=1:n
        for j=i+1:n
            Aij = compute_Aij_scaled_iden(Ltilde, spLogG, CCscale, gamma, dx, i, j);
            A(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Aij;

            % check with the full version
            %if true 
            %  Aij_full = compute_Aij_wittawat2(Ltilde, spLogG, ECTC_cell, gamma,  i, j);
            %  display(sprintf('|Aij - Aij_full|_fro = %.5g',  norm(Aij - Aij_full) ));
            %end
        end
    end
    A = A + A';
    % compute the diagonal
    for i=1:n
        Aii = compute_Aij_scaled_iden(Ltilde, spLogG, CCscale, gamma, dx, i, i);
        A(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx) = Aii;
    end

else

    % subblocks of ECTC_cell are not scaled identity.
    for i=1:n
        for j=i+1:n
            Aij = compute_Aij_wittawat2(Ltilde, spLogG, ECTC_cell, gamma, i, j);
            %display(sprintf('|Aij - Aij2| = %.6f', norm(Aij-Aij2)));

            A(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Aij;
        end
    end
    A = A + A';
    % compute the diagonal
    for i=1:n
        Aii = compute_Aij_wittawat2(Ltilde, spLogG, ECTC_cell, gamma, i, i);
        A(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx) = Aii;
    end
end

end %end compute_A_wittawat(..)

function Aij = compute_Aij_scaled_iden(Ltilde, spLogG, CCscale, gamma, dx, i, j)
% This function assumes that each block in ECTC_cell is a scaled identity matrix.
% This will happen at the latter stage of EM. The scales multipled to the identity 
% matrices are collected in CCscale
%
% - Ltilde: n x n 
% - spLogG: sparse logical graph n x n
% - CCscale: n x n matrix. Assume that ECTC_cell{i, j} = a_ij*I (scaled identity). 
%    Then CCscale(i, j) = a_ij. 
%

    %sgi = logical(sparse(G(:, i)));
    %sgj = logical(sparse(G(j, :)));
    sgi = spLogG(:, i);
    sgj = spLogG(j, :);

    % lambda in the note. depend on i,j. #neighbours of i x #neighbours of j
    Lamb_ij = Ltilde(sgi, sgj) - bsxfun(@plus, Ltilde(sgi, j), Ltilde(i, sgj)) + Ltilde(i, j);
    % mu. depend on i,j. n x n 0-1 sparse matrix.
    % All mu_xxx are logical.
    %Mu_ij = logical(sparse(G(:, i))*sparse(G(j, :)));
    Mu_ij = bsxfun(@and, sgi, sgj);

    % K1 
    T1_ij = CCscale(Mu_ij)'*Lamb_ij(:);

    % K2
    W_p = sum(Lamb_ij, 2);
    T2_ij = CCscale(sgi, j)'*W_p;

    % K3. This has a similar structure as K2.
    W_q = sum(Lamb_ij, 1);
    T3_ij = CCscale(i, sgj)*W_q';

    % T4
    T4_ij = sum(W_p)*CCscale(i, j);

    Aij = gamma^2*(T1_ij+T2_ij+T3_ij+T4_ij)*eye(dx);

end

function Aij = compute_Aij_wittawat2(Ltilde, spLogG, ECTC_cell, gamma, i, j)
    %sgi = logical(sparse(G(:, i)));
    %sgj = logical(sparse(G(j, :)));
    sgi = spLogG(:, i);
    sgj = spLogG(j, :);
    dx = size(ECTC_cell{1, 1}, 1);

    % lambda in the note. depend on i,j. n x n
    Lamb_ij = Ltilde(sgi, sgj) - bsxfun(@plus, Ltilde(sgi, j), Ltilde(i, sgj)) + Ltilde(i, j);
    %%
    % mu. depend on i,j. n x n 0-1 sparse matrix.
    % All mu_xxx are logical.
    %Mu_ij = logical(sparse(G(:, i))*sparse(G(j, :)));
    Mu_ij = bsxfun(@and, sgi, sgj);

    % T1 
    % dx x dx x #1's in Mu_ij
    T1_blocks = cat(3, ECTC_cell{Mu_ij}); 
    T1_ij = mat3d_times_vec(T1_blocks, Lamb_ij(:));

    % T2
    % sparse nxn
    W_ij = Lamb_ij;
    W_p = sum(W_ij, 2);
    if all(abs(W_p) < 1e-10)
        T2_ij = zeros(dx, dx);
    else
        Mu_p = logical(sum(Mu_ij, 2));
        % dx x dx x #1's in Mu_p
        T2_blocks = cat(3, ECTC_cell{Mu_p, j});
        T2_ij = mat3d_times_vec(T2_blocks, W_p);
    end

    W_q = sum(W_ij, 1);
    if all(abs(W_q) < 1e-10)
        T3_ij = zeros(dx, dx);
    else
        % T3. This has a similar structure as T2.
        Mu_q = logical(sum(Mu_ij, 1));
        % dx x dx x #1's in Mu_q
        T3_blocks = cat(3, ECTC_cell{i, Mu_q});
        T3_ij = mat3d_times_vec(T3_blocks, W_q);
    end

    % T4
    ECTC_ij = ECTC_cell{i, j};
    T4_ij = sum(W_p)*ECTC_ij;

    Aij = gamma^2*(T1_ij+T2_ij+T3_ij+T4_ij);
end

function Aij = compute_Aij_wittawat(Ltilde, G, ECTC_cell, gamma, i, j)
    % lambda in the note. depend on i,j. n x n
    Lamb_ij = Ltilde - bsxfun(@plus, Ltilde(:, j), Ltilde(i, :)) + Ltilde(i, j);

    %%
    % mu. depend on i,j. n x n 0-1 sparse matrix.
    % All mu_xxx are logical.
    Mu_ij = logical(sparse(G(:, i))*sparse(G(j, :)));

    % T1 
    % dx x dx x #1's in Mu_ij
    T1_blocks = cat(3, ECTC_cell{Mu_ij}); 
    T1_ij = mat3d_times_vec(T1_blocks, Lamb_ij(Mu_ij));

    % T2
    % sparse nxn
    W_ij = Lamb_ij.*Mu_ij;
    Mu_p = logical(sum(Mu_ij, 2));
    % dx x dx x #1's in Mu_p
    T2_blocks = cat(3, ECTC_cell{Mu_p, j});
    W_p = sum(W_ij, 2);
    T2_ij = mat3d_times_vec(T2_blocks, W_p(Mu_p));

    % T3. This has a similar structure as T2.
    Mu_q = logical(sum(Mu_ij, 1));
    % dx x dx x #1's in Mu_q
    T3_blocks = cat(3, ECTC_cell{i, Mu_q});
    W_q = sum(W_ij, 1);
    T3_ij = mat3d_times_vec(T3_blocks, W_q(Mu_q));

    % T4
    ECTC_ij = ECTC_cell{i, j};
    T4_ij = sum(W_p)*ECTC_ij;

    Aij = gamma^2*(T1_ij+T2_ij+T3_ij+T4_ij);
end



function A = compute_A_mijung(G, mean_c, cov_c,  Y, gamma, epsilon)

% inputs
% (1) G: adjacency matrix
% (2) mean_c : size of (dy, dx*n) 
% (3) cov_c : size of (dx*n, dx*n)
% (4) Y : observations, size of (dy, n)
% (5) gamma: noise precision in likelihood
% (6) epsilon: a constant term added to prior precision for C

% outputs
% A: eq(57)
%

[dy, n] = size(Y);
dx = size(mean_c,2)/n;
% compute Ltilde : inv(epsilon*ones_n*ones_n' + 2*gamma*L)
h = sum(G,2);
L = diag(h) - G;
ones_n = ones(n,1);
Ltilde = inv(epsilon*ones_n*ones_n' + 2*gamma*L); 

ECC = dy*cov_c + mean_c' * mean_c;

A = zeros(n*dx, n*dx); 
% compute the upper part first
for i=1:n
    for j=i+1:n
        
        nonzero_p = find(G(i,:));
        nonzero_q = find(G(j,:));
        
        sum_ECC = compute_sum_ECC(nonzero_p, nonzero_q, i, j, Ltilde, ECC); 

        Ltilde_pq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, ECC, 'pq');
        Ltilde_pj = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, ECC, 'pj');
        Ltilde_iq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, ECC, 'iq');
        Ltilde_ij = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, ECC, 'ij');
        
        ECC_cell_for_sum = mat2cell(Ltilde_pq.*sum_ECC, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        ECC_mat_for_sum = reshape([ECC_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_ECC_pq = sum(ECC_mat_for_sum, 3);
        
        ECC_cell_for_sum = mat2cell(Ltilde_pj.*sum_ECC, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        ECC_mat_for_sum = reshape([ECC_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_ECC_pj = sum(ECC_mat_for_sum, 3);
        
        ECC_cell_for_sum = mat2cell(Ltilde_iq.*sum_ECC, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        ECC_mat_for_sum = reshape([ECC_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_ECC_iq = sum(ECC_mat_for_sum, 3);
        
        ECC_cell_for_sum = mat2cell(Ltilde_ij.*sum_ECC, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        ECC_mat_for_sum = reshape([ECC_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_ECC_ij = sum(ECC_mat_for_sum, 3);
        
        A(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = gamma^2*(Ltilde_ECC_pq - Ltilde_ECC_pj - Ltilde_ECC_iq + Ltilde_ECC_ij);
    end
end

A = A + A'; % then symmetrise it.

% now compute the diagonal part
for i=1:n
    
    j = i; 
    nonzero_p = find(G(i,:));
    nonzero_q = find(G(j,:));
    
    sum_ECC = compute_sum_ECC(nonzero_p, nonzero_q, i, j, Ltilde, ECC);
    
    Ltilde_pq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, ECC, 'pq');
    Ltilde_pj = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, ECC, 'pj');
    Ltilde_iq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, ECC, 'iq');
    Ltilde_ij = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, ECC, 'ij');
    
    ECC_cell_for_sum = mat2cell(Ltilde_pq.*sum_ECC, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    ECC_mat_for_sum = reshape([ECC_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_ECC_pq = sum(ECC_mat_for_sum, 3);
    
    ECC_cell_for_sum = mat2cell(Ltilde_pj.*sum_ECC, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    ECC_mat_for_sum = reshape([ECC_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_ECC_pj = sum(ECC_mat_for_sum, 3);
    
    ECC_cell_for_sum = mat2cell(Ltilde_iq.*sum_ECC, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    ECC_mat_for_sum = reshape([ECC_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_ECC_iq = sum(ECC_mat_for_sum, 3);
    
    ECC_cell_for_sum = mat2cell(Ltilde_ij.*sum_ECC, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    ECC_mat_for_sum = reshape([ECC_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_ECC_ij = sum(ECC_mat_for_sum, 3);
    
    A(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = gamma^2*(Ltilde_ECC_pq - Ltilde_ECC_pj - Ltilde_ECC_iq + Ltilde_ECC_ij);
    
end

end
