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
%ECTC_cell = mat2cell(ECTC, dx*ones(1, n), dx*ones(1, n));

if nnz(G)/numel(G) <= 0.01
    % sparse logical G
    spLogG = logical(sparse(G));
else
    spLogG = logical(G);
end

A = compute_A_ultimate(Ltilde, spLogG, ECTC, gamma, dx);
%An3 = compute_A_n3(Ltilde, spLogG, ECTC, gamma, dx);
%display(sprintf('|A-An3| = %.5f', norm(A-An3, 'fro') ) );

end %end compute_A_wittawat(..)


function A = compute_A_ultimate(Ltilde, spLogG, ECTC, gamma, dx)
% An improved version of compute_A_n3. Memory: O(n^2)
    n = size(Ltilde, 1);

    AllCoeff = zeros(n, n, dx, dx);
    for r=1:dx 
        for s=1:dx
            % n x n covariance matrix. Each element i,j is E[(C_i^\top C_j)_{r,s}]
            % ECC_rs may not be positive definite if r != s. 
            ECC_rs = ECTC(r:dx:end, s:dx:end);
            Coeff_rs_nof = compute_coeff_A_nofactor(Ltilde, spLogG, ECC_rs, gamma);
            Coeff_rs = Coeff_rs_nof;
            AllCoeff(:, :, r, s) = Coeff_rs;
        end
    end
    AllCoeff = permute(AllCoeff, [3 4 1 2]);
    % See http://www.ee.columbia.edu/~marios/matlab/Matlab%20array%20manipulation%20tips%20and%20tricks.pdf
    % section 6.1.1 to understand what I am doing.
    AllCoeff = permute(AllCoeff, [1 3 2 4]);
    A = reshape(AllCoeff, [dx*n, dx*n]);

end

function A = compute_A_n3(Ltilde, spLogG, ECTC, gamma, dx)
% Compute <A> using O(n^3) memory. 
% Wittawat: This function is deprecated. Use compute_A_ultimate
% %

    % decompose Ltilde into La'*La
    [U, V] = eig(Ltilde);
    La = diag(sqrt(diag(V)))*U';
    n = size(La, 1);

    % TODO: We might be able to do SVD on ECTC once and pick the rows and columns 
    % of the SVD factors instead of doing SVD for each (r,s).
    AllCoeff = zeros(n, n, dx, dx);
    for r=1:dx 
        for s=1:dx
            % n x n covariance matrix. Each element i,j is E[(C_i^\top C_j)_{r,s}]
            % ECC_rs may not be positive definite if r != s. 
            ECC_rs = ECTC(r:dx:end, s:dx:end);
            Coeff_rs = compute_coeff_A(La, spLogG, ECC_rs, gamma, dx);
            AllCoeff(:, :, r, s) = Coeff_rs;
        end
    end
    AllCoeff = permute(AllCoeff, [3 4 1 2]);
    % See http://www.ee.columbia.edu/~marios/matlab/Matlab%20array%20manipulation%20tips%20and%20tricks.pdf
    % section 6.1.1 to understand what I am doing.
    AllCoeff = permute(AllCoeff, [1 3 2 4]);
    A = real(reshape(AllCoeff, [dx*n, dx*n]));
end


function Coeff = compute_coeff_A(La, spLogG, CCscale, gamma, dx)
% Much like compute_A_scaled_iden. However return the actual coefficients 
% before taking a Kronecker product with I_dx.
%  - La: A factor such that La'*La = Ltilde.
%  - CCscale is square but not necessarily symmetric.
%

    % decompose CCscale = U'*V 
    [u,s,v] = svd(CCscale);
    U = diag(sqrt(diag(s)))*u';
    V = diag(sqrt(diag(s)))*v';

    Ubar = A_factor(La, U, spLogG);
    Vbar = A_factor(La, V, spLogG);
    Coeff = (gamma^2)*(Ubar'*Vbar);
end

function Coeff = compute_coeff_A_nofactor(Ltilde, spLogG, CCscale, gamma)
% An improved version of compute_coeff_A. Memory requirement: O(n^2) 
% @author Wittawat on 12 Nov 2015
%  - Ltilde: 
%  - CCscale is square but not necessarily symmetric.
%

    % decompose CCscale = U'*V 
    UTV = CCscale;
    G = double(spLogG);
    B = G - diag(sum(G, 1));

    UTVG = UTV*G;
    LTLG = Ltilde*G;
    GTUTV = G'*UTV;
    GTL2 = G'*Ltilde;

    %line1 = B'*(Ltilde.*UTV)*B - B'*(Ltilde.*UTVG) + B'*(LTLG.*UTV);
    line1 = B'*( (Ltilde.*UTV)*B - (Ltilde.*UTVG) + (LTLG.*UTV) );
    line2 = -(Ltilde.*GTUTV)*B + Ltilde.*(G'*UTVG) - GTUTV.*LTLG;
    line3 = (GTL2.*(UTV))*B  -GTL2.*UTVG + UTV.*(G'*LTLG);

    Coeff = (gamma^2)*(line1 + line2 + line3);
end

function M = A_factor(La, U, spLogG )
% Return a matrix M : n*size(U, 1) x n
%
    % TODO: The following code requires O(n^3) storage requirement.     
    n = size(La, 1);
    nu = size(U, 1);
    Gd = double(spLogG);
    Degs = sum(Gd, 1);
    LaU = reshape(MatUtils.colOuterProduct(La, U), [n*nu, n]);
    %M2 = LaU*Gd;
    %M2 = M2 -bsxfun(@times, LaU, Degs);
    M2 = LaU*(Gd - diag(Degs));
    M2 = M2 -reshape(MatUtils.colOuterProduct(La, U*Gd), [n*nu, n]);
    M2 = M2 +reshape(MatUtils.colOuterProduct(La*Gd, U), [n*nu, n]);
    M = M2;

    %n = size(La, 1);
    %M = zeros(n*n, size(U, 1));
    %for i=1:n
    %    Gi = spLogG(i, :);
    %    Lai = La(:, i);
    %    Ui = U(:, i);

    %    s1 = La(:, Gi)*U(:, Gi)';
    %    s2 = -sum(Gi)*Lai*Ui';
    %    s3 = -Lai*sum(U(:, Gi), 2)';
    %    s4 = sum(La(:, Gi), 2)*Ui';

    %    M(:, i) = reshape(s1+s2+s3+s4, [n*size(U, 1), 1]);
    %end
    %display(sprintf('A_factor: |M-M2| = %.6f', norm(M-M2, 'fro') ));
end

function A = compute_A_scaled_iden(Ltilde, spLogG, CCscale, gamma, dx)
% This function assumes that each block in ECTC_cell is a scaled identity matrix.
% This will happen at the latter stage of EM. The scales multipled to the identity 
% matrices are collected in CCscale
%
% - Ltilde: n x n 
% - spLogG: sparse logical graph n x n
% - CCscale: n x n matrix. Assume that ECTC_cell{i, j} = a_ij*I (scaled identity). 
%    Then CCscale(i, j) = a_ij. 

    % decompose Ltilde into La'*La
    [U, V] = eig(Ltilde);
    La = diag(sqrt(diag(V)))*U';
    % decompose CCscale into T'*T
    [UC, VC] = eig(CCscale);
    T = diag(sqrt(diag(VC)))*UC';

    M = A_factor(La, T, spLogG);
    Coeff = real((gamma^2)*(M'*M));
    %clear M;
    A = kron(Coeff, eye(dx));
end

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
    %Mu_ij = bsxfun(@and, sgi, sgj);

    % T1 
    % dx x dx x #1's in Mu_ij
    T1_blocks = cat(3, ECTC_cell{sgi, sgj}); 
    T1_ij = mat3d_times_vec(T1_blocks, Lamb_ij(:));

    % T2
    % sparse nxn
    W_ij = Lamb_ij;
    W_p = sum(W_ij, 2);
    if all(abs(W_p) < 1e-10)
        T2_ij = zeros(dx, dx);
    else
        %Mu_p = logical(sum(Mu_ij, 2));
        % dx x dx x length of sgi
        T2_blocks = cat(3, ECTC_cell{sgi, j});
        T2_ij = mat3d_times_vec(T2_blocks, W_p);
    end

    W_q = sum(W_ij, 1);
    if all(abs(W_q) < 1e-10)
        T3_ij = zeros(dx, dx);
    else
        % T3. This has a similar structure as T2.
        %Mu_q = logical(sum(Mu_ij, 1));
        % dx x dx x length of sgj
        T3_blocks = cat(3, ECTC_cell{i, sgj});
        T3_ij = mat3d_times_vec(T3_blocks, W_q);
    end

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
