% to check new formulation with epsilon for adding epsilon*1*1'
% mijung wrote on Oct 1, 2015

clear all;
clc;
close all;

oldRng = rng();
seednum =151;

rng(seednum);

dx = 2; % dim(x)
dy = 3; % dim(y)
n = 5;  % number of datapoints

alpha = 1; % precision of X (zero-centering)
gamma = 2; % noise precision in likelihood
epsilon = 1e-3;

% (1) define a graph

howmanyneighbors = 2;

Y = randn(dy, n);
G = makeKnnG(Y, howmanyneighbors);

% adjacency matrix
% G = zeros(n,n,howmanyneighbors-1);
%
% for i=1:howmanyneighbors-1
%     G(:,:,i) = diag(ones(n-i,1),i);
%     G(:,:,i) = G(:,:,i) + G(:,:,i)';
% end
%
% G = sum(G,3);

h = sum(G,2);
% laplacian matrix
L = diag(h) - G;

%% (2) generate x and C

invOmega = kron(2*L,speye(dx));
x = mnormrnd(zeros(dx, n), 1, inv(alpha*eye(n) + 2*L) );
X = x(:);

U = eye(dy);
J = kron(ones(n,1), eye(dx));
ep_invOmega = epsilon*J*J' + invOmega;
v = mnormrnd(zeros(dy, dx*n), U, inv(ep_invOmega) );
C = v(:);

%% (3) generate y

invV = gamma*speye(dy);
ones_n = ones(n,1);

V_y = inv(epsilon*ones_n*ones_n' + 2*L*gamma);
E = generateE(G, reshape(C,dy,n*dx), invV, X(:));
E = reshape(E, dy, n);
mu_y = E*V_y;
Y = mnormrnd(mu_y, 1, V_y);
y = Y(:);

%% check prior on x

One_vec_n = ones(n,1);
invOmega = kron(2*L,speye(dx));

log_prob_x = log(mvnpdf(x(:)', zeros(1, n*dx), kron(inv(alpha*eye(n) + 2*L) , eye(dx))));

invPi = alpha*eye(n*dx) + invOmega;
exponent = -0.5*x(:)'*invPi*x(:);

dist = zeros(n, n);
for i=1:n
    for j=1:n
        eta = G(i,j);
        if eta~=0
            dist(i,j) = 0.5*sum((x(:,i) - x(:,j)).^2);
        end
    end
end
exponent_eq = -0.5*alpha*sum(sum(x.^2)) - sum(sum(dist));

log_prob_x_eq = exponent_eq -0.5*n*dx*log(2*pi) + 0.5*logdetns(invPi);

display(sprintf('-0.5 xtrp invPi x: %.3f', exponent ));
display(sprintf('exponent in eq 1: %.3f', exponent_eq ));

display(sprintf('log prob of x from matlab: %.3f', log_prob_x ));
display(sprintf('log prob of x from my derivation: %.3f', log_prob_x_eq ));

%% check prior on C (we don't need beta, so remove it from the model)

before_log = mvnpdf(v(:)', zeros(1, n*dx*dy), kron(inv(epsilon*J*J' + invOmega), eye(dy)));
log_prob_c = log(before_log);
exponent = -epsilon*0.5*trace(v*J*J'*v') - 0.5*trace(invOmega*v'*v);

dist = zeros(n, n);
for i=1:n
    for j=1:n
        %         [i j]
        eta = G(i,j);
        if eta~=0
            dist(i,j) = 0.5*sum(sum((v(:,1+(i-1)*dx:i*dx) - v(:,1+(j-1)*dx:j*dx)).^2));
        end
    end
end

exponent_eq = - sum(sum(dist)) -0.5*epsilon* sum(sum((v*J).^2));
log_prob_c_eq = exponent  - 0.5*n*dx*dy*log(2*pi) + 0.5*dy*logdetns(epsilon*J*J' + invOmega);

display(sprintf('exponent in eq 11: %.3f', exponent ));
display(sprintf('exponent in my derivation: %.3f', exponent_eq ));

display(sprintf('log prob of c from matlab: %.3f', log_prob_c ));
display(sprintf('log prob of c from my derivation: %.3f', log_prob_c_eq ));

%% check likelihood

invV = gamma*eye(dy);
Cmat = reshape(v, dy, dx, n);
E = compute_E(G, invV, Cmat, x, dx, dy, n);
E = reshape(E, dy, n);

M_y = E/(epsilon*ones_n*ones_n' + 2*gamma*L);
exponent = - 0.5*trace(inv(epsilon*ones_n*ones_n' + 2*gamma*L)*(Y-M_y)'*(Y-M_y));

dist = zeros(n, n);
for i=1:n
    for j=1:n
        %         [i j]
        eta = G(i,j);
        if eta~=0
            lin_dist = Y(:,j) - Y(:,i) - v(:, 1+(i-1)*dx:i*dx)*(x(:,j) - x(:,i));
            dist(i,j) = 0.5*gamma*lin_dist'*lin_dist;
        end
    end
end

f = zeros(n,n);
for i=1:n
    for j=1:n
        eta = G(i,j);
        if eta~=0
            f(i,j) = gamma*((x(:,j) - x(:,i))'*v(:, 1+(i-1)*dx:i*dx)')*((x(:,j) - x(:,i))'*v(:, 1+(i-1)*dx:i*dx)')';
        end
    end
end
f = sum(sum(f));

sig_y = inv(kron(epsilon*ones_n*ones_n' + gamma*2*L, eye(dy)));
mu_y = sig_y*E(:);

% [y(:)'*E(:) (mu_y'/sig_y)*y(:)]
% [f (mu_y'/sig_y)*mu_y]
exponent_eq_14 = - sum(sum(dist))-0.5*epsilon*sum(sum(Y,2).^2);
exponent_eq_15 =  -0.5*((Y(:)'/sig_y)*Y(:)-2*Y(:)'*E(:)+f);

display(sprintf('exponent_eq_14: %.3f', exponent_eq_14));
display(sprintf('exponent_eq_15: %.3f', exponent_eq_15 ));

log_prob_y = log(mvnpdf(Y(:)', mu_y', sig_y)); % eq. (22)

logZy = 0.5*((mu_y'/sig_y)*mu_y - f) + 0.5*logdetns(2*pi*sig_y);
log_prob_y_eq = exponent_eq_15 -logZy;
log_prob_y_eq_26 = -0.5*((y'/sig_y)*y - 2*y'*E(:) + E(:)'*sig_y*E(:)) - 0.5*logdetns(2*pi*sig_y);

display(sprintf('log prob of y in eq 20: %.3f', log_prob_y ));
display(sprintf('log prob of y  in eq 15: %.3f', log_prob_y_eq ));
display(sprintf('log prob of y  in eq 26: %.3f', log_prob_y_eq_26 ));


%% check if eq(27) and eq(28) are the same
% y'e = x'b
% and y'e = trace(C'inv(V)H)

ye = y'*E(:);
display(sprintf('y trp e: %.3f',ye ));

diagU = 1;
cov_c = 1e-6*eye(n*dx);
% [Ainv, b] = compute_invA_qC_and_B_qC(G, v, cov_c, invV, y, n, dy, dx, diagU);
[~, b] = compute_suffstat_A_b(G, v, cov_c, Y, gamma, epsilon);
xb = x(:)'*b;
display(sprintf('x trp b: %.3f',xb ));

cov_x = 1e-6*eye(n*dx);
% [~, H] = compute_invGamma_qX_and_H_qX(G, x(:), cov_x, y, n, dy, dx);
% [~, H] = compute_suffstat_Gamma_h(G, x(:), cov_x, Y, gamma, epsilon);
opt_dec = 1; % using decomposition
[Ltilde] = compute_Ltilde(L, epsilon, gamma, opt_dec);
[~, H]  = compute_suffstat_Gamma_h(G, x(:), cov_x, Y, gamma, Ltilde);

trCVinvH = trace(v'*invV*H);
display(sprintf('tra(CtrpinvVH): %.3f', trCVinvH ));

%% so far so good!
% now, check if eq (31) and eq (32) are the same.

C = reshape(C, dy, dx, n);
A_E = zeros(n*dy, n*dx);
for i=1:n
    Ci = C(:,:,i);
    non0_idx = find(G(i,:));
    sum_k_trm = gamma*(sum(C(:,:,non0_idx),3) + length(non0_idx)*Ci);
    
    for j=1:n
        eta_ij = G(i,j);
        Cj = C(:,:,j);
        if i==j
            del_ij = 1;
        else
            del_ij = 0;
        end
        
        A_E(1+(i-1)*dy: i*dy, 1+(j-1)*dx:j*dx) = -eta_ij*gamma*(Cj+Ci) + del_ij*sum_k_trm;
    end
end

etrpSig_ye = E(:)'*sig_y*E(:);
xtrpA_EtrpSig_yA_Ex = x(:)'*A_E'*sig_y*A_E*x(:);
display(sprintf('e trp Sig_y e : %.3f', etrpSig_ye ));
display(sprintf('xtrp A_Etrp Sig_y A_E x  : %.3f', xtrpA_EtrpSig_yA_Ex ));

Q = zeros(n*dx, n);
for i=1:n
    x_i = x(:,i);
    
    non0_idx = find(G(i,:));
    sum_k_trm = length(non0_idx)*x_i - sum(x(:,non0_idx),2);
    
    for j=1:n
        eta_ij = G(i,j);
        x_j = x(:,j);
        if i==j
            del_ij = 1;
        else
            del_ij = 0;
        end
        
        Q(1+(j-1)*dx:j*dx, i) = eta_ij*gamma*(x_i-x_j) + del_ij*gamma*sum_k_trm;
        
    end
end

Ltilde = inv(epsilon*ones_n*ones_n' + 2*gamma*L);
trQLtildeQtrpCtrpC = trace(Q*Ltilde*Q'*v'*v);
display(sprintf('trace(Q Ltilde Qtrp Ctrp C)  : %.3f', trQLtildeQtrpCtrpC ));

%% Then, see how to simplify A_E'*Sig_y*A_E , this will be beneficial in computing sufficient statistics later

% first see if eq 43 is correct:

% define A : =  A_E'*sig_y*A_E
A = zeros(n*dx, n*dx);
% size(Ainv)

for i=1:n
    
    for j=1:n
        
        Aij = zeros(dx, dx);
        for p=1:n
            for q=1:n
                Ltilde_p_q = Ltilde(p, q);
                A_E_p_i = A_E(1+(p-1)*dy:p*dy, 1+(i-1)*dx:i*dx);
                A_E_q_j = A_E(1+(q-1)*dy:q*dy, 1+(j-1)*dx:j*dx);
                Aij_tmp =  Ltilde_p_q*A_E_p_i'*A_E_q_j;
                Aij = Aij  + Aij_tmp;
            end
        end
        
        A(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Aij;
        
        
    end
    
end


AEsigyAE = A_E'*sig_y*A_E;
xtrpAEsigyAEx = x(:)'*AEsigyAE*x(:);

display(sprintf('x trp A_E trp sig_y A_E x  : %.3f', xtrpAEsigyAEx  ));

xtrpAx = x(:)'*A*x(:);
display(sprintf('x trp A x  : %.3f', xtrpAx  ));

% hence eq(43) is correct.


%% check if eq(52) is correct

Gamma = zeros(n*dx, n*dx);
% size(Ainv)

for i=1:n
    
    for j=1:n
        
        Gammaij = zeros(dx, dx);
        
        for k=1:n
            for kprim=1:n
                Ltilde_p_q = Ltilde(k, kprim);
                Gamma_k_i= Q(1+(i-1)*dx:i*dx, k);
                Gamma_kprim_j = Q(1+(j-1)*dx:j*dx, kprim);
                Gammaij_tmp =  Ltilde_p_q*Gamma_k_i*Gamma_kprim_j';
                Gammaij = Gammaij  + Gammaij_tmp;
            end
        end
        
        Gamma(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Gammaij;
        
        
    end
    
end

display(sprintf('trace(Q Ltilde Qtrp Ctrp C)  : %.3f', trQLtildeQtrpCtrpC ));

trGammaCtrpC = trace(Gamma*v'*v);
display(sprintf('trace(Gamma Ctrp C)  : %.3f', trGammaCtrpC ));

%% check if suff stat of Aij (eq 57) is correct

% mean_c : size of (dy, dx*n)
% cov_c : size of (dx*n, dx*n)

mean_c = v;
cov_c = 1e-6*eye(dx*n);

ECC = dy*cov_c + mean_c' * mean_c;

Aij_full = zeros(n*dx, n*dx);

for i=1:n
    
    %     Ci = C(:,:,i);
    
    for j=1:n
        
        %         Cj = C(:,:,j);
        Aij = zeros(dx, dx);
        
        for p=1:n
            
            eta_pi = G(p,i);
            %             Cp = C(:,:,p);
            
            if p==i
                del_pi =1;
            else
                del_pi =0;
            end
            
            
            for q=1:n
                
                eta_qj = G(q,j);
                %                 Cq = C(:,:,q);
                
                if q==j
                    del_qj = 1;
                else
                    del_qj = 0;
                end
                
                Ltilde_p_q = Ltilde(p, q);
                
                %                 trm1 = eta_pi*eta_qj*(Cp'*Cq + Cp'*Cj + Ci'*Cq + Ci'*Cj);
                CpCq = ECC(1+(p-1)*dx:p*dx, 1+(q-1)*dx:q*dx);
                CpCj = ECC(1+(p-1)*dx:p*dx, 1+(j-1)*dx:j*dx);
                CiCq = ECC(1+(i-1)*dx:i*dx, 1+(q-1)*dx:q*dx);
                CiCj = ECC(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx);
                
                trm1 = eta_pi*eta_qj*(CpCq + CpCj + CiCq + CiCj);
                
                q_nonzero = find(G(q,:));
                q_num_nonzero = sum(G(q,:));
                %                 Ckprim = sum(C(:,:,q_nonzero),3);
                %                 trm2 = - eta_pi*del_qj*(Cp'*Ckprim + q_num_nonzero*Cp'*Cq + Ci'*Ckprim +  q_num_nonzero*Ci'*Cq);
                
                CpCkprim = zeros(dx, dx);
                CiCkprim = zeros(dx, dx);
                for kprim=1:q_num_nonzero
                    kprim_idx = q_nonzero(kprim);
                    CpCkprim_tmp = ECC(1+(p-1)*dx:p*dx, 1+(kprim_idx-1)*dx:kprim_idx*dx);
                    CpCkprim = CpCkprim + CpCkprim_tmp;
                    
                    CiCkprim_tmp = ECC(1+(i-1)*dx:i*dx, 1+(kprim_idx-1)*dx:kprim_idx*dx);
                    CiCkprim = CiCkprim + CiCkprim_tmp;
                end
                
                trm2 = - eta_pi*del_qj*(CpCkprim + q_num_nonzero*CpCq + CiCkprim +  q_num_nonzero*CiCq);
                
                p_nonzero = find(G(p,:));
                p_num_nonzero = sum(G(p,:));
                %                 Ck = sum(C(:,:,p_nonzero),3);
                %                 trm3 = -eta_qj*del_pi*(Ck'*Cq + Ck'*Cj + p_num_nonzero*Cp'*Cq + p_num_nonzero*Cp'*Cj);
                
                CkCq = zeros(dx, dx);
                CkCj = zeros(dx, dx);
                for k=1:p_num_nonzero
                    k_idx = p_nonzero(k);
                    CkCq_tmp = ECC(1+(k_idx-1)*dx:k_idx*dx, 1+(q-1)*dx:q*dx);
                    CkCq = CkCq + CkCq_tmp;
                    
                    CkCj_tmp = ECC(1+(k_idx-1)*dx:k_idx*dx, 1+(j-1)*dx:j*dx);
                    CkCj = CkCj + CkCj_tmp;
                end
                
                trm3 = -eta_qj*del_pi*(CkCq + CkCj + p_num_nonzero*CpCq + p_num_nonzero*CpCj);
                
                %                 trm4 = del_pi*del_qj*(Ck'*Ckprim + q_num_nonzero*Ck'*Cq + p_num_nonzero*Cp'*Ckprim + p_num_nonzero*q_num_nonzero*Cp'*Cq);
                
                CkCkprim = zeros(dx, dx);
                for k=1:p_num_nonzero
                    for kprim =1:q_num_nonzero
                        k_idx = p_nonzero(k);
                        kprim_idx = q_nonzero(kprim);
                        CkCkprim_tmp = ECC(1+(k_idx-1)*dx:k_idx*dx, 1+(kprim_idx-1)*dx:kprim_idx*dx);
                        CkCkprim = CkCkprim + CkCkprim_tmp;
                    end
                end
                
                trm4 = del_pi*del_qj*(CkCkprim + q_num_nonzero*CkCq + p_num_nonzero*CpCkprim + p_num_nonzero*q_num_nonzero*CpCq);
                
                Aij_tmp =  gamma^2*Ltilde_p_q*(trm1 + trm2 + trm3 + trm4);
                Aij = Aij  + Aij_tmp;
                
            end
            
        end
        
        Aij_full(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Aij;
        
        
    end
    
end


xtrpAijfullx = x(:)'*Aij_full*x(:);
display(sprintf('x trp Aij full x  : %.3f', xtrpAijfullx  ));


%% eq (58)

% to remove the two for-loops for k and k'

Aij_pq_only = zeros(n*dx, n*dx);

tic;
for i=1:n
    
    for j=1:n
        
        Ltilde_i_j = Ltilde(i,j);
        Aij = zeros(dx, dx);
        
        for p=1:n
            
            eta_pi = G(p,i);
            Ltilde_p_j = Ltilde(p, j);
            
            for q=1:n
                
                eta_qj = G(q,j);
                
                Ltilde_p_q = Ltilde(p, q);
                Ltilde_q_i = Ltilde(q, i);
                
                CpCq = ECC(1+(p-1)*dx:p*dx, 1+(q-1)*dx:q*dx);
                CpCj = ECC(1+(p-1)*dx:p*dx, 1+(j-1)*dx:j*dx);
                CiCq = ECC(1+(i-1)*dx:i*dx, 1+(q-1)*dx:q*dx);
                CiCj = ECC(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx);
                
                Aij_tmp =  gamma^2*eta_pi*eta_qj*(Ltilde_p_q - Ltilde_p_j - Ltilde_q_i + Ltilde_i_j )*(CpCq + CpCj + CiCq + CiCj);
                Aij = Aij  + Aij_tmp;
                
            end
            
        end
        
        Aij_pq_only(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = Aij;
        
        
    end
    
end
toc;

xtrpAijpqonlyx = x(:)'*Aij_pq_only*x(:);
display(sprintf('x trp Aij pq only x  : %.3f', xtrpAijpqonlyx  ));

%% eq (58) without the for-loops for p and q

tic;

Aij_without_pq = zeros(n*dx, n*dx);

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
        
        Aij_without_pq(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = gamma^2*(Ltilde_ECC_pq - Ltilde_ECC_pj - Ltilde_ECC_iq + Ltilde_ECC_ij);
    end
end

Aij_without_pq = Aij_without_pq + Aij_without_pq';

for i=1:n
    j =i;
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
    
    Aij_without_pq(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = gamma^2*(Ltilde_ECC_pq - Ltilde_ECC_pj - Ltilde_ECC_iq + Ltilde_ECC_ij);
end

toc;

xtrpAijwithoutpqx = x(:)'*Aij_without_pq*x(:);
display(sprintf('x trp Aij without pq x  : %.3f', xtrpAijwithoutpqx  ));

%% this was for checking each term in Aij_without_pq

%
% i = 18;
% j = 1;
%
% nonzero_p = find(G(i,:));
% nonzero_q = find(G(j,:));
% whichterm = 'ij';
%
% %  'pq' computes : sum_p sum_q Ltilde(p,q) eta_pi eta_qj <CpCq + CpCj + CiCq + CiCj>
% Ltilde_ECC = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, ECC, whichterm);


% mean_c = v;
% cov_c = 1e-6*eye(dx*n);
%
% ECC = cov_c + mean_c' * mean_c;
%
% Ltilde_i_j = Ltilde(i,j);
% Aij = zeros(dx, dx);
%
% for p=1:n
%
%     eta_pi = G(p,i);
%     Ltilde_p_j = Ltilde(p, j);
%
%     for q=1:n
%
%         eta_qj = G(q,j);
%
%         Ltilde_p_q = Ltilde(p, q);
%         Ltilde_q_i = Ltilde(q, i);
%
%         CpCq = ECC(1+(p-1)*dx:p*dx, 1+(q-1)*dx:q*dx);
%         CpCj = ECC(1+(p-1)*dx:p*dx, 1+(j-1)*dx:j*dx);
%         CiCq = ECC(1+(i-1)*dx:i*dx, 1+(q-1)*dx:q*dx);
%         CiCj = ECC(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx);
%
%         if strcmp(whichterm, 'pq')
%             Aij_tmp =  eta_pi*eta_qj*(Ltilde_p_q )*(CpCq + CpCj + CiCq + CiCj);
%         elseif strcmp(whichterm, 'pj')
%             Aij_tmp =  eta_pi*eta_qj*(Ltilde_p_j)*(CpCq + CpCj + CiCq + CiCj);
%         elseif strcmp(whichterm, 'iq')
%             Aij_tmp =  eta_pi*eta_qj*(Ltilde_q_i)*(CpCq + CpCj + CiCq + CiCj);
%         elseif strcmp(whichterm, 'ij')
%             Aij_tmp =  eta_pi*eta_qj*(Ltilde_i_j)*(CpCq + CpCj + CiCq + CiCj);
%         else
%             display('oopsy, what are you doing?');
%         end
%         %         Aij_tmp =  gamma^2*eta_pi*eta_qj*(Ltilde_p_q - Ltilde_p_j - Ltilde_q_i + Ltilde_i_j )*(CpCq + CpCj + CiCq + CiCj);
%         Aij = Aij  + Aij_tmp;
%
%     end
%
% end
%
%
% norm(Ltilde_ECC-Aij)


%% check if eq 64 is correct

% mean_x : size of (dx*n, 1)
% cov_x : size of (dx*n, dx*n)

mean_x = x(:);
cov_x = 1e-6*eye(dx*n);

EXX = cov_x + mean_x * mean_x';

Gamma_without_pq = zeros(n*dx, n*dx);
tic;

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
        
        Gamma_without_pq(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = gamma^2*(Ltilde_EXX_pq - Ltilde_EXX_pj - Ltilde_EXX_iq + Ltilde_EXX_ij);
    end
end

Gamma_without_pq = Gamma_without_pq + Gamma_without_pq';

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
    
    Gamma_without_pq(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = gamma^2*(Ltilde_EXX_pq - Ltilde_EXX_pj - Ltilde_EXX_iq + Ltilde_EXX_ij);
    
end

toc;


trGamma_without_pqCtrpC = trace(Gamma_without_pq*v'*v);
display(sprintf('trace(Gamma without pq Ctrp C)  : %.3f', trGamma_without_pqCtrpC ));

%% check inverse of Ltilde

Ltilde = inv(epsilon*ones(n,1)*ones(1,n) + 2*gamma*L);

[U_L, D_L, V_L] = svd(L); % L := U_L*D_L*V_L'

D_L_inv = zeros(n,n);
D_L_inv(1:n-1, 1:n-1) = diag(1./diag(D_L(1:n-1, 1:n-1)));

Depsilon_inv = zeros(n,n);
sign_sin_val = V_L(:,end)'*U_L(:,end);
Depsilon_inv(n,n) = sign_sin_val *1./(epsilon*n);

invUsingSvd = V_L*Depsilon_inv*U_L' + 1./(2*gamma)*V_L*D_L_inv*U_L';

norm(Ltilde-invUsingSvd)

rng(oldRng);

%% try to avoid looping over i and j when computing <A> and <Gamma>

% first, let's compute A.

mean_c = v;
cov_c = 1e-6*eye(dx*n);

ECC = dy*cov_c + mean_c' * mean_c;

Ltilde = inv(epsilon*ones(n,1)*ones(1,n) + 2*gamma*L);

tic; 
ECTC_cell = mat2cell(ECC, dx*ones(1, n), dx*ones(1, n));

A = zeros(n*dx, n*dx);

for i=1:n
    for j=i+1:n
        
        Lamb_ij = Ltilde;
        Mu_ij = logical(sparse(G(:, i))*sparse(G(j, :)));
        
        % T1
        % dx x dx x #1's in Mu_ij
        T1_blocks = cat(3, ECTC_cell{Mu_ij});
        T1_ij = mat3d_times_vec(T1_blocks, Lamb_ij(Mu_ij));
        
        A(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = T1_ij;
    end
end


A = A + A';

% compute the diagonal
for i=1:n
        Lamb_ij = Ltilde;
        Mu_ij = logical(sparse(G(:, i))*sparse(G(i, :)));
        
        % T1
        % dx x dx x #1's in Mu_ij
        T1_blocks = cat(3, ECTC_cell{Mu_ij});
        T1_ij = mat3d_times_vec(T1_blocks, Lamb_ij(Mu_ij));
        
    A(1+(i-1)*dx:i*dx, 1+(i-1)*dx:i*dx) = T1_ij;
end
toc;

%%

tic; 

ECTC = repmat(ECC, n, n); 

idx_nonzero = kron(sparse(G(:))*sparse(G(:))', ones(dx, dx));
Ltilde_rep = repmat(kron(Ltilde, ones(dx,dx)), n,n);
Ltilde_nonzero = Ltilde_rep.*idx_nonzero; 

bigmat = (ECTC.*Ltilde_rep).*idx_nonzero;

JJ = sparse(kron(ones(n,1), eye(dx))); 
Jmat = sparse(kron(eye(n), JJ));
Aij_mat = Jmat'*bigmat*Jmat;

toc; 

% norm(Aij_mat-A)

%%
i=2;
j = 3;

Gij = G(i,:)'*G(j,:);
LtildeGij = Ltilde.*Gij;
kronLtilde = kron(LtildeGij, ones(dx,dx));
JJ = kron(ones(n,1), eye(dx)); 

JJ'*(ECC.*kronLtilde)*JJ





