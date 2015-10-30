function [ gamma ] = heu_gamma_mstep( G, Y, epsilon, dx )
%HEU_GAMMA_MSTEP Compute a suitable gamma parameter using a heuristic based 
%on the Mstep update of gamma where q(x), q(c) are replaced with p(x), p(c) 
%respectively. 
% - G: n x n graph 
% - Y: dy x n data
%
%created: 27 Oct 2015
%

    n = size(G, 1);
    dy = size(Y, 1);
    L = diag(sum(G, 1)) - G;
    [U_L, D_L, V_L] = svd(L); % L := U_L*D_L*V_L'
    
    D_L_inv = zeros(n,n);
    D_L_inv(1:n-1, 1:n-1) = diag(1./diag(D_L(1:n-1, 1:n-1)));
    Ltilde_L = V_L*D_L_inv*U_L';

    eig_L = diag(D_L);
    % second smallest. The smallest one is 0.
    alpha = eig_L(end-1);
    %alpha = max(eig_L);
    invOmega = kron(2*L, eye(dx));

    % Pi = covariance of the prior of X 
    Pi = inv(alpha*eye(n*dx) + invOmega);

    % cov_c = prior right covariance matrix of C
    J = kron(ones(n, 1), eye(dx));
    cov_c = inv(epsilon*(J*J') + invOmega);
    spLogG = logical(G);
    Gamma = GammaUtils.compute_Gamma_n3(Ltilde_L, spLogG, Pi, dx );

    % compute the heuristic 
    gamma = dy*0.5*(n-1)/( 0.25*trace(Gamma*cov_c*dy) + trace(Y*L*Y') );


end

