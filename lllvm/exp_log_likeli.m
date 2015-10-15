%% compute the expectation over x and c of log (conditional) likelihood
% i.e., Exp_q(x)q(C) log p(y|C,x)
% mijung wrote on the 15th of Oct, 2015

function lwb_likelihood = exp_log_likeli(mean_c, cov_c, Gamma, H, y, L, gamma, epsilon)

% Input:
% mean_c: dy x n*dx
% cov_c: n*dx x n*dx
% Gamma: n*dx x n*dx
% H: dy x n*dx 
% y: observations, dy x n
% L: laplacian matrix
% gamma: noise precision
% epsilon: a term added to eign val

% Output:
% lwb_likelihood : Exp_q(x)q(C) log p(y|C,x)


% essential quantities
dy = size(mean_c,1);
n = size(L,1);

% the terms that are quadratic in C
term_quad_C = -0.5*dy*trace(Gamma*cov_c) - 0.5*trace(Gamma*mean_c'*mean_c); 

% the term that is linear in C
term_lin_C = gamma*trace(mean_c'*H);

% the term that is constant in C
inv_sig_y = kron(epsilon*ones(n,1)*ones(1,n) + 2*gamma*L, eye(dy)); 
term_const_C = -0.5*y(:)'*inv_sig_y*y(:);

% normaliser term
term_normaliser = -0.5*n*dy*log(2*pi) + 0.5*dy*logdetns(epsilon*ones(n,1)*ones(1,n) + 2*gamma*L);

lwb_likelihood = term_quad_C + term_lin_C + term_const_C + term_normaliser;


