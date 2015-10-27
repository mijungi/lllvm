%% update gamma and then compute log likelihood
% mijung wrote on the 19th of Oct, 2015

function [lwb_likelihood, gamma] = exp_log_likeli_update_gamma(mean_c, cov_c, H, y, L, epsilon, Ltilde,  QhatLtilde_LQhat )

% Input:
% mean_c: dy x n*dx
% cov_c: n*dx x n*dx
% H: dy x n*dx 
% y: observations, dy x n
% L: laplacian matrix
% epsilon: a term added to eign val
% Ltilde.Ltilde_eigL
% QhatLtilde_LQhat : Gamma_L

% Output:
% lwb_likelihood : Exp_q(x)q(C) log p(y|C,x)
% the updated gamma value

% essential quantities
dy = size(mean_c,1);
n = size(L,1);

% % the terms that are quadratic in C
% term_quad_C = -0.5*dy*trace(Gamma*cov_c) - 0.5*trace(Gamma*mean_c'*mean_c); 
% 
% % the term that is linear in C
% term_lin_C = gamma*trace(mean_c'*H);
% 
% % the term that is constant in C
% inv_sig_y = kron(epsilon*ones(n,1)*ones(1,n) + 2*gamma*L, eye(dy)); 
% term_const_C = -0.5*y(:)'*inv_sig_y*y(:);
% % -0.5*trace(epsilon*ones(n,1)*ones(1,n)*y'*y)-gamma*trace(L*y'*y)
% 
% % normaliser term
% term_normaliser = -0.5*n*dy*log(2*pi) + 0.5*dy*logdetns(epsilon*ones(n,1)*ones(1,n) + 2*gamma*L);
% 
% lwb_likelihood = term_quad_C + term_lin_C + term_const_C + term_normaliser;


%% new part


% Ltilde_epsilon = Ltilde.Ltilde_epsilon;
% Ltilde_L = Ltilde.Ltilde_L;
eigL = Ltilde.eigL;

% Qhat*Ltilde_epsilon*Qhat' 

% [~, QhatLtilde_epsilonQhat , QhatLtilde_LQhat ] = compute_Gamma_svd(G, EXX, y, gamma, Ltilde_epsilon, Ltilde_L ); 

% QhatLtilde_epsilonQhat = QhatLhatQhat(G, EXX, Ltilde_epsilon); 
% QhatLtilde_LQhat = QhatLhatQhat(G, EXX, Ltilde_L); 

% make sure if these two terms are the same
secondmoment = dy*cov_c + mean_c'*mean_c;
% l1 = -0.5*gamma^2*trace(QhatLtilde_epsilonQhat*secondmoment) - gamma/4*trace(QhatLtilde_LQhat*secondmoment);
% [l1 term_quad_C]
% 
% l2 = gamma*trace(mean_c'*H);
% [l2 term_lin_C]
% 
% l3 =  -0.5*trace(epsilon*ones(n,1)*ones(1,n)*y'*y) - gamma*trace(L*y'*y);
% [l3 term_const_C]
% 
% % l4 =  -0.5*n*dy*log(2*pi)  + dy*(sum(log(eigL(1:end-1))) + log(n*epsilon) + (n-1)*log(2*gamma))/2;
% l4 =  -0.5*n*dy*log(2*pi) + 0.5*dy*sum(log(eigL(1:end-1))) + 0.5*dy*log(n*epsilon) + 0.5*dy*(n-1)*log(2*gamma);
% [l4 term_normaliser]
% 
% [lwb_likelihood l1+l2+l3+l4]

% gamma update
QhatLtilde_LQhatsecondmoment = trace(QhatLtilde_LQhat*secondmoment);
mean_cH = trace(mean_c'*H); 
Lyy = trace(L*y'*y);

%min_obj =@(a) -(-a/4*QhatLtilde_LQhatsecondmoment + a*mean_cH - a*Lyy + 0.5*dy*(n-1)*log(2*a));
%opt = struct();
%opt.TolX = 1e-6;
%[gamma] = fminbnd(min_obj, 1e-5, 500, opt);
ana_gamma = -0.5*dy*(n-1)/(-0.25*QhatLtilde_LQhatsecondmoment + mean_cH - Lyy  );
gamma = ana_gamma;
%display(sprintf('|ana_gamma - gamma| = %.6f', abs(gamma-ana_gamma) ));
%display(sprintf('gamma: %.6f', gamma));
%gamma = 25;

l1 = -gamma/4*QhatLtilde_LQhatsecondmoment;
l2 = gamma*mean_cH;
l3 =  -0.5*trace(epsilon*ones(n,1)*ones(1,n)*y'*y) - gamma*Lyy;
l4 = -0.5*n*dy*log(2*pi) + 0.5*dy*sum(log(eigL(1:end-1))) + 0.5*dy*log(n*epsilon) + 0.5*dy*(n-1)*log(2*gamma); 

lwb_likelihood = l1+l2+l3+l4;



