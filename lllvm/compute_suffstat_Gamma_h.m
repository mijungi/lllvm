%% to compute the sufficient statistic of Gamma and h
% that appear in eq (54-55) for cov_c and mean_c
% Mijung wrote
% Oct 14, 2015

function [Gamma, H,  Gamma_L] = compute_suffstat_Gamma_h(G, mean_x, cov_x, Y, gamma, Ltilde)

% inputs
% (1) G: adjacency matrix
% (2) mean_x : size of (dx*n, 1) 
% (3) cov_x : size of (dx*n, dx*n)
% (4) Y : observations, size of (dy, n)
% (5) gamma: noise precision in likelihood
% (6) epsilon: a constant term added to prior precision for C
% (7) Ltilde.Ltilde_epsilon, Ltilde.Ltilde_L

% outputs
% (1) Gamma: eq(64)
% (2) H : eq(66)
% (3) Gamma_L

% unpack essential quantities
[dy, n] = size(Y);
dx = size(cov_x,2)/n;

Ltilde_epsilon = Ltilde.Ltilde_epsilon;
Ltilde_L = Ltilde.Ltilde_L;

h = sum(G,2);

% (1) computing H

kronGones = kron(G,ones(1,dx));
kronINones = kron(eye(n),ones(1,dx)); 

Yd = reshape(Y,dy,n);
Xd = reshape(mean_x,dx,n);
e1 = repmat(Xd, n,1)' .* kronGones - ...
    repmat(reshape(Xd*G,n*dx,1)',n,1) .* kronINones;
e2 = (Yd*G)*( repmat(reshape(Xd,n*dx,1)',n,1) .* kronINones ) ;
e3 = ( repmat(h',dy,1).*Yd  )*( kronINones .* repmat(mean_x',n,1) );

H = Yd*e1 -  e2 + e3;

% (2) computing Gamma

EXX = cov_x + mean_x * mean_x';

[Gamma, Gamma_L ] = compute_Gamma_svd(G, EXX, Y, gamma, Ltilde_epsilon, Ltilde_L );

end
