%% compute Dkl between two Gaussians
% mijung wrote on the 15th of Oct, 2015

function dkl = Dkl_Gaussians(m0, cov0, m1, cov1)

% Inputs:
% m0: mean of first Gaussian
% cov0: cov of first Gaussian
% m1: mean of second Gaussian
% cov1: cov of second Gaussian

% Output:
% dkl : Dkl(N0||N1) = 0.5*(trace(inv(cov1)*cov0) + (m1-m0)'*inv(cov1)*(m1-m0) - k + logdet(cov1*inv(cov0))) 

k = size(m0,1);
dkl = 0.5*(trace(inv(cov1)*cov0) + (m1-m0)'*inv(cov1)*(m1-m0) - k + logdetns(cov1*inv(cov0))) ; 