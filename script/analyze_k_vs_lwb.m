function [  ] = analyze_k_vs_lwb()
%ANALYZE_K_VS_LWB This script is to analyze the effect of k (in the KNN graph) 
%on the lower bounds of the EM
%

seednum = 5; 
oldRng = rng();
rng(seednum);

%% generate data from the true model


dx = 2; % dim(x)
dy = 4;
n = 100;  % number of datapoints
data_k = 20;
epsilon = 1e-3;

% select data flag
%data_flag = 1; % 3D Gaussian
%data_flag = 2; % swiss roll with a hole
data_flag = 3; % swiss roll
[n, dy, Y] = getartificial(n, data_flag, data_k);
Y = reshape(Y, dy, n);

true_gamma = 1;
true_alpha = 1;
%[vy, Y, vc, C, vx, X, G,  L, invOmega] = generatedata(dy, dx, n, true_alpha, ... 
%   true_gamma, epsilon, data_k);
%Y = reshape(Y,dy,n);

% k's to try 
ks = 8:2:20;
lwbs_ks = zeros(1, length(ks));
gamma_ks = zeros(1, length(ks));

for i=1:length(ks)
    k = ks(i);
    G = makeKnnG(Y, k);
    L = diag(sum(G, 1)) - G;
    % zero out the diagonal. Has no effect anyway. Just to be consistent with the
    % math.
    G(1:(n+1):end) = 0;

    %% options to lllvm_1ep. Include initializations
    op = struct();
    op.seed = seednum;
    op.max_em_iter = 40;
    % absolute tolerance of the increase of the likelihood.
    % If (like_i - like_{i-1} )  < abs_tol, stop EM.
    op.abs_tol = 1e-1;
    op.G = G;
    op.dx = dx;
    % The factor to be added to the prior covariance of C and likelihood. This must be positive and typically small.
    op.epsilon = epsilon;
    % Intial value of alpha. Alpha appears in precision in the prior for x (low
    % dimensional latent). This will be optimized in M steps.
    op.alpha0 = 1;
    % initial value of gamma.  V^-1 = gamma*I_dy where V is the covariance in the
    % likelihood of  the observations Y.
    op.gamma0 = 1e-4;
    %recorder = create_recorder('print_struct');

    invOmega = kron(2*L,eye(dx));

    % initial value of the posterior mean and cov of c 
    % initializations optional
    op.cov_c0 = eye(dx*n) ;
    op.mean_c0 = randn(dy, dx*n);

    J = kron(ones(n,1), eye(dx));
    %op.cov_c0 = inv(epsilon*J*J' + invOmega);
    %op.mean_c0 =  randn(dy, dx*n)*op.cov_c0';

    display(sprintf('lllvm_1ep with k=%d', k));
    [results, op ] = lllvm_1ep(Y, op);
    lwbs_ks(i) = results.lwbs(end);
    gamma_ks(i) = results.gamma(end);
end

% plot the results 
% plot lwbs vs k
figure
plot(ks, lwbs_ks, '-o');
set(gca, 'FontSize', 16);
title(sprintf('lwbs vs k. seed=%d', seednum));
xlabel('k in KNN graph');
ylabel('final lower bound');

% plot gamma vs k

figure
plot(ks, gamma_ks, '-or');
set(gca, 'FontSize', 16);
title(sprintf('learned gamma vs k. seed=%d', seednum));
xlabel('k in KNN graph');
ylabel('learned gamma');

% export all variables to the base workspace.
allvars = who;
warning('off','putvar:overwrite');
putvar(allvars{:});

rng(oldRng);
end

