function [ ] = run_lllvm_toy( )
%RUN_LLLVM_TOY A script to test lllvm.m on toy data. Export all variables to the 
%base workspace at the end. Does the same thing as in testEM.m
%

seed = 21;
oldRng = rng();
rng(seed);

%% generate data
dx = 2; % dim(x)
dy = 4; % dim(y)
n = 120;  % number of datapoints

% parameters
alpha = 1.0; % precision of X (zero-centering)

beta = 2;
invU = beta*eye(dy);
gamma = 2; % noise precision in likelihood
epsilon_1 = 0;
epsilon_2 = 0;

[vy, Y, vc, C, vx, X, G, invOmega, invV, invU, L, mu_y, V_y] = ...
    generatedata_Uc(dy, dx, n, alpha, invU, gamma, epsilon_1, epsilon_2);

% This struct is only for printing.
trueParams = struct();
trueParams.alpha = alpha;
trueParams.beta = beta;
trueParams.gamma = gamma;
%[vy, Y, vc, C, vx, X, G, invOmega, invPi, invV, invU, L, mu_y, cov_y] = generatedata_Uc_old(dy, dx, n, alpha, invU, gamma);

%% options to lllvm. Include initializations
op = struct();
op.seed = seed;
op.max_em_iter = 40;
% absolute tolerance of the increase of the likelihood.
% If (like_i - like_{i-1} )  < abs_tol, stop EM.
op.abs_tol = 1e-1;
op.G = G;
op.dx = dx;
% The factor to be multipled to an identity matrix to be added to the graph 
% Laplacian. This must be positive and typically small.
op.ep_laplacian = 1e-3;
% Intial value of alpha. Alpha appears in precision in the prior for x (low
% dimensional latent). This will be optimized in M steps.
op.alpha0 = 1;
% initial value of beta 
op.beta0 = 1;
% initial value of gamma.  V^-1 = gamma*I_dy where V is the covariance in the
% likelihood of  the observations Y.
op.gamma0 = 1;
%recorder = create_recorder('print_struct');
store_every_iter = 5;
only_cov_diag = false;
recorder = create_recorder_store_latent(store_every_iter, only_cov_diag);
op.recorder = recorder;

% initial value of the posterior covariance cov_x of X. Size: n*dx x n*dx.
% Appear in q(x) = N(x | mean_x, cov_x) where x is the vectorized n*dx x 1 vector.
L = diag(sum(G, 1)) - G + op.ep_laplacian*eye(n);
invOmega = kron(2*L, eye(dx));
invPi = op.alpha0*eye(n*dx) + invOmega;
op.cov_x0 = inv(eye(n*dx) + invPi) + eye(n*dx);

% initial value of the posterior mean of X. Size: n*dx x 1.
% Appear in q(x) = N(x | mean_x, cov_x) where x is the vectorized n*dx x 1 vector.
op.mean_x0 = op.cov_x0*randn(n*dx, 1) ;

%% Run lllvm 
[ results, op ] = lllvm(Y, op);
mean_c = results.mean_c;
mean_x = results.mean_x;

%% sanity check
% check if the first two moments of y generated from estimated mean, cov,
% params match the ones in original data

% (1) compute mean and cov of likelihood in eq. 8
invV = results.gamma*eye(dy);
E_estimates = compute_E(G, invV, mean_c, mean_x, dx, dy, n);
% This assumes that invV is a scaled identity.
cov_yest = kron(inv(2*L*results.gamma), eye(dy));
mu_yest = cov_yest*E_estimates;

% (2) compare them to mu_y and cov_y
figure(1);
subplot(211);
plot(1:dy*n, mu_y(:), 'k', 1:dy*n, mu_yest, 'r');
title('mean comparison');

subplot(212);
cov_y = kron(V_y, eye(dy));
plot(1:dy*n, diag(cov_y), 'k', 1:dy*n, diag(cov_yest), 'r');
title('cov comparison');

% plot lower bounds 
figure;
plot(results.lwbs, 'o-');
set(gca, 'fontsize', 16);
xlabel('EM iterations');
ylabel('variational lower bounds');

% rec_vars will contains all the recorded variables.
rec_vars = recorder();
% change seed back
rng(oldRng);

% export all variables to the base workspace.
allvars = who;
warning('off','putvar:overwrite');
putvar(allvars{:});

display(op);
display(results);
display(trueParams);

end

