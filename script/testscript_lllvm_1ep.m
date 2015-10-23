% to demonstrate how to use lllvm_1ep.m

clear all;
clc;
clf;

seednum = 1; 
oldRng = rng();
rng(seednum);

%% generate data from the true model

dx = 2; % dim(x)
dy = 3; % dim(y)
n = 50;  % number of datapoints

% parameters
alpha = rand; % precision of X (zero-centering)
gamma = rand; % noise precision in likelihood

epsilon = 1e-3;
howmanyneighbors = 5;

[vy, Y, vc, C, vx, X, G,  L, invOmega] = generatedata(dy, dx, n, alpha, gamma, epsilon, howmanyneighbors);

invPi = alpha*eye(n*dx) + invOmega;

%% to run lllvm_1ep.m

Yraw = Y; 
Y = reshape(Y,dy,n);
% subtract the mean
Y = bsxfun(@minus, Y, mean(Y, 2));

display(sprintf('rank of L is %3f', rank(L)));

if rank(L)< n-1
    display('sorry, choose a larger k so that rank(L)=n-1');
    break;
end

%% options to lllvm_1ep. Include initializations
op = struct();
op.seed = seednum;
op.max_em_iter = 50;
% absolute tolerance of the increase of the likelihood.
% If (like_i - like_{i-1} )  < abs_tol, stop EM.
op.abs_tol = 1e-1;
op.G = G;
op.dx = dx;
% The factor to be added to the prior covariance of C and likelihood. This must be positive and typically small.
op.epsilon = 1e-3;
% Intial value of alpha. Alpha appears in precision in the prior for x (low
% dimensional latent). This will be optimized in M steps.
op.alpha0 = rand;
% initial value of gamma.  V^-1 = gamma*I_dy where V is the covariance in the
% likelihood of  the observations Y.
op.gamma0 = rand;
%recorder = create_recorder('print_struct');
store_every_iter = 5;
only_cov_diag = false;
recorder = create_recorder_store_latent(store_every_iter, only_cov_diag);
op.recorder = recorder;

op.L = L; % laplacian matrix
op.invOmega = kron(2*op.L,eye(dx));
op.invPi0 = op.alpha0*eye(n*dx) + op.invOmega;

% initial value of the posterior mean and cov of c 
% op.cov_c = eye(dx*n) ;
% op.mean_c = randn(dy, dx*n);

J = kron(ones(n,1), eye(dx));
op.cov_c = inv(epsilon*J*J' + invOmega);
op.mean_c =  randn(dy, dx*n)*op.cov_c';

% true/false to store result. If true, record all variables updated in every
% EM iteration.
op.is_results_stored = true;

[results, op ] = lllvm_1ep(Y, op);

%% to visualise results

which_iter = 6; 

figure(1);
plotlearning(dx,dy,n,reshape(C,dy,n*dx),Yraw);
figure(2);
plotlearning(dx,dy,n,reshape(results.mean_c(:,which_iter),dy,n*dx),Yraw);

% figure(2);
% subplot(211);
% plot([vc(:) reshape(results.mean_c(:,which_iter), [], 1)]);
% subplot(212);
% plot([vx reshape(results.mean_x(:,which_iter), [], 1)]);
