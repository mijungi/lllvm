% this script is for demonstrating how lllvm works with swiss roll data
% mijung wrote on 12th of nov, 2015

clear;
clc;

data_flag = 3; % swiss roll
n = 400; % total number of datapoints
k = 9;

% maximum number of EM iterations to perform
max_em_iter = 50;

seednum = 4;

oldRng = rng();
rng(seednum);

% generate artificial data

[n, dy, Y, G, dmat, col, truex] = getartificial(n, data_flag, k);

G = double(G);
dx = 2;

Yraw = Y;
Y = reshape(Y,dy,n);

h = sum(G,2);
L = diag(h) - G;
display(sprintf('rank of L is %3f', rank(L)));

if rank(L)< n-1
    display('sorry, choose a larger k so that rank(L)=n-1');
    break;
end

% options to lllvm_1ep. Include initializations
op = struct();
op.col = col;
op.seed = seednum;
op.max_em_iter = max_em_iter;
% absolute tolerance of the increase of the likelihood.
% If (like_i - like_{i-1} )  < abs_tol, stop EM.
op.abs_tol = 1e-3;
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

% initial value of the posterior mean and cov of c. (optional)
op.cov_c0 = eye(dx*n) ;
op.mean_c0 = randn(dy, dx*n);

[results, op ] = lllvm_1ep(Y, op);

save(strcat('swissroll_demo ', num2str(data_flag), 'seednum ', num2str(seednum), 'n= ', num2str(n), 'k =', num2str(k), '.mat'), 'results', 'Yraw', 'col', 'truex');

% change seed back
rng(oldRng);

%% visualise results

clear all;
clc;

k = 9;
seednum = 4;
data_flag = 3; 
load(strcat('swissroll_demo ', num2str(data_flag), 'seednum ', num2str(seednum), 'n= ', num2str(n), 'k =', num2str(k), '.mat'));

dx = 2;
n = 400;
dy = 3;

which = 50;

figure(3);
plotlearning(dx,dy,n,reshape(results.mean_c(:,which),dy,n*dx),Yraw, col);

figure(4);
subplot(211); plot(results.lwbs);
reshaped_mean_x = reshape(results.mean_x(:,which), dx, []);
subplot(212); scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, col, 'o', 'filled');

