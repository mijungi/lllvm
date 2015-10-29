%% re-do figure 4

clear all;
clc;
close all

data_flag = 3;
k = 9;
seed =4;

load(strcat('fixingalpha_dataflag ', num2str(data_flag), 'seednum ', num2str(seed), 'k ', num2str(k), '.mat'));

%%
% plot the raw data
dx = size(truex,1);
n = size(results.mean_x,1)/dx;
dy = size(results.mean_c,1)/n/dx;

Y = reshape(Yraw,dy,n);

% figure(200);
% subplot(221); scatter3( Y(1,:) , Y(2,:) , Y(3,:) , [] , col , 'o'); grid off;
% subplot(223); scatter(truex(1,:), truex(2,:), 20, col, 'o', 'filled');

[sort_val, sort_idx] = sort(col);
idx_blue_pt = sort_idx(25);
idx_red_pt = sort_idx(end-25);

%% show them in 2D.

scatter(truex(1,:), truex(2,:), 60, col, 'o', 'filled');
hold on;
scatter(truex(1,idx_blue_pt), truex(2,idx_blue_pt), 100, 'bo', 'filled');
scatter(truex(1,idx_red_pt), truex(2,idx_red_pt), 100, 'ro', 'filled');

%% show them in 3D.

figure(2);
scatter3( Y(1,:) , Y(2,:) , Y(3,:), 40  , col , 'o', 'filled'); grid off;
hold on;
scatter3( Y(1,idx_blue_pt) , Y(2,idx_blue_pt) , Y(3,idx_blue_pt) , 100, 'bo', 'filled'); grid off;
scatter3( Y(1,idx_red_pt) , Y(2,idx_red_pt) , Y(3,idx_red_pt) , 100, 'ro', 'filled'); grid off;


%%

% uncomment this for shortcircuiting:
G(idx_blue_pt, idx_red_pt) = 1;
G(idx_red_pt, idx_blue_pt) = 1;
h = sum(G,2);
L = diag(h) - G;


%% options to lllvm_1ep. Include initializations

op = struct();
op.col = col;
op.seed = seed;
op.max_em_iter = 50;
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

save(strcat('shortcircuiting_dataflag ', num2str(data_flag), 'seednum ', num2str(seed), 'k ', num2str(k), '.mat'), 'results', 'Yraw', 'col', 'truex');


%%

reshaped_mean_x = reshape(results.mean_x(:,20), dx, []);
%    -541.7970: without shortcircuiting (epsilon = 0.005)
%  -488.2: with shortcircuiting
scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 80,  col, 'o', 'filled');
% title(' -2069.4: without shortcircuiting');
% hold on;
% scatter(reshaped_mean_x(1,idx_blue_pt), reshaped_mean_x(2,idx_blue_pt), 100, 'bo', 'filled');
% scatter(reshaped_mean_x(1,idx_red_pt), reshaped_mean_x(2,idx_red_pt), 100, 'ro', 'filled');


